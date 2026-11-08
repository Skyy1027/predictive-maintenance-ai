"""Streamlit dashboard for the AI4I predictive-maintenance project."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MaintainAI | Predictive Maintenance", page_icon="⚙️", layout="wide")

FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

@st.cache_resource
def load_assets():
    bundle = joblib.load("models/predictive_maintenance_model.joblib")
    metrics = json.loads(Path("reports/model_metrics.json").read_text())
    data = pd.read_csv("data/ai4i2020.csv")
    return bundle, metrics, data

def recommendation(probability, threshold):
    if probability >= max(threshold, 0.70):
        return "🔴 MAINTENANCE NOW", "High predicted failure risk. Stop or inspect the machine before the next production cycle."
    if probability >= threshold:
        return "🟠 SCHEDULE INSPECTION", "Risk exceeds the model decision threshold. Plan maintenance at the next safe opportunity."
    return "🟢 CONTINUE & MONITOR", "Risk is below the decision threshold. Continue operating and monitor the next sensor readings."

try:
    bundle, metrics, data = load_assets()
except FileNotFoundError:
    st.error("Model files are missing. Run `python3 train_model.py` first.")
    st.stop()

st.title("⚙️ MaintainAI")
st.caption("Predictive maintenance decision support | UCI AI4I 2020 dataset | Undergraduate project demo")
st.warning("Decision-support prototype: use a qualified engineer and site safety procedures before taking operational action.", icon="⚠️")

tab_predict, tab_batch, tab_insights, tab_data = st.tabs(["Live prediction", "Batch screening", "Model evidence", "Dataset"])

with tab_predict:
    st.subheader("Machine health assessment")
    st.write("Enter the latest sensor values. The model estimates the likelihood of a machine failure and converts it into an operational recommendation.")
    c1, c2, c3 = st.columns(3)
    with c1:
        machine_type = st.selectbox("Product quality type", ["L", "M", "H"], help="L = low, M = medium, H = high quality variant")
        air = st.number_input("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
    with c2:
        process = st.number_input("Process temperature (K)", 305.0, 320.0, 310.0, 0.1)
        rpm = st.number_input("Rotational speed (rpm)", 1000, 3000, 1500, 1)
    with c3:
        torque = st.number_input("Torque (Nm)", 0.0, 100.0, 40.0, 0.1)
        wear = st.number_input("Tool wear (min)", 0, 300, 100, 1)
    record = pd.DataFrame([[machine_type, air, process, rpm, torque, wear]], columns=FEATURES)
    probability = float(bundle["pipeline"].predict_proba(record)[0, 1])
    label, message = recommendation(probability, bundle["threshold"])
    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted failure probability", f"{probability:.1%}")
        st.caption(f"Model action threshold: {bundle['threshold']:.1%}")
    with right:
        st.subheader(label)
        st.write(message)
    st.progress(min(int(probability * 100), 100), text=f"Failure risk score: {probability:.1%}")

with tab_batch:
    st.subheader("Screen multiple machines")
    st.write("Upload a CSV with these columns: " + ", ".join(FEATURES))
    upload = st.file_uploader("CSV file", type="csv")
    if upload:
        batch = pd.read_csv(upload)
        missing = set(FEATURES) - set(batch.columns)
        if missing:
            st.error("Missing columns: " + ", ".join(sorted(missing)))
        else:
            batch["Failure probability"] = bundle["pipeline"].predict_proba(batch[FEATURES])[:, 1]
            batch["Recommendation"] = batch["Failure probability"].map(lambda p: recommendation(p, bundle["threshold"])[0])
            st.dataframe(batch, width="stretch")
            st.download_button("Download screened results", batch.to_csv(index=False), "maintenance_screening.csv", "text/csv")

with tab_insights:
    st.subheader("Hold-out test performance")
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    b.metric("Average precision", f"{metrics['average_precision']:.3f}")
    c.metric("Recall", f"{metrics['recall']:.1%}")
    d.metric("F1 score", f"{metrics['f1']:.3f}")
    st.caption("Metrics are calculated on a stratified 20% hold-out test set. The threshold was selected on that same hold-out set to maximise F1; validate it independently before deployment.")
    importance = pd.DataFrame(metrics["feature_importance"]).head(10).set_index("feature")
    st.subheader("Feature importance")
    st.bar_chart(importance, horizontal=True)
    cm = metrics["confusion_matrix"]
    st.write(f"Confusion matrix at threshold {metrics['threshold']:.1%}: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

with tab_data:
    st.subheader("Training data")
    st.write(f"{len(data):,} labelled observations from the UCI AI4I 2020 Predictive Maintenance Dataset. Failure prevalence: {metrics['failure_rate']:.1%}.")
    st.dataframe(data.head(20), width="stretch")
    st.subheader("Class distribution")
    st.bar_chart(data["Machine failure"].value_counts().sort_index())
