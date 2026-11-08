"""Train and evaluate the predictive-maintenance classifier on UCI AI4I 2020."""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/ai4i2020.csv")
MODEL_PATH = Path("models/predictive_maintenance_model.joblib")
METRICS_PATH = Path("reports/model_metrics.json")

FEATURES = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
TARGET = "Machine failure"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    X, y = data[FEATURES], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer([
        ("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), FEATURES[1:]),
    ])
    model = RandomForestClassifier(
        n_estimators=350, min_samples_leaf=2, class_weight="balanced",
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    # Pick the validation threshold with best F1, avoiding a hard-coded 0.50 cutoff
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(f1.argmax())
    threshold = float(thresholds[best_index])
    predictions = (probabilities >= threshold).astype(int)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    feature_importance = sorted(
        [{"feature": n.replace("type__", "").replace("numeric__", ""), "importance": round(float(i), 4)}
         for n, i in zip(transformed_names, importances)],
        key=lambda row: row["importance"], reverse=True,
    )
    metrics = {
        "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "records": int(len(data)), "failure_rate": round(float(y.mean()), 4),
        "test_records": int(len(y_test)), "threshold": round(threshold, 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "precision": round(float(report["1"]["precision"]), 4),
        "recall": round(float(report["1"]["recall"]), 4),
        "f1": round(float(report["1"]["f1-score"]), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "feature_importance": feature_importance,
    }
    joblib.dump({"pipeline": pipeline, "features": FEATURES, "threshold": threshold}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
