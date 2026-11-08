"""Fast pre-deployment checks for data, model artifacts and inference."""
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).parent
required = [
    ROOT / "data/ai4i2020.csv",
    ROOT / "data/sample_machines.csv",
    ROOT / "models/predictive_maintenance_model.joblib",
    ROOT / "reports/model_metrics.json",
]
missing = [str(x) for x in required if not x.exists()]
if missing:
    raise SystemExit("Missing required artifacts: " + ", ".join(missing))

bundle = joblib.load(ROOT / "models/predictive_maintenance_model.joblib")
sample = pd.read_csv(ROOT / "data/sample_machines.csv")
if set(bundle["features"]) - set(sample.columns):
    raise SystemExit("Sample CSV does not match required model features")
probabilities = bundle["pipeline"].predict_proba(sample[bundle["features"]])[:, 1]
if not ((probabilities >= 0).all() and (probabilities <= 1).all()):
    raise SystemExit("Model produced invalid probabilities")

metrics = json.loads((ROOT / "reports/model_metrics.json").read_text())
for key in ("roc_auc", "average_precision", "precision", "recall", "f1", "threshold"):
    if key not in metrics:
        raise SystemExit(f"Missing metric: {key}")
print("PASS: model, dataset and sample inference are ready.")
print("Sample failure probabilities:", ", ".join(f"{p:.1%}" for p in probabilities))
