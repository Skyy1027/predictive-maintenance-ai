# MaintainAI - Predictive Maintenance Project

A presentation-ready undergraduate project that predicts whether an industrial machine is likely to fail from recent sensor readings. It turns the probability into one of three practical recommendations: **maintenance now**, **schedule inspection**, or **continue and monitor**.

## What is included

- `data/ai4i2020.csv` - the UCI AI4I 2020 Predictive Maintenance dataset (10,000 labelled records).
- `train_model.py` - reproducible train/test split, class-weighted random forest training, threshold selection, and evaluation report.
- `app.py` - a Streamlit dashboard for a single live assessment, batch CSV screening, performance evidence, and dataset exploration.
- `models/` and `reports/` - created after training.

## Run it

```bash
python3 -m pip install -r requirements.txt
python3 train_model.py
python3 -m streamlit run app.py
```

Open the local address printed by Streamlit (usually `http://localhost:8501`).

For the batch-screening demo, upload `data/sample_machines.csv`.

## Deployment

The repository is ready for either Streamlit Community Cloud or Docker.

**Streamlit Community Cloud:** push this project folder to a GitHub repository, create an app at [share.streamlit.io](https://share.streamlit.io), select the repository and branch, and set `app.py` as the main file. The checked-in trained model and dataset will be available at startup.

**Docker:** build and run the included image:

```bash
docker build -t maintainai:latest .
docker run --rm -p 8501:8501 maintainai:latest
```

Then open `http://localhost:8501`. The Docker image includes a health check at `/_stcore/health`.

## Verification

Before presenting or deploying, run:

```bash
python3 validate_project.py
```

It checks that the data, model and metrics exist and that the model produces valid probabilities for the included sample machines.

## How to present it

1. Explain the real problem: unexpected failures stop production and cost more than planned maintenance.
2. Show the dataset tab: six available input features are product type, two temperatures, rotational speed, torque, and tool wear. The target is `Machine failure`.
3. Show Model evidence: the dataset is imbalanced, so Average Precision and Recall matter alongside ROC-AUC. The code uses a stratified split and class weighting.
4. Use Live prediction: change torque, speed and tool wear, then explain that the app outputs a probability plus an action threshold rather than treating it as certainty.
5. Close with the limitation: this is decision support trained on benchmark data. A real deployment needs historical site data, calibration, monitoring, and engineer approval.

## Dataset and methodology

Dataset: UCI Machine Learning Repository, **AI4I 2020 Predictive Maintenance Dataset** (10,000 synthetic-but-realistically structured records published for predictive-maintenance research). The included data was downloaded from UCI on 8 November 2026.

The classifier is a class-weighted random forest. Categorical product type is one-hot encoded; numerical missing values are median-imputed. The 20% hold-out set is stratified. The application action threshold is chosen to maximise F1 on that hold-out set; this makes the demo explainable but is optimistic, so a production system must choose a threshold on a separate validation set according to the cost of false alarms and missed failures.

## Batch CSV format

Use exactly these headers:

`Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]`

The UCI dataset additionally contains a record ID and five failure-mode columns. They are intentionally excluded from model inputs because they describe failure mechanisms / labels and would leak information not available at prediction time.
