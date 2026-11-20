# Freight Rate Prediction

Predict `posted_rate` for truckload freight. Assessment brief: [docs/Freight_Rate_ML_Assessment.pdf](docs/Freight_Rate_ML_Assessment.pdf) (original instructions: [docs/assignment_README.md](docs/assignment_README.md)).

> Work in progress. Current stage: exploratory data analysis.

## Layout

```
data/                     input files (paths match the brief and score.py)
  train_test.csv            48,000 labelled loads, Jan–Oct 2025
  validation.csv            12,000 loads to predict, Nov–Dec 2025
  validation_predictions_template.csv
  december_chart_inputs.csv 31 fixed-input rows for the December chart
docs/                     assessment brief
eda/                      EDA scripts, one per question (run in order)
reports/eda/              generated EDA reports (markdown) and figures
score.py                  provided scorer: validates outputs, draws the December chart
requirements.txt          scorer pins + solution dependencies
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12
python -m pip install -r requirements.txt
```

## EDA

```bash
python eda/run_all.py          # all steps (~80 s), or a single one, e.g. python eda/05_temporal.py
```

| Step | Question it answers |
|---|---|
| `01_data_audit.py` | Do the files match the brief? Schema, IDs, coverage, missingness, duplicates, impossible values |
| `02_univariate.py` | What does each column look like on its own? Target scale, skew, spikes, caps |
| `03_data_quality.py` | What is each anomaly, how big is it, and what is the evidence for its mechanism? |
| `04_drivers.py` | What drives the rate? Distance → equipment → weight/market → geography, one at a time |
| `05_temporal.py` | Which effects depend on the date, and are they carried by a column or only by the calendar? |
| `06_drift_and_leakage.py` | Will the patterns carry over to Nov–Dec? Is any feature too good to be true? |
| `07_december_inputs.py` | What will the model see for the December chart, and what is missing? |
| `08_trap_proof.py` | Does `quote_signal` help a model on future months? (with/without, random vs time-forward, shuffle counterfactual) |
| `09_structure.py` | Is price a product of independent effects? Ladder, interactions, functional shapes, noise |
| `10_time_shape.py` | Exact shape of drift and quarter-end ramp; out-of-sample check; expected December curve |
| `11_unseen_cities.py` | Can neighbouring cities price a city never seen in training? |
| `12_volume_effects.py` | Do load counts (day, city, equipment, lane, imbalance) affect price? |

Reports land in `reports/eda/` (index: `reports/eda/README.md`).
