# Freight Rate Prediction

Predict `posted_rate` for truckload freight. Assessment brief: [docs/Freight_Rate_ML_Assessment.pdf](docs/Freight_Rate_ML_Assessment.pdf) (original instructions: [docs/assignment_README.md](docs/assignment_README.md)).

> Final model: **structural v3**, a multiplicative price formula fitted on log price (distance curve, equipment, weight band, market index,
> city and directed-lane effects, quarter-end ramp, damped drift). Future-month test: 1.39% MAPE / $47.7 RMSE.

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
src/freight/              solution package: cleaning (data.py), features, models, time-aware evaluation, config.py (final settings), zoo.py (shared leak-free features)
modeling/                 modelling experiments (run in order)
reports/model/            generated modelling reports and figures
score.py                  provided scorer: validates outputs, draws the December chart
predict.py                final pipeline: train on Jan–Oct, write the deliverables, run the scorer
validation_predictions.csv  final predictions for the 12,000 validation loads
scorer_results/           candidate_december.png from score.py
report/                   PDF report (validation & split approach, December chart) and its build script
requirements.txt          scorer pins + solution dependencies
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12
python -m pip install -r requirements.txt
```

## Reproduce the submission

```bash
python predict.py      # ~3 s: writes validation_predictions.csv, fills data/december_chart_inputs.csv, runs score.py
```

`predict.py` cleans all files the same way, removes corrupted training prices (678), fits the final model
(`src/freight/config.py`) on the cleaned Jan–Oct loads, predicts, and calls `score.py`, which validates both files and
draws `scorer_results/candidate_december.png`. It only fills the `predicted_rate` column of the December file, so it
can be re-run safely; the outputs are identical on every run.

## Report

[`report/Freight_Rate_Prediction_Report.pdf`](report/Freight_Rate_Prediction_Report.pdf): validation and data-split approach,
data-quality fixes, model and alternatives, and the December chart. Rebuild with `python report/build_report.py`
(after `python predict.py`).

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

## Modelling

```bash
python modeling/01_structural.py   # baselines + structural model ladder and hyperparameter sweeps (~25 s)
python modeling/02_gbm.py          # gradient-boosted trees: design choices, random-search tuning, hybrid, diagnosis (~7 min)
python modeling/03_compare.py      # finalists at that stage (v2 / hybrid on v2), December behaviour, preview scorer run (~1 min)
python modeling/04_feature_screen.py   # feature-engineering audit: inventory, screen of untested features, lane effect (~40 s)
python modeling/05_model_zoo.py        # 9 standard model types on the same leak-free features, each tuned (~5 min)
python modeling/06_ensembles.py        # averaging, residual stacking, leave-one-fold-out blend (~1 min)
python modeling/07_neural_net.py       # multi-layer perceptron: standalone and stacked on the structural model (~2 min)
python modeling/08_loose_ends.py       # untried ideas (metric weighting, recency, joint/neighbour lane effects) + robustness checks (~1.5 min)
python modeling/09_hybrid_check.py     # stress test of structural v3 vs v3 + residual trees on 6 rounds × 4 hidden-city draws (~1 min)
```

All models are scored on **future months** (train up to a cut-off, test on the next two months, 4 folds) with 4 cities
hidden from training in each fold, mirroring the real task (Nov–Dec, 12% of loads in unseen cities). Metrics: MAPE and RMSE
(plus MAE, median APE), on test rows with genuine labels.
