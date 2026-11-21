# Modelling 06 · Ensembles: can combining models beat structural v3?

Same folds, same scoring. Base models are the best tuned setting of each type from Modelling 05.

## 1. Averaging (geometric mean of predictions)

| model | MAPE (%) | RMSE ($) | MAE ($) | MAPE seen cities (%) | MAPE unseen cities (%) | MAPE month 2 (%) | bias (%) | worst-fold MAPE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| average: v3 + Linear regression (least squares) | 1.394 | 47.671 | 32.944 | 1.370 | 1.566 | 1.470 | 0.046 | 1.496 |
| average: v3 + Ridge regression | 1.394 | 47.671 | 32.944 | 1.370 | 1.566 | 1.470 | 0.046 | 1.496 |
| Structural v3 (reference) | 1.394 | 47.676 | 32.948 | 1.370 | 1.564 | 1.470 | 0.046 | 1.494 |
| average: v3 + Lasso regression | 1.394 | 47.674 | 32.947 | 1.370 | 1.566 | 1.471 | 0.045 | 1.497 |
| average: v3 + LightGBM | 1.455 | 49.627 | 34.366 | 1.431 | 1.629 | 1.546 | -0.285 | 1.630 |
| average: v3 + XGBoost | 1.458 | 49.728 | 34.415 | 1.434 | 1.629 | 1.554 | -0.309 | 1.664 |
| average: v3 + Gradient boosting (scikit-learn HGB) | 1.481 | 51.081 | 35.190 | 1.458 | 1.647 | 1.584 | -0.268 | 1.613 |
| average: v3 + top 3 flexible (LightGBM, XGBoost, Gradient boosting (scikit-learn HGB)) | 1.527 | 52.055 | 36.068 | 1.504 | 1.694 | 1.632 | -0.448 | 1.721 |
| average: v3 + Random forest | 1.569 | 54.045 | 37.288 | 1.548 | 1.723 | 1.628 | -0.322 | 1.729 |
| average: v3 + Extra trees | 1.574 | 54.468 | 37.566 | 1.556 | 1.710 | 1.644 | -0.188 | 1.682 |
| average: v3 + k-nearest neighbours | 1.819 | 61.908 | 43.125 | 1.805 | 1.912 | 1.879 | 0.016 | 1.863 |

Why averaging rarely helps here: correlation of the models' log errors with v3's errors (1 = they make the same mistakes):

| index | error correlation with v3 |
| --- | --- |
| Lasso regression | 0.999 |
| Ridge regression | 0.999 |
| Linear regression (least squares) | 0.999 |
| LightGBM | 0.886 |
| XGBoost | 0.876 |
| Gradient boosting (scikit-learn HGB) | 0.874 |
| Extra trees | 0.759 |
| Random forest | 0.747 |
| k-nearest neighbours | 0.589 |

## 2. Stacking by residual learning (flexible model on top of v3)

Each learner sees all 20 features plus v3's own prediction, and learns v3's leftover error. v3's predictions for training rows are 5-fold out-of-fold, so the learner sees honest leftovers, not ones v3 has already memorised.

| model | MAPE (%) | RMSE ($) | MAE ($) | MAPE seen cities (%) | MAPE unseen cities (%) | MAPE month 2 (%) | bias (%) | worst-fold MAPE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Structural v3 (reference) | 1.394 | 47.676 | 32.948 | 1.370 | 1.564 | 1.470 | 0.046 | 1.494 |
| v3 + random forest on residuals | 1.395 | 47.787 | 33.040 | 1.367 | 1.593 | 1.508 | -0.085 | 1.520 |
| v3 + XGBoost on residuals | 1.395 | 47.763 | 33.048 | 1.367 | 1.595 | 1.510 | -0.064 | 1.527 |
| v3 + scikit-learn HGB on residuals | 1.403 | 47.852 | 33.155 | 1.375 | 1.607 | 1.521 | -0.091 | 1.530 |
| v3 + LightGBM on residuals | 1.404 | 47.995 | 33.235 | 1.375 | 1.608 | 1.527 | -0.089 | 1.536 |

### Tuning: v3 + LightGBM on residuals

| learning_rate | n_estimators | num_leaves | min_child_samples | reg_lambda | colsample_bytree | subsample | MAPE (%) | RMSE ($) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050 | 200 | 7 | 300 | 10.000 | 1.000 | 0.800 | 1.404 | 47.995 |
| 0.050 | 1,000 | 7 | 300 | 10.000 | 1.000 | 0.800 | 1.407 | 48.004 |
| 0.050 | 200 | 15 | 100 | 10.000 | 1.000 | 0.800 | 1.410 | 48.124 |
| 0.020 | 200 | 31 | 50 | 1.000 | 1.000 | 1.000 | 1.412 | 48.246 |
| 0.050 | 500 | 15 | 50 | 1.000 | 0.800 | 0.800 | 1.413 | 48.310 |
| 0.050 | 200 | 31 | 100 | 1.000 | 0.800 | 1.000 | 1.418 | 48.429 |
| 0.050 | 1,000 | 31 | 100 | 10.000 | 1.000 | 1.000 | 1.427 | 48.513 |
| 0.050 | 1,000 | 15 | 50 | 10.000 | 1.000 | 1.000 | 1.432 | 48.826 |

### Tuning: v3 + XGBoost on residuals

| learning_rate | n_estimators | max_depth | min_child_weight | reg_lambda | subsample | colsample_bytree | MAPE (%) | RMSE ($) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.020 | 200 | 3 | 50 | 1.000 | 0.800 | 1.000 | 1.395 | 47.763 |
| 0.020 | 200 | 4 | 50 | 10.000 | 0.800 | 0.800 | 1.396 | 47.790 |
| 0.050 | 200 | 4 | 100 | 10.000 | 0.800 | 0.800 | 1.399 | 47.924 |
| 0.050 | 200 | 3 | 100 | 10.000 | 1.000 | 1.000 | 1.401 | 47.986 |
| 0.050 | 500 | 3 | 100 | 1.000 | 0.800 | 0.800 | 1.407 | 48.186 |
| 0.020 | 200 | 6 | 50 | 1.000 | 1.000 | 1.000 | 1.410 | 48.287 |
| 0.050 | 200 | 6 | 20 | 1.000 | 0.800 | 1.000 | 1.420 | 48.541 |
| 0.050 | 1,000 | 6 | 20 | 1.000 | 0.800 | 0.800 | 1.453 | 49.357 |

### Tuning: v3 + scikit-learn HGB on residuals

| learning_rate | max_iter | max_leaf_nodes | min_samples_leaf | l2_regularization | MAPE (%) | RMSE ($) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.020 | 200 | 7 | 400 | 1.000 | 1.403 | 47.852 |
| 0.020 | 200 | 15 | 100 | 1.000 | 1.405 | 47.939 |
| 0.020 | 400 | 7 | 400 | 1.000 | 1.405 | 48.034 |
| 0.050 | 200 | 7 | 400 | 10.000 | 1.406 | 48.051 |
| 0.020 | 400 | 31 | 400 | 1.000 | 1.409 | 48.056 |
| 0.050 | 200 | 15 | 200 | 10.000 | 1.409 | 48.124 |

### Tuning: v3 + random forest on residuals

| max_features | min_samples_leaf | MAPE (%) | RMSE ($) |
| --- | --- | --- | --- |
| 0.300 | 50 | 1.395 | 47.787 |
| 0.300 | 20 | 1.396 | 47.824 |
| 0.500 | 100 | 1.401 | 47.977 |

## 3. Learned blend (non-negative weights, fitted leave-one-fold-out)

Weights chosen when each fold is held out (models with weight < 0.5% in every fold omitted):

| index | train Jan–May → test Jun–Jul | train Jan–Jun → test Jul–Aug | train Jan–Jul → test Aug–Sep | train Jan–Aug → test Sep–Oct |
| --- | --- | --- | --- | --- |
| Structural v3 (reference) | 0.740 | 0.000 | 0.746 | 0.712 |
| LightGBM | 0.000 | 0.033 | 0.000 | 0.000 |
| Gradient boosting (scikit-learn HGB) | 0.000 | 0.030 | 0.000 | 0.000 |
| v3 + LightGBM on residuals | 0.000 | 0.089 | 0.000 | 0.288 |
| v3 + XGBoost on residuals | 0.000 | 0.340 | 0.000 | 0.000 |
| v3 + random forest on residuals | 0.260 | 0.508 | 0.254 | 0.000 |

## 4. Ensemble leaderboard

| model | MAPE (%) | RMSE ($) | MAE ($) | MAPE seen cities (%) | MAPE unseen cities (%) | MAPE month 2 (%) | bias (%) | worst-fold MAPE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| learned blend (leave-one-fold-out) | 1.394 | 47.760 | 33.008 | 1.368 | 1.574 | 1.488 | 0.020 | 1.518 |
| average: v3 + Linear regression (least squares) | 1.394 | 47.671 | 32.944 | 1.370 | 1.566 | 1.470 | 0.046 | 1.496 |
| average: v3 + Ridge regression | 1.394 | 47.671 | 32.944 | 1.370 | 1.566 | 1.470 | 0.046 | 1.496 |
| average: v3 + XGBoost on residuals + v3 + random forest on residuals | 1.394 | 47.758 | 33.032 | 1.366 | 1.594 | 1.509 | -0.073 | 1.523 |
| Structural v3 (reference) | 1.394 | 47.676 | 32.948 | 1.370 | 1.564 | 1.470 | 0.046 | 1.494 |
| average: v3 + Lasso regression | 1.394 | 47.674 | 32.947 | 1.370 | 1.566 | 1.471 | 0.045 | 1.497 |
| v3 + random forest on residuals | 1.395 | 47.787 | 33.040 | 1.367 | 1.593 | 1.508 | -0.085 | 1.520 |
| v3 + XGBoost on residuals | 1.395 | 47.763 | 33.048 | 1.367 | 1.595 | 1.510 | -0.064 | 1.527 |
| average: v3 + LightGBM on residuals + v3 + random forest on residuals | 1.398 | 47.855 | 33.113 | 1.370 | 1.599 | 1.517 | -0.084 | 1.527 |
| average: v3 + LightGBM on residuals + v3 + XGBoost on residuals | 1.398 | 47.848 | 33.120 | 1.370 | 1.601 | 1.517 | -0.075 | 1.531 |
| average: v3 + scikit-learn HGB on residuals + v3 + random forest on residuals | 1.398 | 47.795 | 33.079 | 1.370 | 1.599 | 1.514 | -0.087 | 1.524 |
| average: v3 + XGBoost on residuals + v3 + scikit-learn HGB on residuals | 1.398 | 47.791 | 33.089 | 1.370 | 1.601 | 1.515 | -0.081 | 1.528 |

Ensembles that beat structural v3 by the clear-gain bar: **none**.

Per fold, best ensemble vs v3 (MAPE %):

| fold | Structural v3 (reference) | learned blend (leave-one-fold-out) |
| --- | --- | --- |
| train Jan–Aug → test Sep–Oct | 1.489 | 1.494 |
| train Jan–Jul → test Aug–Sep | 1.281 | 1.258 |
| train Jan–Jun → test Jul–Aug | 1.494 | 1.518 |
| train Jan–May → test Jun–Jul | 1.313 | 1.305 |
