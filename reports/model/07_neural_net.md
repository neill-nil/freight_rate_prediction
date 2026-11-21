# Modelling 07 · Neural network (multi-layer perceptron)

Same features, folds and scoring as Modelling 05–06. scikit-learn MLPRegressor with standardised inputs and target, Adam, early stopping on 10% of the training rows.

## 1. Result

| model | MAPE (%) | RMSE ($) | MAE ($) | MAPE seen cities (%) | MAPE unseen cities (%) | MAPE month 2 (%) | bias (%) | worst-fold MAPE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Structural v3 (reference) | 1.394 | 47.676 | 32.948 | 1.370 | 1.564 | 1.470 | 0.046 | 1.494 |
| v3 + MLP on residuals | 1.407 | 48.460 | 33.414 | 1.383 | 1.571 | 1.496 | 0.026 | 1.614 |
| standalone MLP | 1.587 | 54.451 | 37.650 | 1.567 | 1.724 | 1.677 | -0.230 | 1.940 |

Beats structural v3 by the clear-gain bar: **no**.

MAPE per fold:

| fold | Structural v3 (reference) | standalone MLP | v3 + MLP on residuals |
| --- | --- | --- | --- |
| train Jan–Aug → test Sep–Oct | 1.489 | 1.940 | 1.614 |
| train Jan–Jul → test Aug–Sep | 1.281 | 1.438 | 1.303 |
| train Jan–Jun → test Jul–Aug | 1.494 | 1.477 | 1.407 |
| train Jan–May → test Jun–Jul | 1.313 | 1.491 | 1.303 |

## 2. Tuning

### standalone MLP

| hidden layers | L2 (alpha) | learning rate | MAPE (%) | RMSE ($) | worst-fold MAPE (%) |
| --- | --- | --- | --- | --- | --- |
| (256, 128) | 0.0001 | 0.0005 | 1.5865 | 54.4512 | 1.9404 |
| (128, 64, 32) | 0.0001 | 0.0010 | 1.5873 | 54.4070 | 1.8501 |
| (128, 64) | 0.0010 | 0.0010 | 1.6274 | 55.2974 | 1.9261 |
| (128, 64) | 0.0001 | 0.0010 | 1.6361 | 55.7484 | 1.9420 |
| (64, 32) | 0.0001 | 0.0010 | 1.6939 | 57.8992 | 1.9792 |

### v3 + MLP on residuals

| hidden layers | L2 (alpha) | learning rate | MAPE (%) | RMSE ($) | worst-fold MAPE (%) |
| --- | --- | --- | --- | --- | --- |
| (64, 32) | 0.0100 | 0.0010 | 1.4068 | 48.4601 | 1.6141 |
| (64, 32) | 0.0010 | 0.0010 | 1.4146 | 48.8731 | 1.6155 |
| (32, 16) | 0.0010 | 0.0010 | 1.4215 | 48.8152 | 1.6995 |
