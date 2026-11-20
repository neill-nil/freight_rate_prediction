# Modelling 02 · Gradient-boosted trees: tuned challenger and hybrid

Same folds, hidden cities and scoring as Modelling 01. Model: scikit-learn HistGradientBoostingRegressor on cleaned inputs (log distance, equipment, weight, market_index, pickup/delivery lat & lon, days to quarter end; optionally a time index). City names are deliberately not used, so new cities are handled through coordinates.

## 1. Two design choices, with default settings

| model | MAPE (%) | RMSE ($) | MAE ($) | MAPE seen cities (%) | MAPE unseen cities (%) | MAPE month 1 (%) | MAPE month 2 (%) | bias (%) | worst-fold MAPE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target log_rate, no time index | 3.084 | 98.202 | 73.070 | 3.072 | 3.154 | 2.657 | 3.502 | -2.914 | 3.886 |
| target log_rate, with time index | 1.992 | 67.756 | 46.717 | 1.977 | 2.098 | 1.833 | 2.151 | -0.486 | 2.235 |
| target log_rpm, no time index | 3.026 | 96.260 | 72.216 | 3.020 | 3.050 | 2.612 | 3.431 | -2.922 | 3.878 |
| target log_rpm, with time index | 1.877 | 63.471 | 44.193 | 1.856 | 2.014 | 1.729 | 2.024 | -0.516 | 2.135 |

- **Target**: log price per mile removes most of the steep distance trend before the trees start, so their step-shaped fit only has to follow a gentle slope.
- **Time index**: trees can't extrapolate. With a time index they hold the last training level (like 'freeze'); without it they average over all training months and miss the drift entirely.
- Chosen for tuning: **target log_rpm, with time index**.

## 2. Hyperparameter tuning (random search, 24 settings × 4 time-forward folds)

Search space: learning_rate ∈ [0.02, 0.04, 0.06, 0.1]; max_iter ∈ [300, 600, 1000, 1500]; max_leaf_nodes ∈ [15, 31, 63, 127]; min_samples_leaf ∈ [20, 50, 100, 200]; l2_regularization ∈ [0.0, 0.1, 1.0, 10.0]; max_features ∈ [0.5, 0.8, 1.0]. Ranked by mean future-month MAPE:

| learning_rate | max_iter | max_leaf_nodes | min_samples_leaf | l2_regularization | max_features | MAPE (%) | RMSE ($) | worst-fold MAPE (%) | MAPE unseen cities (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.040 | 1,000 | 31 | 20 | 0.100 | 0.800 | 1.820 | 62.015 | 2.124 | 1.969 |
| 0.040 | 1,500 | 31 | 50 | 1.000 | 0.800 | 1.834 | 62.262 | 2.134 | 1.973 |
| 0.060 | 1,500 | 15 | 100 | 0.100 | 0.800 | 1.838 | 63.390 | 2.083 | 1.965 |
| 0.040 | 1,000 | 31 | 200 | 10.000 | 0.500 | 1.850 | 63.354 | 2.112 | 1.977 |
| 0.040 | 1,000 | 127 | 200 | 0.000 | 0.500 | 1.862 | 64.129 | 2.105 | 1.991 |
| 0.040 | 1,500 | 31 | 100 | 10.000 | 1.000 | 1.862 | 63.342 | 2.129 | 2.011 |
| 0.100 | 1,500 | 15 | 50 | 10.000 | 0.800 | 1.870 | 63.858 | 2.093 | 1.998 |
| 0.100 | 600 | 31 | 200 | 0.000 | 0.500 | 1.883 | 65.125 | 2.149 | 2.020 |
| 0.020 | 1,000 | 127 | 50 | 0.100 | 1.000 | 1.893 | 65.183 | 2.126 | 2.028 |
| 0.020 | 600 | 127 | 20 | 0.100 | 0.800 | 1.896 | 65.391 | 2.128 | 2.035 |

Spread among the top 5 settings: MAPE 1.820–1.862%. Worst setting tried: 2.049%. Chosen: learning_rate=0.04, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.1, max_features=0.8.

Which settings mattered (Spearman correlation with MAPE across the 24 tries; negative = higher value is better): learning_rate -0.05, max_iter -0.46, max_leaf_nodes +0.48, min_samples_leaf +0.02, l2_regularization -0.10, max_features +0.44.

## 3. Hybrid: structural model + small trees on its residuals

The trees alone lose to the structural model (they can't extrapolate the drift and are weaker on unseen cities). But they might still spot patterns the structural model misses. The hybrid keeps the structural model and lets small trees learn only its leftovers. Tuned with a random search of 12 settings.

| learning_rate | max_iter | max_leaf_nodes | min_samples_leaf | l2_regularization | MAPE (%) | RMSE ($) | worst-fold MAPE (%) | MAPE unseen cities (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050 | 400 | 31 | 100 | 10.000 | 1.488 | 50.831 | 1.582 | 1.565 |
| 0.050 | 400 | 31 | 400 | 1.000 | 1.495 | 51.105 | 1.591 | 1.568 |
| 0.050 | 400 | 7 | 200 | 1.000 | 1.505 | 51.423 | 1.605 | 1.548 |
| 0.050 | 400 | 7 | 200 | 10.000 | 1.506 | 51.458 | 1.603 | 1.548 |
| 0.020 | 400 | 31 | 400 | 1.000 | 1.510 | 51.658 | 1.614 | 1.554 |
| 0.050 | 200 | 15 | 100 | 1.000 | 1.511 | 51.614 | 1.612 | 1.553 |

All 12 hybrid settings score 1.488–1.555% vs 1.617% for the structural model alone, so the gain is not luck of one setting. **The trees are finding something real.** What?

## 4. What did the residual trees find?

Diagnosis by restriction: refit the hybrid letting the residual trees see only some inputs. If a restricted version keeps the gain, the pattern lives in those inputs.

| residual trees may look at | MAPE (%) | RMSE ($) | MAPE unseen cities (%) |
| --- | --- | --- | --- |
| nothing (= structural alone) | 1.617 | 55.613 | 1.646 |
| all inputs | 1.488 | 50.831 | 1.565 |
| geography: both ends (4 coordinates) | 1.557 | 53.398 | 1.678 |
| geography: pickup end only | 1.615 | 55.533 | 1.642 |
| distance, equipment, weight, market, days to quarter end | 1.530 | 52.267 | 1.556 |
| …without days to quarter end | 1.624 | 55.760 | 1.652 |
| days to quarter end alone | 1.622 | 55.912 | 1.651 |
| equipment alone | 1.617 | 55.613 | 1.646 |
| days to quarter end + equipment | 1.538 | 52.641 | 1.570 |
| days to quarter end + distance | 1.633 | 56.159 | 1.665 |
| days to quarter end + weight | 1.624 | 56.004 | 1.649 |

- **Days to quarter end only helps together with equipment**: neither alone helps, the pair recovers most of the non-geographic gain. So the quarter-end rush is not the same size for every trailer type.
- **Geography helps only when both ends are visible**: pickup coordinates alone add nothing (city effects already cover that). So there is a small route-level (origin-region × destination-region) effect beyond the two city effects.

Confirmed directly: residual of a structural model without any ramp, last 14 days of each quarter vs the rest (%):

| equipment (train Jan–Aug) | last 14 days of quarter | rest of quarter | quarter-end lift (pts) |
| --- | --- | --- | --- |
| Dry Van | 1.57 | -0.21 | 1.79 |
| Flatbed | 5.01 | -0.67 | 5.68 |
| Reefer | 3.23 | -0.45 | 3.68 |

## 5. Feeding the discovery back: structural v2, then the hybrid on top

Structural v2 = structural + a separate ramp height per equipment (2 extra coefficients). Fitted quarter-end lift (train Jan–Aug): Dry Van +2.6%, Reefer +5.4%, Flatbed +8.4%. The drift was also tried per equipment: no gain.

Hybrid retuned on top of v2 (12 settings): 1.483–1.518% MAPE.

| learning_rate | max_iter | max_leaf_nodes | min_samples_leaf | l2_regularization | MAPE (%) | RMSE ($) | worst-fold MAPE (%) | MAPE unseen cities (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050 | 400 | 31 | 100 | 1.000 | 1.483 | 50.498 | 1.570 | 1.566 |
| 0.050 | 400 | 15 | 100 | 1.000 | 1.490 | 50.842 | 1.581 | 1.553 |
| 0.050 | 200 | 31 | 400 | 1.000 | 1.495 | 50.978 | 1.591 | 1.551 |
| 0.020 | 400 | 31 | 100 | 1.000 | 1.497 | 51.044 | 1.590 | 1.550 |

## 6. Head-to-head on future months

| model | MAPE (%) | RMSE ($) | MAE ($) | MAPE seen cities (%) | MAPE unseen cities (%) | MAPE month 1 (%) | MAPE month 2 (%) | bias (%) | worst-fold MAPE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| structural v1 (Modelling 01) | 1.617 | 55.613 | 38.186 | 1.613 | 1.646 | 1.571 | 1.663 | 0.061 | 1.740 |
| trees alone, tuned | 1.820 | 62.015 | 43.028 | 1.797 | 1.969 | 1.692 | 1.947 | -0.532 | 2.124 |
| hybrid on v1 | 1.488 | 50.831 | 35.115 | 1.477 | 1.565 | 1.406 | 1.568 | -0.029 | 1.582 |
| structural v2 (+ ramp per equipment) | 1.532 | 52.308 | 36.132 | 1.528 | 1.564 | 1.461 | 1.602 | 0.052 | 1.626 |
| hybrid on v2 | 1.483 | 50.498 | 34.966 | 1.471 | 1.566 | 1.401 | 1.562 | -0.028 | 1.570 |

MAPE per fold:

| fold | structural v1 (Modelling 01) | trees alone, tuned | hybrid on v1 | structural v2 (+ ramp per equipment) | hybrid on v2 |
| --- | --- | --- | --- | --- | --- |
| train Jan–Aug → test Sep–Oct | 1.740 | 1.908 | 1.582 | 1.626 | 1.570 |
| train Jan–Jul → test Aug–Sep | 1.542 | 1.453 | 1.378 | 1.432 | 1.364 |
| train Jan–Jun → test Jul–Aug | 1.638 | 1.793 | 1.547 | 1.620 | 1.552 |
| train Jan–May → test Jun–Jul | 1.548 | 2.124 | 1.445 | 1.451 | 1.444 |

Hybrid on v2 beats structural v2 in 4 of 4 folds; the remaining gain is the route-level geography effect.
