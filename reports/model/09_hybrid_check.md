# Modelling 09 · Is the hybrid reliably better than structural v3?

Hybrid on v3 = structural v3 + gradient-boosted trees (400 trees, 31 leaves, ≥ 100 loads per leaf) fitted to v3's log leftovers. Criteria fixed before running; the hybrid is chosen only if it meets all five.

## 1. Round by round (hybrid − v3, averaged over the 4 hidden-city draws; negative = hybrid better)

| fold | v3 MAPE | hybrid MAPE | Δ MAPE | Δ RMSE ($) | Δ MAPE unseen cities | draws hybrid better (MAPE) | draws hybrid better (RMSE) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train Jan–Apr → test May–Jun (fresh) | 1.426 | 1.399 | -0.027 | -3.682 | -0.021 | 4 | 4 |
| train Jan–Aug → test Sep–Oct | 1.487 | 1.468 | -0.020 | 1.424 | -0.003 | 4 | 0 |
| train Jan–Jul → test Aug–Sep | 1.284 | 1.251 | -0.033 | -1.735 | -0.033 | 4 | 4 |
| train Jan–Jun → test Jul–Aug | 1.500 | 1.472 | -0.027 | -2.014 | -0.047 | 4 | 4 |
| train Jan–Mar → test Apr–May (fresh) | 1.436 | 1.437 | 0.001 | -1.726 | 0.011 | 3 | 4 |
| train Jan–May → test Jun–Jul | 1.315 | 1.338 | 0.022 | 0.125 | 0.019 | 0 | 1 |

## 2. Criteria

| criterion | result | met |
| --- | --- | --- |
| C1 fresh rounds: better on average (MAPE, RMSE) | -0.013 pts, -2.70 $ | True |
| C2 better in ≥ 75% of the 24 round × draw pairs | MAPE 79%, RMSE 71% | False |
| C3 never worse by > 0.03 MAPE points | largest loss +0.032 pts | False |
| C4 unseen-city loads not worse on average | -0.012 pts | True |
| C5 Nov–Dec monthly median adjustment within ±0.2% | 0.18% | True |

Average over all 24 pairs: hybrid − v3 = -0.014 MAPE points, -1.27 $ RMSE (v3 1.408% / $49.41; hybrid 1.394% / $48.15).

**Decision: structural v3** (not all criteria met).

## 3. What the trees change in Nov–Dec (both fitted on all of Jan–Oct)

Adjustment = hybrid prediction ÷ v3 prediction − 1, on the 12,000 validation loads.

| loads | n | median adjustment (%) | mean (%) | 5th pct (%) | 95th pct (%) |
| --- | --- | --- | --- | --- | --- |
| November | 5,836 | -0.183 | -0.166 | -0.644 | 0.408 |
| December | 6,164 | -0.168 | -0.148 | -0.719 | 0.497 |
| known cities | 10,553 | -0.175 | -0.157 | -0.684 | 0.455 |
| new city | 1,447 | -0.179 | -0.159 | -0.675 | 0.464 |
| all | 12,000 | -0.176 | -0.157 | -0.683 | 0.455 |

December chart (Lexington → Fort Wayne): the trees change the 31 daily prices by -0.17% to +0.82%.

## 4. What the trees learn

Residual trees allowed to see one feature group only (4 standard rounds, first hidden-city draw).

| trees see | MAPE (%) | RMSE ($) | gain vs v3 (MAPE pts) |
| --- | --- | --- | --- |
| nothing (structural v3) | 1.394 | 47.676 | 0.000 |
| quarter-end timing + equipment | 1.399 | 48.009 | -0.004 |
| geography (4 coordinates) | 1.396 | 47.734 | -0.001 |
| market index | 1.393 | 47.667 | 0.001 |
| distance + weight | 1.391 | 47.537 | 0.003 |
| all 9 features | 1.382 | 47.255 | 0.012 |
