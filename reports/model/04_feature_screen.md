# Modelling 04 · Feature-engineering audit

Have we used everything the data offers before trying more model types? Inventory, a screen of untested candidates, and a time-forward confirmation of the one that mattered.

## 1. Inventory: from 12 raw inputs to model features

| raw input | structural (linear) model | tree models | evidence |
| --- | --- | --- | --- |
| pickup, delivery (city names) | linear: one shared effect per city (63 columns); unseen city → 5-nearest-neighbour effect; lane effect (pickup→delivery) | trees: replaced by coordinates | eda/04, 09, 11; modelling 04 |
| pickup/delivery lat & lon | kNN distances for unseen cities | trees: 4 features | eda/03, 11 |
| distance | log distance, its square and cube (curved distance effect) | trees: log distance; target = log price per mile | eda/04, 09; modelling 02 |
| equipment | Reefer, Flatbed indicators; × quarter-end ramp | trees: categorical | eda/04; modelling 02 |
| weight | abs() for sign flips, equipment-median for blanks, clipped to 20k–40k | trees: cleaned weight | eda/03, 09 |
| date | quarter-end ramp (28 days) and its × equipment; slow drift (damped beyond training) | trees: days to quarter end, time index | eda/05, 10; modelling 01–02 |
| market_index | linear term; blanks → same-date mean | trees: same | eda/03, 10 |
| quote_signal | DROPPED: leakage trap (meaning flips by month, none in Nov–Dec) | DROPPED | eda/06, 08 |

Structural v2 design: 74 columns (11 engineered terms + 63 city effects) + intercept. Lane effects (below) add one shrunk level per directed lane seen in training.

Engineered, tested, and dropped:

| family | engineered & tested | why dropped | evidence |
| --- | --- | --- | --- |
| calendar | day of week, month, 11 US holidays, quarter step levels, curved ramp, other ramp lengths | no gain after market_index + ramp + drift; month can't generalise to unseen Nov–Dec | eda/05, 10; modelling 01, 04 |
| market | log vs linear, seasonal level / weekly / per-load split, trailing 14/28/42-day means, momentum, volatility, × equipment, × distance | ≤ 1 basis point | eda/09, 10; modelling 04 |
| weight & distance shapes | log weight, linear weight, other bands, kink, cap flag, 70-mile floor flag, log(d+c), degree 1/2/4, × equipment | worse or ≤ 0.5 bp | eda/09; modelling 01, 04 |
| missing-value flags | weight was missing, market_index was missing | 0 bp: values are missing at random | eda/03, 09 |
| geography | separate pickup/delivery effects, lat/lon surfaces, region-pair effects, travel direction (Δlat/Δlon, bearing), city × equipment | ≤ 0.6 bp, or worse | eda/04, 09, 11; modelling 04 |
| volume | 11 load counts (day, equipment-day, city in/out, imbalance, city-month, lane-month) | ≤ 0.03% per std | eda/12 |
| drift | drift × equipment | 0 gain | modelling 02 |

## 2. Screen of candidates not tested before

Base: structural v2, out-of-fold residual std 1.793%, MAPE 1.429%. Each candidate is added on its own. The lane effect is each route's mean leftover, shrunk toward 0: level = sum of residuals / (number of loads + λ).

| candidate | columns | residual-std gain (basis points) | MAPE gain (pts) |
| --- | --- | --- | --- |
| lane effect, directed (A→B ≠ B→A), shrink λ=3 | 1 level per lane | 23.046 | 0.186 |
| lane effect, directed (A→B ≠ B→A), shrink λ=10 | 1 level per lane | 19.355 | 0.156 |
| lane effect, either direction, shrink λ=3 | 1 level per lane | 10.861 | 0.089 |
| lane effect, either direction, shrink λ=10 | 1 level per lane | 10.641 | 0.087 |
| market momentum (level change over 28 days) | 1 | 1.207 | 0.010 |
| day of week | 6 | 0.834 | 0.007 |
| market momentum (level change over 14 days) | 1 | 0.793 | 0.007 |
| region-pair effects (8 regions) | 63 | 0.559 | 0.005 |
| market momentum (level change over 7 days) | 1 | 0.492 | 0.004 |
| day of week × equipment | 12 | 0.388 | 0.003 |
| region-pair effects (6 regions) | 35 | 0.240 | 0.002 |
| market volatility (14-day std) | 1 | 0.229 | 0.001 |
| region-pair effects (4 regions) | 15 | 0.164 | 0.001 |
| second ramp, length 42 days, × equipment | 3 | 0.064 | 0.001 |
| second ramp, length 14 days, × equipment | 3 | 0.082 | 0.001 |
| travel direction: bearing (sin, cos) | 2 | 0.001 | 0.000 |
| travel direction: Δlat, Δlon | 2 | 0.010 | 0.000 |
| distance curve per equipment | 4 | 0.007 | 0.000 |
| city × equipment | 126 | -0.297 | -0.003 |

- **One clear winner: the directed lane effect.** Every other candidate is at noise level.
- Directed beats either-direction, so routes are not symmetric: A→B can be pricier than B→A beyond what the two cities explain.
- Correction to the EDA (04): it found lane-level variation of ±2.1% after city effects vs ±1.8% expected from noise, i.e. about ±1% genuine lane effect, and called it small. With ~10 loads per lane, a ±1% effect is learnable and turns out to be the largest one left.

## 3. Confirmation on the real test (future months, hidden cities)

| model | MAPE (%) | RMSE ($) | MAE ($) | MAPE seen cities (%) | MAPE unseen cities (%) | MAPE month 2 (%) | bias (%) | worst-fold MAPE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| structural v2 | 1.532 | 52.308 | 36.132 | 1.528 | 1.564 | 1.602 | 0.052 | 1.626 |
| v2 + lane effect λ=2 | 1.393 | 47.675 | 32.933 | 1.369 | 1.564 | 1.469 | 0.049 | 1.493 |
| v2 + lane effect λ=3 | 1.394 | 47.676 | 32.948 | 1.370 | 1.564 | 1.470 | 0.046 | 1.494 |
| v2 + lane effect λ=5 | 1.401 | 47.866 | 33.094 | 1.378 | 1.564 | 1.476 | 0.042 | 1.500 |
| v2 + lane effect λ=10 | 1.420 | 48.510 | 33.539 | 1.400 | 1.564 | 1.495 | 0.041 | 1.519 |
| hybrid on v2 (previous front-runner) | 1.483 | 50.498 | 34.966 | 1.471 | 1.566 | 1.562 | -0.028 | 1.570 |
| hybrid on v2 + lane effect λ=3 | 1.382 | 47.255 | 32.674 | 1.357 | 1.557 | 1.467 | -0.011 | 1.472 |

MAPE per fold:

| fold | structural v2 | v2 + lane effect λ=3 | hybrid on v2 (previous front-runner) | hybrid on v2 + lane effect λ=3 |
| --- | --- | --- | --- | --- |
| train Jan–Aug → test Sep–Oct | 1.626 | 1.489 | 1.570 | 1.472 |
| train Jan–Jul → test Aug–Sep | 1.432 | 1.281 | 1.364 | 1.253 |
| train Jan–Jun → test Jul–Aug | 1.620 | 1.494 | 1.552 | 1.468 |
| train Jan–May → test Jun–Jul | 1.451 | 1.313 | 1.444 | 1.335 |

- **Structural v3 = v2 + lane effect (λ=3)**: 1.394% MAPE, $47.7 RMSE, better than the previous hybrid in every fold.
- Trees on top add only 0.012 MAPE points and $0.42 RMSE, below the 0.02 / $1 'clear gain' bar, so the simpler v3 becomes the front-runner. The trees had mostly been approximating the lane effect from coordinates.
- Unseen-city loads don't benefit (1.564%): their lanes have never been seen. That is the main place left to gain.
- λ = 2–3 is best; theory agrees (λ ≈ noise variance ÷ lane-effect variance ≈ 1.6² ÷ 1.0² ≈ 3).
