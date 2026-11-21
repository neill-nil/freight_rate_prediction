"""The final model's settings, fixed here so `predict.py` needs nothing but the data files.

Every value was chosen on the time-forward folds (train up to a cut-off, test the next two months, 4 hidden cities):
  distance_degree=3         log-distance cubic                      eda/09, modelling 01
  weight_bounds=20k–40k     weight only matters inside this band     eda/09, modelling 01
  ramp_days=28              quarter-end ramp length                  eda/10, modelling 01
  ramp_by_equipment=True    the ramp is larger for Reefer / Flatbed  modelling 02 (diagnosis of the hybrid)
  trend='damped'            drift continues at half speed            modelling 01, 08 (best on average, never worst fold)
  unseen='knn', knn_k=5     new city = its 5 nearest cities          eda/11, modelling 01
  lane_shrink=3             directed lane effect, shrunk (n + 3)     modelling 04
"""
from __future__ import annotations

STRUCTURAL_V3 = {
    "distance_degree": 3,
    "weight_bounds": (20_000, 40_000),
    "ramp_days": 28,
    "ramp_by_equipment": True,
    "trend": "damped",
    "unseen": "knn",
    "knn_k": 5,
    "lane_shrink": 3,
}

# trees on the structural model's leftovers (tuned in modelling 02; evaluated on v3 in modelling 04, 08, 09)
HYBRID_TREES = {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 1.0}

FINAL_MODEL = "structural_v3"  # decided in modelling 09
