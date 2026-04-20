#!/usr/bin/env bash
set -euo pipefail

echo "Running preseason PERFORMANCE-ONLY next-season NBA salary models..."
echo "Selection metric: ${NBAML_SELECTION_METRIC:-usd_mae}"
echo "Games threshold: ${NBAML_TRAIN_GAMES_THRESHOLD:-15}"
echo "RandomizedSearchCV iters: ${NBAML_PRESEASON_N_ITER_SINGLE:-6}"
echo "CV splits: ${NBAML_PRESEASON_CV_SPLITS:-3}"
echo "Blend weight step: ${NBAML_BLEND_WEIGHT_STEP:-0.10}"
echo "LightGBM enabled: $([[ \"${NBAML_DISABLE_LGBM:-1}\" == \"1\" ]] && echo \"0\" || echo \"1\")"
echo "Cache enabled: ${NBAML_USE_CACHE:-1} (force retrain: ${NBAML_FORCE_RETRAIN:-0})"
echo

python3 RefinedFeaturesPreseasonPerformanceV1.py
python3 PointGuardPreseasonPerformanceV1.py
python3 ShootingGuardPreseasonPerformanceV1.py
python3 PowerForwardPreseasonPerformanceV1.py
python3 SmallForwardPreseasonPerformanceV1.py
python3 CenterPreseasonPerformanceV1.py

echo
echo "Done. JSON outputs are in ./results/"
