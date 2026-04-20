#!/usr/bin/env bash
set -euo pipefail

echo "Running stable NBA salary models..."
echo "Selection metric: ${NBAML_SELECTION_METRIC:-usd_mae}"
echo "Games threshold: ${NBAML_TRAIN_GAMES_THRESHOLD:-15}"
echo "Cache enabled: ${NBAML_USE_CACHE:-1} (force retrain: ${NBAML_FORCE_RETRAIN:-0})"
echo

python3 RefinedFeaturesV1.py
python3 PointGuardModelV1.py
python3 ShootingGuardModelV1.py
python3 PowerForwardModelV1.py
python3 SmallForwardModelV1.py
python3 CenterModelV1.py

echo
echo "Done. JSON outputs are in ./results/"
