#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m jupyter nbconvert \
  --execute \
  --to notebook \
  --inplace \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=python3 \
  "NBA_Performance_Salary_NN_ClassStyle_Stronger.ipynb"
