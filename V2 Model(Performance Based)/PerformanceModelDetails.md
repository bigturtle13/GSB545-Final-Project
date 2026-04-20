# PositionModelsPreseasonPerformanceV1

Performance-only preseason NBA salary models that predict **next season salary (t+1)** from information known after season `t`.

## Design Goal

- Predict `next_salary` from player/team performance context.
- Exclude all salary-history predictors from model feature packs.
- Default dataset: `NBADataCleanV4.csv`.
- Uses V4 additions (service bucket, missed games, durability trend, BRef impact metrics, trend features, non-payroll cap context, and position/archetype scarcity features).

## What Is Included

- Feature packs:
  - `prior_perf_core_v1`
  - `prior_perf_rolling_v1`
- Models:
  - XGBoost
  - CatBoost
  - LightGBM (disabled by default for faster runs)
  - XGB/CatBoost blend (weight tuned on validation window)
- Tuning/CV:
  - `RandomizedSearchCV`
  - `StratifiedKFold` on binned regression target (fallback: `KFold`)

## Efficiency Defaults

- `NBAML_PRESEASON_N_ITER_SINGLE=6`
- `NBAML_PRESEASON_CV_SPLITS=3`
- `NBAML_BLEND_WEIGHT_STEP=0.10`
- `NBAML_DISABLE_LGBM=1`

These defaults are intentionally lighter than the full preseason run to reduce iteration time.

## Run All

```bash
cd "/Users/amritdhillon/Desktop/Advanced ML/Final Project/NBA Data/PositionModelsPreseasonPerformanceV1"
NBAML_SELECTION_METRIC=usd_mae bash run_all_models.sh
```

To explicitly set the dataset:

```bash
NBAML_CSV_PATH="/Users/amritdhillon/Desktop/Advanced ML/Final Project/NBA Data/NBADataCleanV4.csv" \
NBAML_SELECTION_METRIC=usd_mae bash run_all_models.sh
```

## Fast Iteration Run

```bash
cd "/Users/amritdhillon/Desktop/Advanced ML/Final Project/NBA Data/PositionModelsPreseasonPerformanceV1"
bash run_fast_dev.sh
```

## Useful Environment Variables

- `NBAML_SELECTION_METRIC` = `usd_mae` | `usd_rmse` | `log_mae` | `log_rmse`
- `NBAML_TRAIN_GAMES_THRESHOLD` (default `15`)
- `NBAML_PRESEASON_N_ITER_SINGLE` (default `6`)
- `NBAML_PRESEASON_CV_SPLITS` (default `3`)
- `NBAML_BLEND_WEIGHT_STEP` (default `0.10`)
- `NBAML_USE_CACHE` (default `1`)
- `NBAML_FORCE_RETRAIN` (default `0`)
- `NBAML_DISABLE_XGB` / `NBAML_DISABLE_LGBM` (set to `1` to disable)
- `NBAML_MODEL_N_JOBS` (default `-1`)

## Outputs

Metrics JSON files are written to `./results/`.
Each JSON includes:

- selected feature pack/model
- final holdout metrics
- tuning configuration
- naive baseline (`next salary = prior-season salary`) for context
- top XGBoost importances
