# GSB545-Final-Project
NBA Salary Prediction ML Models

Last updated: 2026-04-20

## 1. Project Objective

Predict an NBA player's salary for season `t+1` using information available through season `t`.

This repository currently tracks two modeling tracks:

- **V1 (Stable Contract-Path Model):** strongest raw accuracy, uses salary-history features.
- **V2 (Performance Model):** performance-first design, excludes salary-history features, aligns better with basketball-performance interpretation.

---

## 2. Modeling Tracks

### V1: Stable Contract-Path Package

Folder: `NBA Data/PositionModelsStable`

Key characteristics:

- Uses salary-history features (`prev_salary`, `prev2_salary`, `prev_salary_growth`, etc.).
- Best for minimum USD error in current experiments.
- More contract-path dependent (strong persistence from prior salary levels).

Final holdout results (2023-2025):

| Script | Selected Model | MAE (USD) | RMSE (USD) | R² (USD) |
|---|---|---:|---:|---:|
| `RefinedFeaturesV1.py` | CatBoost | 2,797,707 | 4,966,766 | 0.82199 |
| `PointGuardModelV1.py` | Blend | 4,097,235 | 7,328,379 | 0.73725 |
| `ShootingGuardModelV1.py` | CatBoost | 2,733,830 | 4,760,295 | 0.75966 |
| `PowerForwardModelV1.py` | CatBoost | 2,958,632 | 4,773,451 | 0.85339 |
| `SmallForwardModelV1.py` | Blend | 2,763,402 | 5,098,684 | 0.78764 |
| `CenterModelV1.py` | LightGBM | 3,138,772 | 5,411,882 | 0.75661 |

### V2: Performance Model (No Salary-History Features)

Folder: `NBA Data/PositionModelsPreseasonPerformanceV1`
Dataset: `NBA Data/NBADataCleanV4.csv`

Design goals:

- Predict `next_salary` from prior-season and rolling performance context.
- Exclude salary-history predictors to reduce direct contract carryover.
- Keep runtime practical using randomized search + cached JSON outputs.

Latest quick-run config used for results below:

- `NBAML_PRESEASON_N_ITER_SINGLE=2`
- `NBAML_PRESEASON_CV_SPLITS=2`
- `NBAML_BLEND_WEIGHT_STEP=0.20`
- `NBAML_DISABLE_LGBM=0`
- `NBAML_SELECTION_METRIC=usd_mae`
- Holdout: 2023-2025

Latest V2 quick-run results (cleaned V4):

| Script | Selected Model | MAE (USD) | RMSE (USD) | R² (USD) |
|---|---|---:|---:|---:|
| `RefinedFeaturesPreseasonPerformanceV1.py` | Blend | 3,559,808 | 5,497,536 | 0.81161 |
| `PointGuardPreseasonPerformanceV1.py` | LightGBM | 4,773,229 | 7,397,616 | 0.75884 |
| `ShootingGuardPreseasonPerformanceV1.py` | LightGBM | 3,348,456 | 5,007,428 | 0.80471 |
| `PowerForwardPreseasonPerformanceV1.py` | LightGBM | 4,846,800 | 7,256,180 | 0.71818 |
| `SmallForwardPreseasonPerformanceV1.py` | Blend | 3,361,509 | 5,250,346 | 0.78458 |
| `CenterPreseasonPerformanceV1.py` | CatBoost | 4,077,101 | 6,227,152 | 0.70672 |

---

## 3. Data and Feature Definitions (V2)

V2 is built on `NBACleanDataV3.csv` with selected V4 additions.  
The table below documents every **direct V4 dataset column** used by the current V2 model pipeline (union across position scripts in the latest run).

### 3.1 Direct V4 columns used by V2 (65)

| Column | Definition |
|---|---|
| `adv_ast_to` | Advanced assist-to-turnover metric. |
| `adv_def_rating` | Advanced defensive rating (lower is better). |
| `adv_net_rating` | Advanced net rating (offensive minus defensive impact). |
| `adv_true_shooting_pct` | True shooting percentage (2PT, 3PT, FT efficiency combined). |
| `adv_usage_pct` | Usage rate (% of team possessions used while on court). |
| `age` | Player age in that season. |
| `all_star_selections_through_prev_season` | Cumulative All-Star selections up to the prior season. |
| `archetype_confidence_score` | Confidence score for assigned archetype label. |
| `archetype_is_balanced_connector` | 1 if player archetype is balanced connector. |
| `archetype_is_interior_big_finisher` | 1 if player archetype is interior big finisher. |
| `archetype_is_lead_creator_guard` | 1 if player archetype is lead creator guard. |
| `archetype_is_rim_protector_big` | 1 if player archetype is rim protector big. |
| `archetype_is_scoring_guard_wing` | 1 if player archetype is scoring guard/wing. |
| `archetype_is_stretch_big` | 1 if player archetype is stretch big. |
| `archetype_is_three_and_d_wing` | 1 if player archetype is three-and-D wing. |
| `archetype_scarcity_index` | Season-level scarcity score for that archetype. |
| `archetype_share_in_season` | Share of players in that archetype during the season. |
| `assists_pg` | Assists per game. |
| `bbr_bpm` | Basketball-Reference Box Plus/Minus. |
| `bbr_per` | Basketball-Reference Player Efficiency Rating. |
| `bbr_vorp` | Basketball-Reference Value Over Replacement Player. |
| `bbr_ws_per_48` | Basketball-Reference Win Shares per 48 minutes. |
| `blocks_pg` | Blocks per game. |
| `cba_bucket_is_0_2` | 1 if years-of-service bucket is 0-2. |
| `cba_bucket_is_10_plus` | 1 if years-of-service bucket is 10+. |
| `cba_bucket_is_3_6` | 1 if years-of-service bucket is 3-6. |
| `cba_bucket_is_7_9` | 1 if years-of-service bucket is 7-9. |
| `def_rebounds_pg` | Defensive rebounds per game. |
| `effective_fg_pct` | Effective field-goal percentage. |
| `fg_attempted_pg` | Field-goal attempts per game. |
| `fouls_pg` | Personal fouls per game. |
| `ft_attempted_pg` | Free-throw attempts per game. |
| `ft_pct` | Free-throw percentage. |
| `games` | Games played. |
| `games_missed` | Estimated games missed in the season. |
| `games_missed_rate` | Games missed as a proportion of team games. |
| `games_started` | Games started. |
| `height_inches` | Height in inches. |
| `minutes_per_game` | Minutes per game. |
| `off_rebounds_pg` | Offensive rebounds per game. |
| `points_pg` | Points per game. |
| `position_is_combo` | 1 if player has combo/multi-position designation. |
| `prev_all_def_any` | Prior-season indicator for All-Defense selection. |
| `prev_all_nba_any` | Prior-season indicator for All-NBA selection. |
| `prev_assists_pg` | Prior-season assists per game. |
| `prev_award_all_star` | Prior-season All-Star indicator. |
| `prev_minutes_per_game` | Prior-season minutes per game. |
| `prev_points_pg` | Prior-season points per game. |
| `prev_rebounds_pg` | Prior-season rebounds per game. |
| `prev_team_win_pct_regular` | Prior-season team regular-season win percentage. |
| `rebounds_pg` | Rebounds per game. |
| `reg_plus_minus_pg` | Plus/minus per game (regular season). |
| `rolling3_points_pg_prior` | Trailing 3-season average of points per game prior to current season. |
| `season` | Season-end year for the current row. |
| `steals_pg` | Steals per game. |
| `team_net_points_pg` | Team net points per game. |
| `team_region_central` | 1 if team is mapped to central region. |
| `team_region_east` | 1 if team is mapped to east region. |
| `team_region_west` | 1 if team is mapped to west region. |
| `team_win_pct_regular` | Team regular-season win percentage. |
| `three_pt_attempted_pg` | Three-point attempts per game. |
| `three_pt_pct` | Three-point percentage. |
| `turnovers_pg` | Turnovers per game. |
| `weight_lbs` | Weight in pounds. |
| `years_of_service` | NBA years of service for CBA/experience context. |

### 3.2 Engineered model-input features (computed at runtime, 14)

These are not direct V4 columns; they are deterministically created in the training pipeline from V4 fields.

| Engineered feature | Definition |
|---|---|
| `availability_rate` | `games / team_games_regular`. |
| `championships_through_prev_season` | Cumulative championships through prior season. |
| `draft_number_clean` | Numeric cleaned draft number (`0` for missing/undrafted). |
| `draft_round_clean` | Numeric cleaned draft round (`0` for missing/undrafted). |
| `experience` | `season - draft_year_clean`, clipped to valid range. |
| `is_big` | 1 if listed position includes PF or C. |
| `is_guard` | 1 if listed position includes PG or SG. |
| `is_undrafted` | 1 if draft data indicates undrafted/missing. |
| `is_wing` | 1 if listed position includes SF. |
| `rolling3_net_rating_prior` | Trailing 3-season average of prior `adv_net_rating`. |
| `rolling3_team_win_pct_prior` | Trailing 3-season average of prior team win %. |
| `rolling3_ts_prior` | Trailing 3-season average of prior true-shooting %. |
| `starter_share` | `games_started / games`. |
| `target_season` | Prediction target season-end year (`season + 1`). |

### 3.3 Explicit exclusion in V2

V2 excludes salary-history predictors such as:

- `prev_salary`
- `prev2_salary`
- `prev_salary_growth`
- salary volatility / raise-cut salary-history features

---

## 4. What the V2 Performance Model Learned

### RefinedFeatures (All Players)

- Best model: **Blend** (`XGB 0.40`, `CAT 0.60`).
- Good global fit quality (R² `0.8116`) with MAE around `$3.56M`.
- Top predictors: `cba_bucket_is_0_2`, `minutes_per_game`, `prev_minutes_per_game`, `cba_bucket_is_10_plus`, `points_pg`.
- Compensation bands are strongly linked to role load and service/CBA stage.
- Useful as a global valuation model, but still noisy at exact dollar level.

### Point Guard

- Best model: **LightGBM**.
- MAE around `$4.77M`, R² `0.7588`.
- Top predictors: `minutes_per_game`, `assists_pg`, `points_pg`, `target_season`, `years_of_service`.
- Playmaking + minutes + career stage are driving PG salary signal.
- Dollar precision remains limited for PG outliers.

### Shooting Guard

- Best model: **LightGBM**.
- MAE around `$3.35M`, R² `0.8047` (one of the strongest position fits).
- Top predictors: `minutes_per_game`, `cba_bucket_is_0_2`, `points_pg`, `fg_attempted_pg`, `prev_points_pg`.
- Scoring volume and role continuity are central for SG valuation.
- Strong tiering behavior; exact contract value still has residual noise.

### Power Forward

- Best model: **LightGBM**.
- MAE around `$4.85M`, R² `0.7182`.
- Top predictors: `minutes_per_game`, `points_pg`, `cba_bucket_is_0_2`, `draft_round_clean`, `years_of_service`.
- PF salaries show broad signal from production and tenure.
- Position remains one of the hardest for exact-dollar forecasting.

### Small Forward

- Best model: **Blend**.
- MAE around `$3.36M`, R² `0.7846`.
- Top predictors: `cba_bucket_is_0_2`, `minutes_per_game`, `points_pg`, `years_of_service`, `draft_round_clean`.
- Wing salary structure responds to usage, scoring, and service tier.
- Good middle-ground position model in this performance-first setup.

### Center

- Best model: **CatBoost**.
- MAE around `$4.08M`, R² `0.7067`.
- Top predictors: `minutes_per_game`, `def_rebounds_pg`, `rebounds_pg`, `prev_minutes_per_game`, `years_of_service`.
- Big-man compensation signal is strongly tied to role, rebounding, and durability.
- Contract structure effects still create large residual error for some players.

---

## 5. Current Limitations and Checks

- V2 quick-run settings (`n_iter=2`, `cv=2`) are intentionally fast; use full run for publication-grade estimates.
- In current snapshots, naive salary persistence can still beat V2 MAE, which indicates unresolved contract-structure effects.
- `LightGBM` feature-name warnings are expected sklearn/lightgbm interface warnings and do not invalidate metrics.

Recommended next validation step:

- Run full V2 (`n_iter=6`, `cv=3`) and report both quick/full metrics in versioned results.

---

## 6. Reproducible Run Commands (V2)

From `NBA Data/PositionModelsPreseasonPerformanceV1`:

Quick run:

```bash
NBAML_CSV_PATH="/Users/amritdhillon/Desktop/Advanced ML/Final Project/NBA Data/NBADataCleanV4.csv" \
NBAML_DISABLE_XGB=0 \
NBAML_DISABLE_LGBM=0 \
NBAML_PRESEASON_N_ITER_SINGLE=2 \
NBAML_PRESEASON_CV_SPLITS=2 \
NBAML_BLEND_WEIGHT_STEP=0.20 \
NBAML_SELECTION_METRIC=usd_mae \
NBAML_FORCE_RETRAIN=1 \
bash run_fast_dev.sh
```

Full run:

```bash
NBAML_CSV_PATH="/Users/amritdhillon/Desktop/Advanced ML/Final Project/NBA Data/NBADataCleanV4.csv" \
NBAML_DISABLE_XGB=0 \
NBAML_DISABLE_LGBM=0 \
NBAML_PRESEASON_N_ITER_SINGLE=6 \
NBAML_PRESEASON_CV_SPLITS=3 \
NBAML_BLEND_WEIGHT_STEP=0.10 \
NBAML_SELECTION_METRIC=usd_mae \
NBAML_FORCE_RETRAIN=1 \
bash run_all_models.sh
```

Outputs:

- `NBA Data/PositionModelsPreseasonPerformanceV1/results/*_metrics.json`

---

## 7. Summary

- **V1** remains best for pure predictive accuracy because it uses salary path dependence.
- **V2** is better aligned with the project question of performance-driven salary prediction.
- The current V2 package is ready to publish as a performance-first benchmark, with clear limitations documented.
