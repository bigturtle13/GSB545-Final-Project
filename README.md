# GSB545-Final-Project
NBA Salary Prediction ML Model

Date: 2026-04-20  

## 1. Project Objective

Predict an NBA player's season salary using:

- player production
- advanced metrics
- role/availability
- draft/background context
- lag and rolling history
- awards/championship reputation

The goal was to keep the model package:

- accurate in USD error terms
- faster to run than the heavier tuned versions
- reproducible for teammates with cached JSON outputs

## 2. Modeling Setup

### Time setup

- Seasons covered: 2000-2025
- Holdout evaluation: 2023-2025
- Validation window for model/feature-pack selection: 2021-2022
- Training filter: `games >= 15`

Why `games >= 15`: this removes very small-sample seasons that inject high variance and noisy salary relationships.

### Models compared in each script

- XGBoost
- CatBoost
- LightGBM
- XGBoost/CatBoost blend

Selection criterion used: `usd_mae`.

### Position-specific scripts

- All players: `RefinedFeaturesV1.py`
- Point Guard: `PointGuardModelV1.py`
- Shooting Guard: `ShootingGuardModelV1.py`
- Power Forward: `PowerForwardModelV1.py`
- Small Forward: `SmallForwardModelV1.py`
- Center: `CenterModelV1.py`

## 3. Data and Feature Dictionary (Used in Stable Package)

### Core identifiers and context

- `season`: NBA season year
- `position`: player listed position(s)
- `team_abbr`: team abbreviation
- `age`, `height_inches`, `weight_lbs`
- `draft_year`, `draft_round`, `draft_number`

### Target

- `salary` (USD, season salary)
- modeled target: `log_salary = log(salary)`

### Engineered availability/role features

- `starter_share`
- `availability_rate`
- `is_guard`, `is_wing`, `is_big`
- `is_undrafted`
- `draft_round_clean`, `draft_number_clean`

### Lag and trend features (time-safe)

- `prev_salary`, `prev2_salary`, `prev_salary_growth`
- `salary_volatility_3yr`
- `big_raise_last_year`, `big_cut_last_year`
- `prev_points_pg`, `prev_assists_pg`, `prev_rebounds_pg`
- `prev_minutes_per_game`, `prev_team_win_pct_regular`
- `rolling3_points_pg_prior`, `rolling3_ts_prior`, `rolling3_net_rating_prior`, `rolling3_team_win_pct_prior`

### Reputation/history features

- `all_star_selections_through_prev_season`
- `prev_award_all_star`
- `prev_all_nba_any`
- `prev_all_def_any`
- `championships_through_prev_season`

### Team region features

- `team_region_east`
- `team_region_west`
- `team_region_central`

### Position add-on packs

- `guard_plus_v1`: adds guard-relevant shot/ballhandling volume features
- `frontcourt_plus_v1`: adds big-man interior/defense/rebounding context features

## 4. Metric Definitions

- `MAE (USD)`: average absolute salary error in dollars
- `RMSE (USD)`: square-rooted mean squared error in dollars (penalizes big misses more)
- `R² (USD)`: variance explained in salary scale
- log-space metrics were also tracked to stabilize modeling, but model selection here used `usd_mae`

## 5. Final Holdout Results (2023-2025)

### Selected model by script

| Script | Selected Model | MAE (USD) | RMSE (USD) | R² (USD) | Feature Pack |
|---|---:|---:|---:|---:|---|
| RefinedFeaturesV1 | CatBoost | 2,797,707 | 4,966,766 | 0.82199 | core_v1 |
| PointGuardModelV1 | Blend | 4,097,235 | 7,328,379 | 0.73725 | guard_plus_v1 |
| ShootingGuardModelV1 | CatBoost | 2,733,830 | 4,760,295 | 0.75966 | guard_plus_v1 |
| PowerForwardModelV1 | CatBoost | 2,958,632 | 4,773,451 | 0.85339 | core_v1 |
| SmallForwardModelV1 | Blend | 2,763,402 | 5,098,684 | 0.78764 | core_v1 |
| CenterModelV1 | LightGBM | 3,138,772 | 5,411,882 | 0.75661 | frontcourt_plus_v1 |

## 6. Do These Align With the “Best Models So Far” Goal?

Comparison against latest heavy `PositionModelsV1` JSON outputs:

| Script | Stable MAE (USD) | Prior V1 MAE (USD) | Delta (Stable - V1) |
|---|---:|---:|---:|
| RefinedFeaturesV1 | 2,797,707 | 2,772,176 | +25,531 |
| PointGuardModelV1 | 4,097,235 | 4,242,398 | -145,163 |
| ShootingGuardModelV1 | 2,733,830 | 2,739,200 | -5,370 |
| PowerForwardModelV1 | 2,958,632 | 3,021,889 | -63,256 |
| SmallForwardModelV1 | 2,763,402 | 2,750,718 | +12,684 |
| CenterModelV1 | 3,138,772 | 3,266,729 | -127,957 |

Summary:

- Better in 4 of 6 model scripts
- Slightly worse in 2 of 6 scripts (Refined and SF, both by < 1%)
- Average MAE improvement across all six: about **50,589 USD** (about **1.62% better**)

Runtime comparison:

- Stable total runtime (all 6): about **118.65s**
- Prior V1 total runtime (all 6): about **1596.56s**
- Speedup: about **13.46x faster**

Conclusion: this stable package meets the stated goal of using the stronger practical setup (accuracy-speed-reproducibility balance) for team deployment.

## 7. Interpreting the Learned Salary Signal

Across position runs, consistently strong drivers include:

- `prev_salary`
- `prev_minutes_per_game`
- `rolling3_points_pg_prior`
- `experience`
- `is_undrafted`
- `prev_points_pg`

Interpretation:

- salary is heavily path-dependent (previous contract level is a dominant predictor)
- role stability and sustained contribution over time matter more than one-season noise
- draft pedigree still contributes, but less than prior compensation and role continuity

