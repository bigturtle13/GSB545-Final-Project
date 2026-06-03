# GSB545-Final-Project
# NBA Salary Prediction Using Machine Learning

## Project Overview

This project predicts NBA player salary in season **t+1** using information available through season **t**.

Two modeling tracks were developed:

- **V1 (Stable Contract-Path Model)** – Includes salary-history variables and maximizes predictive accuracy.
- **V2 (Performance-Based Model)** – Excludes salary-history variables and evaluates how much future salary can be explained by player performance, role, durability, experience, and team context.

---

## Key Results

### V1: Stable Contract-Path Model

| Model | Algorithm | MAE | RMSE | R² |
|---------|---------|---------:|---------:|---------:|
| Global | CatBoost | $2.80M | $4.97M | 0.822 |
| PG | Blend | $4.10M | $7.33M | 0.737 |
| SG | CatBoost | $2.73M | $4.76M | 0.760 |
| SF | Blend | $2.76M | $5.10M | 0.788 |
| PF | CatBoost | $2.96M | $4.77M | 0.853 |
| C | LightGBM | $3.14M | $5.41M | 0.757 |

### V2: Performance-Based Model

| Model | Algorithm | MAE | RMSE | R² |
|---------|---------|---------:|---------:|---------:|
| Global | Blend | $3.56M | $5.50M | 0.812 |
| PG | LightGBM | $4.77M | $7.40M | 0.759 |
| SG | LightGBM | $3.35M | $5.01M | 0.805 |
| SF | Blend | $3.36M | $5.25M | 0.785 |
| PF | LightGBM | $4.85M | $7.26M | 0.718 |
| C | CatBoost | $4.08M | $6.23M | 0.707 |

---

## Dataset

- Seasons: 2000–2025
- ~10,000 player-season observations
- Traditional box-score statistics
- Advanced metrics (PER, BPM, VORP, WS/48, TS%, Usage)
- Team context variables
- Career and award history
- Archetype-based features
- CBA service bucket features

### Target Variable

`next_salary` = player salary in season t+1

---

## Models

- XGBoost
- CatBoost
- LightGBM
- Blended Ensemble (XGBoost + CatBoost)
- Neural Network Baseline

---

## Major Findings

1. Salary history improves forecasting accuracy.
2. Performance-only models still explain over 80% of salary variance.
3. Minutes played is consistently the strongest predictor.
4. CBA service buckets are among the most important features.
5. Shooting Guards and Small Forwards are easiest to model.
6. Power Forwards and Centers are most difficult to model.

---

## Reproducing Results

### Quick Run

```bash
NBAML_PRESEASON_N_ITER_SINGLE=2
NBAML_PRESEASON_CV_SPLITS=2
bash run_fast_dev.sh
```

### Full Run

```bash
NBAML_PRESEASON_N_ITER_SINGLE=6
NBAML_PRESEASON_CV_SPLITS=3
bash run_all_models.sh
```

---

## Conclusion

NBA salaries are highly predictable using machine learning. While salary-history variables produce the strongest forecasting accuracy, performance-based models still explain most future salary variation using only basketball-related information, making them valuable tools for player valuation, contract analysis, and roster planning.
