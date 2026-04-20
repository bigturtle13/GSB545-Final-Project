import json
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NBA_DATA_DIR = os.path.dirname(THIS_DIR)
DEFAULT_CSV = os.path.join(NBA_DATA_DIR, "NBACleanDataV2.csv")
RESULTS_DIR = os.path.join(THIS_DIR, "results")

TRAIN_GAMES_THRESHOLD = int(os.environ.get("NBAML_TRAIN_GAMES_THRESHOLD", "15"))
SELECTION_METRIC = os.environ.get("NBAML_SELECTION_METRIC", "usd_mae").strip().lower()
VALID_SELECTION_METRICS = {"usd_mae", "usd_rmse", "log_mae", "log_rmse"}
if SELECTION_METRIC not in VALID_SELECTION_METRICS:
    SELECTION_METRIC = "usd_mae"

CACHE_ENABLED = os.environ.get("NBAML_USE_CACHE", "1") == "1"
FORCE_RETRAIN = os.environ.get("NBAML_FORCE_RETRAIN", "0") == "1"
ENABLE_XGB = os.environ.get("NBAML_DISABLE_XGB", "0") != "1"
ENABLE_LGBM = os.environ.get("NBAML_DISABLE_LGBM", "0") != "1"
VERSION_TAG = "stable_v1.0.0"


def sdiv(num, den):
    den = den.replace(0, np.nan)
    return num / den


def _metrics_dict(y_true_log, pred_log):
    rmse_log = np.sqrt(mean_squared_error(y_true_log, pred_log))
    mae_log = mean_absolute_error(y_true_log, pred_log)
    r2_log = r2_score(y_true_log, pred_log)

    y_true_usd = np.exp(y_true_log)
    pred_usd = np.exp(pred_log)

    rmse_usd = np.sqrt(mean_squared_error(y_true_usd, pred_usd))
    mae_usd = mean_absolute_error(y_true_usd, pred_usd)
    r2_usd = r2_score(y_true_usd, pred_usd)
    return {
        "r2_log": float(r2_log),
        "rmse_log": float(rmse_log),
        "mae_log": float(mae_log),
        "r2_usd": float(r2_usd),
        "rmse_usd": float(rmse_usd),
        "mae_usd": float(mae_usd),
    }


def eval_preds(name, y_true_log, pred_log):
    metrics = _metrics_dict(y_true_log, pred_log)
    print(f"\n{name}")
    print("R2 (log):", metrics["r2_log"])
    print("RMSE (log):", metrics["rmse_log"])
    print("MAE (log):", metrics["mae_log"])
    print("R2 (USD):", metrics["r2_usd"])
    print("RMSE (USD):", metrics["rmse_usd"])
    print("MAE (USD):", metrics["mae_usd"])
    return metrics


def _selection_score(y_true_log, pred_log, metric=SELECTION_METRIC):
    if metric.startswith("usd_"):
        y_true = np.exp(y_true_log)
        pred = np.exp(pred_log)
    else:
        y_true = y_true_log
        pred = pred_log

    if metric.endswith("_mae"):
        return float(mean_absolute_error(y_true, pred))
    return float(np.sqrt(mean_squared_error(y_true, pred)))


def _position_mask(position_series, code):
    pos = position_series.fillna("").astype(str).str.upper()
    pattern = rf"(?:^|-){re.escape(code.upper())}(?:-|$)"
    return pos.str.contains(pattern, regex=True, na=False)


def _cache_signature(model_name, position_code, csv_path):
    dataset_stat = os.stat(csv_path)
    return {
        "version_tag": VERSION_TAG,
        "model_name": model_name,
        "position_code": position_code,
        "dataset_path": os.path.abspath(csv_path),
        "dataset_size": int(dataset_stat.st_size),
        "dataset_mtime": int(dataset_stat.st_mtime),
        "selection_metric": SELECTION_METRIC,
        "train_games_threshold": TRAIN_GAMES_THRESHOLD,
        "xgb_enabled": bool(XGB_AVAILABLE and ENABLE_XGB),
        "catboost_enabled": bool(CATBOOST_AVAILABLE),
        "lightgbm_enabled": bool(LIGHTGBM_AVAILABLE and ENABLE_LGBM),
    }


def _load_cached_results_if_valid(model_name, signature):
    if not CACHE_ENABLED or FORCE_RETRAIN:
        return None
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if payload.get("cache_signature") != signature:
        return None
    return payload


def _print_cached_summary(payload, model_name):
    print(f"Loaded cached results for {model_name}.")
    print(f"Cached timestamp: {payload.get('timestamp_utc')}")
    print(f"Selected model: {payload.get('auto_selected_model')}")
    selected = payload.get("auto_selected_metrics", {})
    if selected:
        print("\nAuto-Selected (cached)")
        print("R2 (log):", selected.get("r2_log"))
        print("RMSE (log):", selected.get("rmse_log"))
        print("MAE (log):", selected.get("mae_log"))
        print("R2 (USD):", selected.get("r2_usd"))
        print("RMSE (USD):", selected.get("rmse_usd"))
        print("MAE (USD):", selected.get("mae_usd"))
    print(f"Cached JSON: {os.path.join(RESULTS_DIR, f'{model_name}_metrics.json')}")


def _save_results_json(model_name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics JSON: {out_path}")


def _ensure_engineered_columns(df):
    df = df.copy()
    df = df[df["salary"] > 0].copy()
    df = df.sort_values(["player_name", "season"]).reset_index(drop=True)
    g = df.groupby("player_name", group_keys=False)

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["log_salary"] = np.log(df["salary"])

    valid_draft = df["draft_year"].where((df["draft_year"] >= 1960) & (df["draft_year"] <= df["season"]))
    rookie_season = df.groupby("player_name")["season"].transform("min")
    df["draft_year_clean"] = valid_draft.fillna(rookie_season)
    df["experience"] = (df["season"] - df["draft_year_clean"]).clip(lower=0, upper=25)

    df["starter_share"] = sdiv(df["games_started"], df["games"])
    df["availability_rate"] = sdiv(df["games"], df["team_games_regular"])

    pos = df["position"].fillna("").astype(str).str.upper()
    df["is_guard"] = pos.str.contains("PG|SG", regex=True).astype(int)
    df["is_wing"] = pos.str.contains("SF", regex=True).astype(int)
    df["is_big"] = pos.str.contains("PF|C", regex=True).astype(int)

    draft_round_num = pd.to_numeric(df["draft_round"], errors="coerce")
    draft_number_num = pd.to_numeric(df["draft_number"], errors="coerce")
    is_undrafted = (
        draft_round_num.isna()
        | draft_number_num.isna()
        | (draft_round_num <= 0)
        | (draft_number_num <= 0)
    )
    df["is_undrafted"] = is_undrafted.astype(int)
    df["draft_round_clean"] = draft_round_num.fillna(0)
    df["draft_number_clean"] = draft_number_num.fillna(0)

    df["prev_salary"] = g["salary"].shift(1)
    df["prev2_salary"] = g["salary"].shift(2)
    df["prev_salary_growth"] = sdiv(df["prev_salary"] - df["prev2_salary"], df["prev2_salary"])
    df["salary_volatility_3yr"] = g["salary"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    df["big_raise_last_year"] = (df["prev_salary_growth"] > 0.50).astype(float)
    df["big_cut_last_year"] = (df["prev_salary_growth"] < -0.20).astype(float)

    df["prev_points_pg"] = g["points_pg"].shift(1)
    df["prev_assists_pg"] = g["assists_pg"].shift(1)
    df["prev_rebounds_pg"] = g["rebounds_pg"].shift(1)
    df["prev_minutes_per_game"] = g["minutes_per_game"].shift(1)
    df["prev_team_win_pct_regular"] = g["team_win_pct_regular"].shift(1)

    df["rolling3_points_pg_prior"] = g["points_pg"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["rolling3_ts_prior"] = g["adv_true_shooting_pct"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["rolling3_net_rating_prior"] = g["adv_net_rating"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["rolling3_team_win_pct_prior"] = g["team_win_pct_regular"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    df["all_star_selections_through_prev_season"] = g["award_all_star"].transform(
        lambda s: s.fillna(0).shift(1).fillna(0).cumsum()
    )
    df["prev_award_all_star"] = g["award_all_star"].shift(1).fillna(0)
    df["prev_all_nba_any"] = g["all_nba_any"].shift(1).fillna(0)
    df["prev_all_def_any"] = g["all_def_any"].shift(1).fillna(0)

    if "championships_won_through_season" in df.columns:
        df["championships_through_prev_season"] = (
            g["championships_won_through_season"].shift(1).fillna(0)
        )
    elif "won_championship" in df.columns:
        df["championships_through_prev_season"] = g["won_championship"].transform(
            lambda s: s.fillna(0).shift(1).fillna(0).cumsum()
        )
    else:
        df["championships_through_prev_season"] = 0

    region_map = {
        "ATL": "east",
        "BOS": "east",
        "BKN": "east",
        "BRK": "east",
        "NJN": "east",
        "CHA": "east",
        "CHH": "east",
        "MIA": "east",
        "NYK": "east",
        "ORL": "east",
        "PHI": "east",
        "TOR": "east",
        "WAS": "east",
        "CHI": "central",
        "CLE": "central",
        "DET": "central",
        "IND": "central",
        "MIL": "central",
        "DAL": "west",
        "DEN": "west",
        "GSW": "west",
        "HOU": "west",
        "LAC": "west",
        "LAL": "west",
        "MEM": "west",
        "MIN": "west",
        "NOP": "west",
        "NOH": "west",
        "NOK": "west",
        "OKC": "west",
        "PHX": "west",
        "PHO": "west",
        "POR": "west",
        "SAC": "west",
        "SAS": "west",
        "UTA": "west",
        "SEA": "west",
        "VAN": "west",
    }
    team_region = df["team_abbr"].map(region_map).fillna("unknown")
    df["team_region_east"] = (team_region == "east").astype(int)
    df["team_region_west"] = (team_region == "west").astype(int)
    df["team_region_central"] = (team_region == "central").astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _core_features():
    return [
        "season",
        "age",
        "experience",
        "height_inches",
        "weight_lbs",
        "draft_round_clean",
        "draft_number_clean",
        "is_undrafted",
        "is_guard",
        "is_wing",
        "is_big",
        "team_region_east",
        "team_region_west",
        "team_region_central",
        "games",
        "minutes_per_game",
        "starter_share",
        "availability_rate",
        "points_pg",
        "assists_pg",
        "rebounds_pg",
        "turnovers_pg",
        "three_pt_pct",
        "ft_pct",
        "effective_fg_pct",
        "adv_usage_pct",
        "adv_true_shooting_pct",
        "adv_net_rating",
        "adv_def_rating",
        "reg_plus_minus_pg",
        "team_win_pct_regular",
        "team_net_points_pg",
        "prev_salary",
        "prev2_salary",
        "prev_salary_growth",
        "salary_volatility_3yr",
        "big_raise_last_year",
        "big_cut_last_year",
        "prev_points_pg",
        "prev_assists_pg",
        "prev_rebounds_pg",
        "prev_minutes_per_game",
        "rolling3_points_pg_prior",
        "rolling3_ts_prior",
        "rolling3_net_rating_prior",
        "rolling3_team_win_pct_prior",
        "prev_team_win_pct_regular",
        "all_star_selections_through_prev_season",
        "prev_award_all_star",
        "prev_all_nba_any",
        "prev_all_def_any",
        "championships_through_prev_season",
    ]


def _feature_pack_map(position_code):
    core = _core_features()
    guard_plus = core + [
        "games_started",
        "fg_attempted_pg",
        "ft_attempted_pg",
        "three_pt_attempted_pg",
        "adv_ast_to",
        "steals_pg",
    ]
    frontcourt_plus = core + [
        "games_started",
        "fg_attempted_pg",
        "ft_attempted_pg",
        "three_pt_attempted_pg",
        "adv_ast_to",
        "steals_pg",
        "blocks_pg",
        "off_rebounds_pg",
        "def_rebounds_pg",
        "fouls_pg",
    ]
    key = None if position_code is None else position_code.upper()
    if key == "PG":
        return {"core_v1": core, "guard_plus_v1": guard_plus}
    if key in {"PF", "C"}:
        return {"core_v1": core, "frontcourt_plus_v1": frontcourt_plus}
    if key == "SF":
        return {"core_v1": core, "frontcourt_plus_v1": frontcourt_plus}
    if key == "SG":
        return {"core_v1": core, "guard_plus_v1": guard_plus}
    return {"core_v1": core}


def _default_xgb_params(position_code):
    base = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 1,
        "learning_rate": 0.02,
        "max_depth": 6,
        "min_child_weight": 2,
        "n_estimators": 1300,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
    }
    key = None if position_code is None else position_code.upper()
    by_pos = {
        "PG": {"max_depth": 5, "n_estimators": 1400, "learning_rate": 0.018},
        "SG": {"max_depth": 5, "n_estimators": 1400, "learning_rate": 0.018},
        "PF": {"max_depth": 6, "n_estimators": 1300, "learning_rate": 0.02},
        "SF": {"max_depth": 6, "n_estimators": 1300, "learning_rate": 0.02},
        "C": {"max_depth": 4, "min_child_weight": 4, "n_estimators": 1500, "learning_rate": 0.017},
    }
    base.update(by_pos.get(key, {}))
    return base


def _default_cat_params(position_code):
    base = {
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "depth": 6,
        "learning_rate": 0.03,
        "n_estimators": 1600,
        "l2_leaf_reg": 6.0,
        "random_seed": 42,
        "verbose": 0,
    }
    key = None if position_code is None else position_code.upper()
    by_pos = {
        None: {"depth": 6, "learning_rate": 0.025, "n_estimators": 1700, "l2_leaf_reg": 7.0},
        "PF": {"depth": 5, "learning_rate": 0.025, "n_estimators": 1800, "l2_leaf_reg": 8.0},
        "SG": {"depth": 6, "learning_rate": 0.028, "n_estimators": 1700, "l2_leaf_reg": 6.0},
    }
    base.update(by_pos.get(key, {}))
    return base


def _default_lgbm_params(position_code):
    base = {
        "objective": "regression",
        "random_state": 42,
        "n_estimators": 1300,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "n_jobs": 1,
        "verbosity": -1,
    }
    key = None if position_code is None else position_code.upper()
    by_pos = {
        "PG": {"num_leaves": 23, "learning_rate": 0.022},
        "PF": {"num_leaves": 39, "learning_rate": 0.018},
        "C": {"num_leaves": 39, "learning_rate": 0.018},
    }
    base.update(by_pos.get(key, {}))
    return base


def _numeric_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), make_column_selector(dtype_include=np.number))
        ],
        remainder="drop",
    )


def _fit_predict_xgb(position_code, X_train, y_train, X_pred):
    pre = _numeric_preprocessor()
    model = XGBRegressor(**_default_xgb_params(position_code))
    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_pred)
    return pred, pipe


def _fit_predict_cat(position_code, X_train, y_train, X_pred):
    model = CatBoostRegressor(**_default_cat_params(position_code))
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_pred_i = imputer.transform(X_pred)
    model.fit(X_train_i, y_train)
    pred = model.predict(X_pred_i)
    return pred, model, imputer


def _fit_predict_lgbm(position_code, X_train, y_train, X_pred):
    pre = _numeric_preprocessor()
    model = LGBMRegressor(**_default_lgbm_params(position_code))
    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_pred)
    return pred, pipe


def _blend_weight_from_validation(y_val, pred_xgb_val, pred_cat_val):
    weights = np.arange(0.0, 1.01, 0.05)
    best_w, best_score = 0.5, float("inf")
    for w in weights:
        pred = w * pred_xgb_val + (1 - w) * pred_cat_val
        score = _selection_score(y_val, pred, metric=SELECTION_METRIC)
        if score < best_score:
            best_score = score
            best_w = float(w)
    return best_w, float(best_score)


def run_experiment(model_name, position_code=None, csv_path=DEFAULT_CSV, model_choice="auto"):
    csv_path = os.environ.get("NBAML_CSV_PATH", csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find dataset: {csv_path}")

    signature = _cache_signature(model_name, position_code, csv_path)
    cached = _load_cached_results_if_valid(model_name, signature)
    if cached is not None:
        _print_cached_summary(cached, model_name)
        return cached

    started_at = time.time()
    df = pd.read_csv(csv_path)
    df = _ensure_engineered_columns(df)

    if position_code is not None:
        df = df.loc[_position_mask(df["position"], position_code)].copy()
        print(f"Running {model_name} on position={position_code} | rows={len(df):,}")
    else:
        print(f"Running {model_name} on all positions | rows={len(df):,}")
    print(f"Selected model strategy: {model_choice}")
    print(f"Selection metric: {SELECTION_METRIC}")
    print(f"Training games threshold: >= {TRAIN_GAMES_THRESHOLD}")

    if df.empty:
        print("No rows after filtering. Exiting.")
        return None

    y = df["log_salary"]

    eligible_train = df["games"] >= TRAIN_GAMES_THRESHOLD
    train_mask_recent = (df["season"] <= 2020) & eligible_train
    blend_val_mask = df["season"].between(2021, 2022) & eligible_train
    tune_mask = (df["season"] <= 2022) & eligible_train
    holdout_mask = df["season"] >= 2023

    if int(train_mask_recent.sum()) == 0 or int(blend_val_mask.sum()) == 0 or int(holdout_mask.sum()) == 0:
        print("Not enough rows in train/validation/holdout split. Exiting.")
        return None

    print(
        "Rows - train:",
        int(train_mask_recent.sum()),
        "blend_val:",
        int(blend_val_mask.sum()),
        "holdout:",
        int(holdout_mask.sum()),
    )

    xgb_allowed = bool(XGB_AVAILABLE and ENABLE_XGB)
    cat_allowed = bool(CATBOOST_AVAILABLE)
    lgbm_allowed = bool(LIGHTGBM_AVAILABLE and ENABLE_LGBM)

    if model_choice == "auto":
        model_options = []
        if xgb_allowed:
            model_options.append("xgb")
        if cat_allowed:
            model_options.append("catboost")
        if lgbm_allowed:
            model_options.append("lightgbm")
        if xgb_allowed and cat_allowed:
            model_options.append("blend")
    elif model_choice == "xgb":
        model_options = ["xgb"] if xgb_allowed else []
    elif model_choice == "catboost":
        model_options = ["catboost"] if cat_allowed else []
    elif model_choice == "lightgbm":
        model_options = ["lightgbm"] if lgbm_allowed else []
    else:
        model_options = ["blend"] if (xgb_allowed and cat_allowed) else []

    if not model_options:
        print("No available model options. Install xgboost/catboost/lightgbm as needed.")
        return None

    pack_map = _feature_pack_map(position_code)
    val_candidates = []

    selection_started = time.time()
    for pack_name, pack_cols in pack_map.items():
        features = [c for c in pack_cols if c in df.columns]
        if not features:
            continue
        X_pack = df[features].copy()
        for c in features:
            X_pack[c] = pd.to_numeric(X_pack[c], errors="coerce")

        X_train = X_pack.loc[train_mask_recent]
        y_train = y.loc[train_mask_recent]
        X_val = X_pack.loc[blend_val_mask]
        y_val = y.loc[blend_val_mask]

        preds_val = {}
        if "xgb" in model_options or "blend" in model_options:
            pred_xgb_val, _ = _fit_predict_xgb(position_code, X_train, y_train, X_val)
            preds_val["xgb"] = pred_xgb_val
            if "xgb" in model_options:
                val_candidates.append(
                    {
                        "pack": pack_name,
                        "model": "xgb",
                        "features": features,
                        "score": _selection_score(y_val, pred_xgb_val, metric=SELECTION_METRIC),
                    }
                )

        if "catboost" in model_options or "blend" in model_options:
            pred_cat_val, _, _ = _fit_predict_cat(position_code, X_train, y_train, X_val)
            preds_val["catboost"] = pred_cat_val
            if "catboost" in model_options:
                val_candidates.append(
                    {
                        "pack": pack_name,
                        "model": "catboost",
                        "features": features,
                        "score": _selection_score(y_val, pred_cat_val, metric=SELECTION_METRIC),
                    }
                )

        if "lightgbm" in model_options:
            pred_lgbm_val, _ = _fit_predict_lgbm(position_code, X_train, y_train, X_val)
            val_candidates.append(
                {
                    "pack": pack_name,
                    "model": "lightgbm",
                    "features": features,
                    "score": _selection_score(y_val, pred_lgbm_val, metric=SELECTION_METRIC),
                }
            )

        if "blend" in model_options and "xgb" in preds_val and "catboost" in preds_val:
            w, blend_score = _blend_weight_from_validation(y_val, preds_val["xgb"], preds_val["catboost"])
            val_candidates.append(
                {
                    "pack": pack_name,
                    "model": "blend",
                    "features": features,
                    "score": blend_score,
                    "blend_weight": w,
                }
            )

    if not val_candidates:
        print("No valid validation candidates found.")
        return None

    val_candidates = sorted(
        val_candidates,
        key=lambda x: (x["score"], 1 if x["model"] == "blend" else 0),
    )
    best = val_candidates[0]
    model_features = best["features"]
    selected_pack = best["pack"]
    selected_model = best["model"]
    selection_runtime = time.time() - selection_started

    print(f"Model features used: {len(model_features)}")
    print(f"Selected feature pack by validation: {selected_pack}")
    print(f"Selected model by validation: {selected_model}")
    print(f"Validation {SELECTION_METRIC}: {best['score']:.6f}")
    print("Top validation candidates:")
    for c in val_candidates[:6]:
        extras = ""
        if c["model"] == "blend":
            extras = f" w={c.get('blend_weight', 0.5):.2f}"
        print(
            f"  pack={c['pack']:<18} model={c['model']:<8} "
            f"{SELECTION_METRIC}={c['score']:.6f}{extras}"
        )

    X = df[model_features].copy()
    for c in model_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y_holdout = y.loc[holdout_mask]

    preds = {}
    metrics = {}
    xgb_pipe_final = None

    if xgb_allowed:
        pred_xgb_holdout, xgb_pipe_final = _fit_predict_xgb(
            position_code, X.loc[tune_mask], y.loc[tune_mask], X.loc[holdout_mask]
        )
        preds["xgb"] = pred_xgb_holdout
        metrics["xgb"] = eval_preds("XGBoost (final)", y_holdout, pred_xgb_holdout)

    if cat_allowed:
        pred_cat_holdout, _, _ = _fit_predict_cat(
            position_code, X.loc[tune_mask], y.loc[tune_mask], X.loc[holdout_mask]
        )
        preds["catboost"] = pred_cat_holdout
        metrics["catboost"] = eval_preds("CatBoost (final)", y_holdout, pred_cat_holdout)

    if lgbm_allowed:
        pred_lgbm_holdout, _ = _fit_predict_lgbm(
            position_code, X.loc[tune_mask], y.loc[tune_mask], X.loc[holdout_mask]
        )
        preds["lightgbm"] = pred_lgbm_holdout
        metrics["lightgbm"] = eval_preds("LightGBM (final)", y_holdout, pred_lgbm_holdout)

    final_blend_weight = None
    if "xgb" in preds and "catboost" in preds:
        pred_xgb_val, _ = _fit_predict_xgb(
            position_code, X.loc[train_mask_recent], y.loc[train_mask_recent], X.loc[blend_val_mask]
        )
        pred_cat_val, _, _ = _fit_predict_cat(
            position_code, X.loc[train_mask_recent], y.loc[train_mask_recent], X.loc[blend_val_mask]
        )
        final_blend_weight, blend_val_score = _blend_weight_from_validation(
            y.loc[blend_val_mask], pred_xgb_val, pred_cat_val
        )
        print(
            f"Best blend weight on 2021-2022 (XGB weight): {final_blend_weight:.2f}, "
            f"{SELECTION_METRIC}: {blend_val_score:.6f}"
        )
        pred_blend_holdout = final_blend_weight * preds["xgb"] + (1 - final_blend_weight) * preds["catboost"]
        preds["blend"] = pred_blend_holdout
        metrics["blend"] = eval_preds(
            f"Blend (final, XGB={final_blend_weight:.2f}, CAT={1-final_blend_weight:.2f})",
            y_holdout,
            pred_blend_holdout,
        )

    selected_model_name = selected_model if selected_model in preds else None
    if selected_model_name is None:
        for fallback in ["blend", "xgb", "catboost", "lightgbm"]:
            if fallback in preds:
                selected_model_name = fallback
                break

    selected_metrics = eval_preds(
        f"Auto-Selected ({selected_model_name})",
        y_holdout,
        preds[selected_model_name],
    )

    if xgb_pipe_final is not None:
        fi = pd.Series(
            xgb_pipe_final.named_steps["model"].feature_importances_,
            index=model_features,
        ).sort_values(ascending=False)
        print("\nTop 20 XGB feature importances:")
        print(fi.head(20))
        fi_top20 = fi.head(20).to_dict()
    else:
        fi_top20 = {}

    total_runtime = time.time() - started_at
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cache_signature": signature,
        "model_name": model_name,
        "position_code": position_code,
        "dataset": os.path.abspath(csv_path),
        "selection_metric": SELECTION_METRIC,
        "train_games_threshold": TRAIN_GAMES_THRESHOLD,
        "rows": {
            "total": int(len(df)),
            "train": int(train_mask_recent.sum()),
            "blend_val": int(blend_val_mask.sum()),
            "tune": int(tune_mask.sum()),
            "holdout": int(holdout_mask.sum()),
        },
        "feature_pack_selected": selected_pack,
        "feature_count": int(len(model_features)),
        "feature_names": model_features,
        "runtime_seconds": {
            "selection": float(selection_runtime),
            "total": float(total_runtime),
        },
        "library_availability": {
            "xgboost": bool(xgb_allowed),
            "catboost": bool(cat_allowed),
            "lightgbm": bool(lgbm_allowed),
        },
        "validation_candidates": val_candidates,
        "final_blend_weight": final_blend_weight,
        "final_metrics": metrics,
        "auto_selected_model": selected_model_name,
        "auto_selected_metrics": selected_metrics,
        "xgb_top20_importance": fi_top20,
    }
    _save_results_json(model_name, payload)
    return payload
