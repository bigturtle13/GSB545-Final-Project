import json
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

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
DEFAULT_CSV = os.path.join(NBA_DATA_DIR, "NBADataCleanV4.csv")
RESULTS_DIR = os.path.join(THIS_DIR, "results")

RANDOM_STATE = int(os.environ.get("NBAML_RANDOM_STATE", "42"))
TRAIN_GAMES_THRESHOLD = int(os.environ.get("NBAML_TRAIN_GAMES_THRESHOLD", "15"))
SELECTION_METRIC = os.environ.get("NBAML_SELECTION_METRIC", "usd_mae").strip().lower()
VALID_SELECTION_METRICS = {"usd_mae", "usd_rmse", "log_mae", "log_rmse"}
if SELECTION_METRIC not in VALID_SELECTION_METRICS:
    SELECTION_METRIC = "usd_mae"

N_ITER_SINGLE = int(os.environ.get("NBAML_PRESEASON_N_ITER_SINGLE", "6"))
CV_SPLITS = int(os.environ.get("NBAML_PRESEASON_CV_SPLITS", "3"))
BLEND_WEIGHT_STEP = float(os.environ.get("NBAML_BLEND_WEIGHT_STEP", "0.10"))
INCLUDE_NO_SALARY_PACK = False

CACHE_ENABLED = os.environ.get("NBAML_USE_CACHE", "1") == "1"
FORCE_RETRAIN = os.environ.get("NBAML_FORCE_RETRAIN", "0") == "1"
ENABLE_XGB = os.environ.get("NBAML_DISABLE_XGB", "0") != "1"
ENABLE_LGBM = os.environ.get("NBAML_DISABLE_LGBM", "1") != "1"
MODEL_N_JOBS = int(os.environ.get("NBAML_MODEL_N_JOBS", "-1"))

VERSION_TAG = "preseason_performance_v1.0.0"


TEAM_REGION_MAP = {
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


def sdiv(num, den):
    den = den.replace(0, np.nan)
    return num / den


def _ensure_col(df, col, default=np.nan):
    if col not in df.columns:
        df[col] = default


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
    m = _metrics_dict(y_true_log, pred_log)
    print(f"\n{name}")
    print("R2 (log):", m["r2_log"])
    print("RMSE (log):", m["rmse_log"])
    print("MAE (log):", m["mae_log"])
    print("R2 (USD):", m["r2_usd"])
    print("RMSE (USD):", m["rmse_usd"])
    print("MAE (USD):", m["mae_usd"])
    return m


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


def _selection_scorer(metric=SELECTION_METRIC):
    if metric == "usd_mae":
        return make_scorer(lambda y, p: mean_absolute_error(np.exp(y), np.exp(p)), greater_is_better=False)
    if metric == "usd_rmse":
        return make_scorer(
            lambda y, p: np.sqrt(mean_squared_error(np.exp(y), np.exp(p))),
            greater_is_better=False,
        )
    if metric == "log_rmse":
        return make_scorer(lambda y, p: np.sqrt(mean_squared_error(y, p)), greater_is_better=False)
    return make_scorer(mean_absolute_error, greater_is_better=False)


def _position_mask(position_series, code):
    pos = position_series.fillna("").astype(str).str.upper()
    pattern = rf"(?:^|-){re.escape(code.upper())}(?:-|$)"
    return pos.str.contains(pattern, regex=True, na=False)


def _cache_signature(model_name, position_code, csv_path):
    st = os.stat(csv_path)
    return {
        "version_tag": VERSION_TAG,
        "model_name": model_name,
        "position_code": position_code,
        "dataset_path": os.path.abspath(csv_path),
        "dataset_size": int(st.st_size),
        "dataset_mtime": int(st.st_mtime),
        "selection_metric": SELECTION_METRIC,
        "train_games_threshold": TRAIN_GAMES_THRESHOLD,
        "n_iter_single": N_ITER_SINGLE,
        "cv_splits": CV_SPLITS,
        "blend_weight_step": BLEND_WEIGHT_STEP,
        "include_no_salary_pack": INCLUDE_NO_SALARY_PACK,
        "xgb_enabled": bool(XGB_AVAILABLE and ENABLE_XGB),
        "catboost_enabled": bool(CATBOOST_AVAILABLE),
        "lightgbm_enabled": bool(LIGHTGBM_AVAILABLE and ENABLE_LGBM),
    }


def _load_cache(model_name, signature):
    if not CACHE_ENABLED or FORCE_RETRAIN:
        return None
    os.makedirs(RESULTS_DIR, exist_ok=True)
    p = os.path.join(RESULTS_DIR, f"{model_name}_metrics.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("cache_signature") == signature:
            return d
    except Exception:
        return None
    return None


def _print_cached(payload, model_name):
    print(f"Loaded cached results for {model_name}.")
    print("Cached timestamp:", payload.get("timestamp_utc"))
    print("Selected model:", payload.get("auto_selected_model"))
    m = payload.get("auto_selected_metrics", {})
    if m:
        print("\nAuto-Selected (cached)")
        print("R2 (log):", m.get("r2_log"))
        print("RMSE (log):", m.get("rmse_log"))
        print("MAE (log):", m.get("mae_log"))
        print("R2 (USD):", m.get("r2_usd"))
        print("RMSE (USD):", m.get("rmse_usd"))
        print("MAE (USD):", m.get("mae_usd"))


def _save_results(model_name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{model_name}_metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Saved metrics JSON:", out)


def _ensure_engineered_columns(df):
    df = df.copy()

    raw_numeric_cols = [
        "season",
        "salary",
        "age",
        "height_inches",
        "weight_lbs",
        "draft_year",
        "draft_round",
        "draft_number",
        "games",
        "games_started",
        "team_games_regular",
        "minutes_per_game",
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
        "fg_attempted_pg",
        "ft_attempted_pg",
        "three_pt_attempted_pg",
        "adv_ast_to",
        "steals_pg",
        "blocks_pg",
        "off_rebounds_pg",
        "def_rebounds_pg",
        "fouls_pg",
        "award_all_star",
        "all_nba_any",
        "all_def_any",
        "championships_won_through_season",
        "won_championship",
    ]

    for col in raw_numeric_cols:
        _ensure_col(df, col, default=np.nan)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    _ensure_col(df, "player_name", default="")
    _ensure_col(df, "position", default="")
    _ensure_col(df, "team_abbr", default="")

    df = df[df["salary"] > 0].copy()
    df = df.sort_values(["player_name", "season"]).reset_index(drop=True)
    g = df.groupby("player_name", group_keys=False)

    df["next_salary"] = g["salary"].shift(-1)
    df = df[df["next_salary"] > 0].copy()
    df["next_log_salary"] = np.log(df["next_salary"])
    df["target_season"] = df["season"] + 1

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

    # Prior-season and rolling context available before the target season starts.
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
        df["championships_through_prev_season"] = g["championships_won_through_season"].shift(1).fillna(0)
    elif "won_championship" in df.columns:
        df["championships_through_prev_season"] = g["won_championship"].transform(
            lambda s: s.fillna(0).shift(1).fillna(0).cumsum()
        )
    else:
        df["championships_through_prev_season"] = 0

    team_region = df["team_abbr"].map(TEAM_REGION_MAP).fillna("unknown")
    df["team_region_east"] = (team_region == "east").astype(int)
    df["team_region_west"] = (team_region == "west").astype(int)
    df["team_region_central"] = (team_region == "central").astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _dedupe(seq):
    return list(dict.fromkeys(seq))


def _prior_core_features():
    return [
        "season",
        "target_season",
        "age",
        "experience",
        "years_of_service",
        "cba_bucket_is_0_2",
        "cba_bucket_is_3_6",
        "cba_bucket_is_7_9",
        "cba_bucket_is_10_plus",
        "height_inches",
        "weight_lbs",
        "draft_round_clean",
        "draft_number_clean",
        "is_undrafted",
        "is_guard",
        "is_wing",
        "is_big",
        "position_is_combo",
        "team_region_east",
        "team_region_west",
        "team_region_central",
        "position_share_in_season",
        "position_scarcity_index",
        "games",
        "games_missed",
        "games_missed_rate",
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
        "bbr_per",
        "bbr_ws_per_48",
        "bbr_bpm",
        "bbr_vorp",
        "team_win_pct_regular",
        "team_net_points_pg",
        "season_salary_cap",
        "season_luxury_tax_threshold",
        "season_tax_to_cap_ratio",
        "season_tax_active_flag",
        "season_first_apron_threshold",
        "season_second_apron_threshold",
        "season_apron_available_flag",
    ]


def _rolling_feature_additions():
    return [
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
        "durability_3yr_availability_mean",
        "durability_3yr_availability_std",
        "durability_3yr_availability_slope",
        "trend_3yr_points_pg_slope",
        "trend_3yr_adv_true_shooting_pct_slope",
        "trend_3yr_bbr_bpm_slope",
        "trend_3yr_bbr_ws_per_48_slope",
        "impact_metrics_imputed_flag",
        "archetype_confidence_score",
        "archetype_share_in_season",
        "archetype_scarcity_index",
        "archetype_is_lead_creator_guard",
        "archetype_is_scoring_guard_wing",
        "archetype_is_three_and_d_wing",
        "archetype_is_stretch_big",
        "archetype_is_rim_protector_big",
        "archetype_is_interior_big_finisher",
        "archetype_is_balanced_connector",
    ]


def _feature_pack_map(position_code):
    core = _prior_core_features()
    rolling = _rolling_feature_additions()

    guard_add = [
        "games_started",
        "fg_attempted_pg",
        "ft_attempted_pg",
        "three_pt_attempted_pg",
        "adv_ast_to",
        "steals_pg",
    ]
    frontcourt_add = [
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
    if key in {"PG", "SG"}:
        rolling_plus = _dedupe(core + rolling + guard_add)
    elif key in {"PF", "SF", "C"}:
        rolling_plus = _dedupe(core + rolling + frontcourt_add)
    else:
        rolling_plus = _dedupe(core + rolling + guard_add + frontcourt_add)

    return {
        "prior_perf_core_v1": core,
        "prior_perf_rolling_v1": rolling_plus,
    }


class CatBoostRegressorCompat(BaseEstimator, RegressorMixin):
    """sklearn-compatible CatBoost wrapper for RandomizedSearchCV on newer sklearn versions."""

    def __init__(
        self,
        depth=6,
        learning_rate=0.03,
        n_estimators=1000,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    ):
        self.depth = depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files

    def fit(self, X, y):
        self.imputer_ = SimpleImputer(strategy="median")
        X_i = self.imputer_.fit_transform(X)
        self.model_ = CatBoostRegressor(
            depth=self.depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=self.verbose,
            allow_writing_files=self.allow_writing_files,
        )
        self.model_.fit(X_i, y)
        self.n_features_in_ = X_i.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["imputer_", "model_"])
        X_i = self.imputer_.transform(X)
        return self.model_.predict(X_i)

    @property
    def feature_importances_(self):
        check_is_fitted(self, attributes=["model_"])
        return self.model_.feature_importances_

    def __sklearn_tags__(self):
        return super().__sklearn_tags__()


def _build_estimator(model_name):
    if model_name == "xgb":
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=MODEL_N_JOBS,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
    elif model_name == "catboost":
        return CatBoostRegressorCompat(
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        )
    elif model_name == "lightgbm":
        model = LGBMRegressor(
            objective="regression",
            random_state=RANDOM_STATE,
            n_jobs=MODEL_N_JOBS,
            verbosity=-1,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def _xgb_param_distributions():
    return {
        "model__n_estimators": [400, 700, 1000, 1300],
        "model__learning_rate": [0.02, 0.03, 0.05],
        "model__max_depth": [4, 5, 6],
        "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__gamma": [0.0, 0.1],
        "model__reg_alpha": [0.0, 0.1],
        "model__reg_lambda": [1.0, 2.0],
    }


def _cat_param_distributions():
    return {
        "depth": [4, 5, 6, 7],
        "learning_rate": [0.02, 0.03, 0.05],
        "n_estimators": [600, 900, 1200, 1500],
        "l2_leaf_reg": [3.0, 6.0, 9.0],
    }


def _lgbm_param_distributions():
    return {
        "model__n_estimators": [400, 700, 1000, 1300],
        "model__learning_rate": [0.02, 0.03, 0.05],
        "model__num_leaves": [15, 31, 63],
        "model__max_depth": [-1, 4, 6],
        "model__min_child_samples": [15, 30, 60],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__reg_alpha": [0.0, 0.1],
        "model__reg_lambda": [1.0, 2.0],
    }


def _param_distributions(model_name):
    if model_name == "xgb":
        return _xgb_param_distributions()
    if model_name == "catboost":
        return _cat_param_distributions()
    if model_name == "lightgbm":
        return _lgbm_param_distributions()
    return {}


def _build_cv_splits(X, y, n_splits):
    y_ser = pd.Series(y).reset_index(drop=True)
    n_rows = len(y_ser)
    if n_rows < 4:
        return None

    n_splits = max(2, min(int(n_splits), n_rows))
    max_bins = min(10, int(y_ser.nunique()))

    for bins in range(max_bins, 1, -1):
        try:
            y_bins = pd.qcut(y_ser, q=bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        y_bins = pd.Series(y_bins).astype(int)
        if y_bins.nunique() < 2:
            continue
        min_count = int(y_bins.value_counts().min())
        if min_count < n_splits:
            continue
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return list(splitter.split(X, y_bins))

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(X))


def _tune_single_model(model_name, X_tune, y_tune):
    cv_splits = _build_cv_splits(X_tune, y_tune, CV_SPLITS)
    if not cv_splits:
        return None

    estimator = _build_estimator(model_name)
    param_dist = _param_distributions(model_name)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=max(1, N_ITER_SINGLE),
        scoring=_selection_scorer(SELECTION_METRIC),
        cv=cv_splits,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
        error_score="raise",
    )
    search.fit(X_tune, y_tune)
    return {
        "search": search,
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_score": float(-search.best_score_),
    }


def _blend_weight_grid():
    step = BLEND_WEIGHT_STEP if BLEND_WEIGHT_STEP > 0 else 0.10
    return np.round(np.arange(0.0, 1.000001, step), 4)


def _fit_with_params(model_name, params, X_train, y_train, X_pred):
    est = _build_estimator(model_name)
    est.set_params(**params)
    est.fit(X_train, y_train)
    pred = est.predict(X_pred)
    return pred, est


def _selection_folds(df, eligible_train):
    folds = [
        {
            "name": "2019_2020",
            "train": (df["season"] <= 2018) & eligible_train,
            "val": df["season"].between(2019, 2020) & eligible_train,
        },
        {
            "name": "2021_2022",
            "train": (df["season"] <= 2020) & eligible_train,
            "val": df["season"].between(2021, 2022) & eligible_train,
        },
    ]
    return [f for f in folds if int(f["train"].sum()) > 0 and int(f["val"].sum()) > 0]


def run_experiment(model_name, position_code=None, csv_path=DEFAULT_CSV, model_choice="auto"):
    csv_path = os.environ.get("NBAML_CSV_PATH", csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find dataset: {csv_path}")

    signature = _cache_signature(model_name, position_code, csv_path)
    cached = _load_cache(model_name, signature)
    if cached is not None:
        _print_cached(cached, model_name)
        return cached

    started_at = time.time()
    df = pd.read_csv(csv_path)
    df = _ensure_engineered_columns(df)

    if position_code is not None:
        df = df.loc[_position_mask(df["position"], position_code)].copy()
        print(f"Running {model_name} on position={position_code} | rows={len(df):,}")
    else:
        print(f"Running {model_name} on all positions | rows={len(df):,}")

    print("Target:", "next season salary (t+1)")
    print("Selection metric:", SELECTION_METRIC)
    print("Training games threshold:", f">= {TRAIN_GAMES_THRESHOLD}")
    print("RandomizedSearchCV n_iter:", N_ITER_SINGLE)
    print("CV splits (StratifiedKFold or fallback):", CV_SPLITS)

    if df.empty:
        print("No rows after filtering. Exiting.")
        return None

    y = df["next_log_salary"]

    # Feature-season split: season=t features predict salary at t+1.
    eligible_train = (df["games"] >= TRAIN_GAMES_THRESHOLD) & df["next_log_salary"].notna()
    train_mask_recent = (df["season"] <= 2020) & eligible_train
    blend_val_mask = df["season"].between(2021, 2022) & eligible_train
    tune_mask = (df["season"] <= 2022) & eligible_train
    holdout_mask = (df["season"] >= 2023) & df["next_log_salary"].notna()

    print(
        "Rows - train:",
        int(train_mask_recent.sum()),
        "blend_val:",
        int(blend_val_mask.sum()),
        "tune:",
        int(tune_mask.sum()),
        "holdout:",
        int(holdout_mask.sum()),
    )

    if int(train_mask_recent.sum()) == 0 or int(tune_mask.sum()) == 0 or int(holdout_mask.sum()) == 0:
        print("Not enough rows in train/tune/holdout split. Exiting.")
        return None

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
    else:
        model_options = [model_choice]

    if not model_options:
        print("No models available. Install xgboost/catboost/lightgbm as needed.")
        return None

    pack_map = _feature_pack_map(position_code)
    candidates = []
    tune_artifacts = {}
    tuning_started_at = time.time()

    for pack_name, pack_cols in pack_map.items():
        features = [c for c in pack_cols if c in df.columns]
        features = [c for c in features if df.loc[tune_mask, c].notna().any()]
        if not features:
            continue

        X_pack = df[features].copy()
        for c in features:
            X_pack[c] = pd.to_numeric(X_pack[c], errors="coerce")

        X_tune = X_pack.loc[tune_mask]
        y_tune = y.loc[tune_mask]
        X_train_recent = X_pack.loc[train_mask_recent]
        y_train_recent = y.loc[train_mask_recent]
        X_blend = X_pack.loc[blend_val_mask]
        y_blend = y.loc[blend_val_mask]

        tune_artifacts.setdefault(pack_name, {})

        for mdl in [m for m in model_options if m != "blend"]:
            try:
                t = _tune_single_model(mdl, X_tune, y_tune)
                if t is None:
                    continue
                tune_artifacts[pack_name][mdl] = t

                pred_val, _ = _fit_with_params(mdl, t["best_params"], X_train_recent, y_train_recent, X_blend)
                val_score = _selection_score(y_blend, pred_val, metric=SELECTION_METRIC)

                candidates.append(
                    {
                        "pack": pack_name,
                        "model": mdl,
                        "features": features,
                        "score": float(val_score),
                        "best_cv_score": float(t["best_cv_score"]),
                        "best_params": t["best_params"],
                    }
                )
            except Exception as ex:
                print(f"Tune fail | pack={pack_name}, model={mdl}: {ex}")

        if "blend" in model_options and "xgb" in tune_artifacts[pack_name] and "catboost" in tune_artifacts[pack_name]:
            try:
                xgb_params = tune_artifacts[pack_name]["xgb"]["best_params"]
                cat_params = tune_artifacts[pack_name]["catboost"]["best_params"]

                pred_xgb_val, _ = _fit_with_params(
                    "xgb", xgb_params, X_train_recent, y_train_recent, X_blend
                )
                pred_cat_val, _ = _fit_with_params(
                    "catboost", cat_params, X_train_recent, y_train_recent, X_blend
                )

                best_w = 0.5
                best_score = float("inf")
                for w in _blend_weight_grid():
                    p = w * pred_xgb_val + (1 - w) * pred_cat_val
                    s = _selection_score(y_blend, p, metric=SELECTION_METRIC)
                    if s < best_score:
                        best_score = s
                        best_w = float(w)

                candidates.append(
                    {
                        "pack": pack_name,
                        "model": "blend",
                        "features": features,
                        "score": float(best_score),
                        "best_cv_score": None,
                        "best_params": {
                            "xgb": xgb_params,
                            "catboost": cat_params,
                        },
                        "blend_weight": best_w,
                    }
                )
            except Exception as ex:
                print(f"Blend tuning fail | pack={pack_name}: {ex}")

    if not candidates:
        print("No successful tuning candidates found. Exiting.")
        return None

    candidates = sorted(candidates, key=lambda c: (c["score"], 1 if c["model"] == "blend" else 0))
    best = candidates[0]
    selected_pack = best["pack"]
    selected_model = best["model"]
    model_features = best["features"]
    tuning_runtime = time.time() - tuning_started_at

    print(f"Model features used: {len(model_features)}")
    print(f"Selected feature pack by validation: {selected_pack}")
    print(f"Selected model by validation: {selected_model}")
    print(f"Validation {SELECTION_METRIC}: {best['score']:.6f}")
    print("Top validation candidates:")
    for row in candidates[:8]:
        extra = ""
        if row["model"] == "blend":
            extra = f" w={row.get('blend_weight', 0.5):.2f}"
        print(
            f"  pack={row['pack']:<24} model={row['model']:<8} "
            f"{SELECTION_METRIC}={row['score']:.6f}{extra}"
        )

    X = df[model_features].copy()
    for c in model_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y_holdout = y.loc[holdout_mask]

    preds = {}
    metrics = {}
    xgb_model_final = None

    # Use best params for each family within selected pack.
    best_by_family = {}
    for fam in ["xgb", "catboost", "lightgbm", "blend"]:
        fam_rows = [c for c in candidates if c["pack"] == selected_pack and c["model"] == fam]
        if fam_rows:
            best_by_family[fam] = fam_rows[0]

    if xgb_allowed and "xgb" in best_by_family:
        row = best_by_family["xgb"]
        pred, est = _fit_with_params(
            "xgb",
            row["best_params"],
            X.loc[tune_mask],
            y.loc[tune_mask],
            X.loc[holdout_mask],
        )
        preds["xgb"] = pred
        metrics["xgb"] = eval_preds("XGBoost (final)", y_holdout, pred)
        xgb_model_final = est.named_steps["model"]

    if cat_allowed and "catboost" in best_by_family:
        row = best_by_family["catboost"]
        pred, _ = _fit_with_params(
            "catboost",
            row["best_params"],
            X.loc[tune_mask],
            y.loc[tune_mask],
            X.loc[holdout_mask],
        )
        preds["catboost"] = pred
        metrics["catboost"] = eval_preds("CatBoost (final)", y_holdout, pred)

    if lgbm_allowed and "lightgbm" in best_by_family:
        row = best_by_family["lightgbm"]
        pred, _ = _fit_with_params(
            "lightgbm",
            row["best_params"],
            X.loc[tune_mask],
            y.loc[tune_mask],
            X.loc[holdout_mask],
        )
        preds["lightgbm"] = pred
        metrics["lightgbm"] = eval_preds("LightGBM (final)", y_holdout, pred)

    final_blend_weight = None
    if "xgb" in preds and "catboost" in preds:
        xgb_params_blend = best_by_family.get("xgb", {}).get("best_params")
        cat_params_blend = best_by_family.get("catboost", {}).get("best_params")

        pred_xgb_val, _ = _fit_with_params(
            "xgb",
            xgb_params_blend,
            X.loc[train_mask_recent],
            y.loc[train_mask_recent],
            X.loc[blend_val_mask],
        )
        pred_cat_val, _ = _fit_with_params(
            "catboost",
            cat_params_blend,
            X.loc[train_mask_recent],
            y.loc[train_mask_recent],
            X.loc[blend_val_mask],
        )

        final_blend_weight = 0.5
        best_blend_score = float("inf")
        for w in _blend_weight_grid():
            p = w * pred_xgb_val + (1 - w) * pred_cat_val
            s = _selection_score(y.loc[blend_val_mask], p, SELECTION_METRIC)
            if s < best_blend_score:
                best_blend_score = s
                final_blend_weight = float(w)

        print(f"Best blend weight on 2021-2022 (XGB weight): {final_blend_weight:.2f}, {SELECTION_METRIC}: {best_blend_score:.6f}")

        pred_blend_holdout = final_blend_weight * preds["xgb"] + (1 - final_blend_weight) * preds["catboost"]
        preds["blend"] = pred_blend_holdout
        metrics["blend"] = eval_preds(
            f"Blend (final, XGB={final_blend_weight:.2f}, CAT={1-final_blend_weight:.2f})",
            y_holdout,
            pred_blend_holdout,
        )

    selected_model_name = selected_model if selected_model in preds else None
    if selected_model_name is None:
        for m in ["blend", "xgb", "catboost", "lightgbm"]:
            if m in preds:
                selected_model_name = m
                break
    if selected_model_name is None:
        print("No successful final model predictions were produced.")
        return None

    selected_metrics = eval_preds(f"Auto-Selected ({selected_model_name})", y_holdout, preds[selected_model_name])

    baseline_pred_log = np.log(np.clip(df.loc[holdout_mask, "salary"], 1.0, None))
    baseline_metrics = eval_preds(
        "Naive Baseline (next salary = prior-season salary)",
        y_holdout,
        baseline_pred_log,
    )

    if xgb_model_final is not None and hasattr(xgb_model_final, "feature_importances_"):
        fi = pd.Series(xgb_model_final.feature_importances_, index=model_features).sort_values(ascending=False)
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
        "prediction_target": "next_season_salary",
        "target_column": "next_log_salary",
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
            "tuning": float(tuning_runtime),
            "total": float(total_runtime),
        },
        "tuning_config": {
            "randomized_search_n_iter": int(N_ITER_SINGLE),
            "cv_splits": int(CV_SPLITS),
            "blend_weight_step": float(BLEND_WEIGHT_STEP),
            "include_no_salary_pack": bool(INCLUDE_NO_SALARY_PACK),
            "cv_method": "StratifiedKFold on binned target (fallback: KFold)",
        },
        "library_availability": {
            "xgboost": bool(xgb_allowed),
            "catboost": bool(cat_allowed),
            "lightgbm": bool(lgbm_allowed),
        },
        "validation_candidates": [
            {
                "pack": c["pack"],
                "model": c["model"],
                "score": c["score"],
                "best_cv_score": c.get("best_cv_score"),
                "blend_weight": c.get("blend_weight"),
            }
            for c in candidates
        ],
        "selected_model": selected_model,
        "selected_model_best_params": best.get("best_params"),
        "final_blend_weight": final_blend_weight,
        "final_metrics": metrics,
        "auto_selected_model": selected_model_name,
        "auto_selected_metrics": selected_metrics,
        "naive_baseline_prior_salary_metrics": baseline_metrics,
        "xgb_top20_importance": fi_top20,
    }

    _save_results(model_name, payload)
    return payload
