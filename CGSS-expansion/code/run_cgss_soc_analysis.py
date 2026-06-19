from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import font_manager
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "CGSS-expansion"
DATA_ROOT = ROOT / "CGSS_clean"
BASIC_ROOT = ROOT / "basic-expansion" / "2.3-clustering" / "results" / "tables"

RESULTS = OUT_ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
REPORT_DIR = OUT_ROOT / "report"

RANDOM_STATE = 20260618
MIN_OCC_YEAR_N = 5


def ensure_dirs() -> None:
    for path in [OUT_ROOT / "code", TABLES, FIGURES, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    font_paths = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font in font_candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def read_csv(path: Path, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=dtype, encoding="utf-8-sig")


def normalize_isco(value: object) -> str | np.nan:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    text = text.replace(".0", "")
    digits = re.sub(r"\D", "", text)
    if not digits:
        return np.nan
    return digits.zfill(4)[-4:]


def build_soc_kmeans_crosswalk() -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = read_csv(DATA_ROOT / "isco_soc_mapping_long.csv", dtype={"isco08": str, "minor_soc": str, "soc2010": str})
    clusters = read_csv(BASIC_ROOT / "occupation_clusters.csv", dtype={"unit_id": str})
    clusters = clusters.loc[clusters["level"] == "SOC职业组别"].copy()
    clusters = clusters.rename(
        columns={
            "unit_id": "minor_soc",
            "cluster": "soc_cluster_id",
            "cluster_type": "soc_cluster_type",
            "base_exposure": "soc_cluster_base_exposure",
            "change_2018_2024": "soc_cluster_change_2018_2024",
            "slope_per_year": "soc_cluster_slope_per_year",
        }
    )
    keep = [
        "minor_soc",
        "unit_label",
        "soc_cluster_id",
        "soc_cluster_type",
        "selected_k",
        "soc_cluster_base_exposure",
        "soc_cluster_change_2018_2024",
        "soc_cluster_slope_per_year",
        "volatility",
        "mean_dynamic",
    ]
    clusters = clusters[keep].drop_duplicates("minor_soc")

    long = mapping.merge(clusters, on="minor_soc", how="left")
    long["isco08"] = long["isco08"].map(normalize_isco)
    long = long.dropna(subset=["isco08", "soc_cluster_type"])

    counts = (
        long.groupby(["isco08", "soc_cluster_id", "soc_cluster_type"], dropna=False)
        .agg(
            n_soc_pairs=("soc2010", "nunique"),
            n_minor_soc=("minor_soc", "nunique"),
            mean_soc_cluster_base=("soc_cluster_base_exposure", "mean"),
            mean_soc_cluster_slope=("soc_cluster_slope_per_year", "mean"),
        )
        .reset_index()
    )
    totals = counts.groupby("isco08", as_index=False)["n_soc_pairs"].sum().rename(columns={"n_soc_pairs": "n_soc_pairs_total"})
    counts = counts.merge(totals, on="isco08", how="left")
    counts["soc_cluster_assignment_share"] = counts["n_soc_pairs"] / counts["n_soc_pairs_total"]
    counts = counts.sort_values(
        ["isco08", "n_soc_pairs", "soc_cluster_assignment_share", "mean_soc_cluster_base"],
        ascending=[True, False, False, False],
    )
    exact = counts.drop_duplicates("isco08").copy()
    exact["soc_cluster_source_level"] = "exact_isco_soc"

    fallback_parts = [exact]
    for prefix_len in [3, 2, 1]:
        temp = exact.copy()
        temp["prefix"] = temp["isco08"].str[:prefix_len]
        by_prefix = (
            temp.groupby(["prefix", "soc_cluster_id", "soc_cluster_type"], dropna=False)
            .agg(
                n_source_isco_for_soc_cluster=("isco08", "nunique"),
                mean_soc_cluster_base=("mean_soc_cluster_base", "mean"),
                mean_soc_cluster_slope=("mean_soc_cluster_slope", "mean"),
            )
            .reset_index()
        )
        prefix_totals = (
            by_prefix.groupby("prefix", as_index=False)["n_source_isco_for_soc_cluster"]
            .sum()
            .rename(columns={"n_source_isco_for_soc_cluster": "n_source_isco_total_for_soc_cluster"})
        )
        by_prefix = by_prefix.merge(prefix_totals, on="prefix", how="left")
        by_prefix["soc_cluster_assignment_share"] = (
            by_prefix["n_source_isco_for_soc_cluster"] / by_prefix["n_source_isco_total_for_soc_cluster"]
        )
        by_prefix = by_prefix.sort_values(
            ["prefix", "n_source_isco_for_soc_cluster", "soc_cluster_assignment_share", "mean_soc_cluster_base"],
            ascending=[True, False, False, False],
        )
        top_prefix = by_prefix.drop_duplicates("prefix").copy()
        top_prefix["isco08"] = top_prefix["prefix"]
        top_prefix["soc_cluster_source_level"] = f"isco_prefix_{prefix_len}_soc"
        fallback_parts.append(top_prefix)

    fallback = pd.concat(fallback_parts, ignore_index=True, sort=False)
    return exact, fallback


def attach_soc_clusters(df: pd.DataFrame, fallback: pd.DataFrame, code_col: str = "current_isco08") -> pd.DataFrame:
    out = df.copy()
    out[code_col] = out[code_col].map(normalize_isco)
    out["_isco_exact"] = out[code_col]
    result = out.merge(
        fallback.loc[fallback["soc_cluster_source_level"] == "exact_isco_soc", [
            "isco08",
            "soc_cluster_id",
            "soc_cluster_type",
            "soc_cluster_assignment_share",
            "soc_cluster_source_level",
            "mean_soc_cluster_base",
            "mean_soc_cluster_slope",
        ]],
        left_on="_isco_exact",
        right_on="isco08",
        how="left",
    )

    for prefix_len in [3, 2, 1]:
        missing = result["soc_cluster_type"].isna() & result[code_col].notna()
        if not missing.any():
            continue
        prefix_map = fallback.loc[
            fallback["soc_cluster_source_level"] == f"isco_prefix_{prefix_len}_soc",
            [
                "isco08",
                "soc_cluster_id",
                "soc_cluster_type",
                "soc_cluster_assignment_share",
                "soc_cluster_source_level",
                "mean_soc_cluster_base",
                "mean_soc_cluster_slope",
            ],
        ].rename(columns={"isco08": "_prefix"})
        temp = result.loc[missing, [code_col]].copy()
        temp["_prefix"] = temp[code_col].str[:prefix_len]
        matched = temp.merge(prefix_map, on="_prefix", how="left")
        idx = result.index[missing]
        for col in [
            "soc_cluster_id",
            "soc_cluster_type",
            "soc_cluster_assignment_share",
            "soc_cluster_source_level",
            "mean_soc_cluster_base",
            "mean_soc_cluster_slope",
        ]:
            values = matched[col].to_numpy()
            fill_mask = pd.notna(values)
            target_idx = idx[fill_mask]
            result.loc[target_idx, col] = values[fill_mask]

    drop_cols = [c for c in ["_isco_exact", "isco08"] if c in result.columns]
    return result.drop(columns=drop_cols)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_share(mask: pd.Series, weights: pd.Series) -> float:
    valid = weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(weights[valid & mask.fillna(False)].sum() / weights[valid].sum())


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    worker = read_csv(DATA_ROOT / "cgss_worker_ai_exposure.csv", dtype={"current_isco08": str, "source_prefix": str})
    panel = read_csv(DATA_ROOT / "occupation_year_panel.csv", dtype={"current_isco08": str, "source_prefix": str})
    exact_soc, fallback_soc = build_soc_kmeans_crosswalk()

    worker = attach_soc_clusters(worker, fallback_soc, "current_isco08")
    panel = attach_soc_clusters(panel, fallback_soc, "current_isco08")

    for df in [worker, panel]:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["post2023"] = (df["year"] == 2023).astype(int)
        df["event_2021"] = (df["year"] == 2021).astype(int)
        df["event_2023"] = (df["year"] == 2023).astype(int)
        for col in [
            "ai_exposure",
            "ai_exposure_base",
            "ai_exposure_change_2018_2024",
            "ai_exposure_slope_2018_2024",
            "ai_exposure_volatility",
            "weight",
            "age",
            "age_sq",
            "female",
            "log_labor_income",
            "log_total_income",
            "weighted_count",
            "weighted_employment_share",
            "mean_age",
            "college_plus_share",
            "female_share",
            "sample_count",
            "year_total_weighted_current_isco",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    worker["college_plus"] = (worker["education_group"] == "大专及以上").astype(float)
    worker["urban"] = (pd.to_numeric(worker["isurban_code"], errors="coerce") == 1).astype(float)
    worker["age_group"] = pd.cut(
        worker["age"],
        bins=[15, 34, 49, 65, 120],
        labels=["青年", "中年", "较高年龄", "老年"],
        right=True,
    )

    diagnostics = {
        "worker_rows": int(len(worker)),
        "panel_rows": int(len(panel)),
        "worker_with_soc_cluster": int(worker["soc_cluster_type"].notna().sum()),
        "panel_with_soc_cluster": int(panel["soc_cluster_type"].notna().sum()),
        "exact_soc_isco_count": int(exact_soc["isco08"].nunique()),
    }
    return worker, panel, exact_soc, diagnostics


def save_mapping_diagnostics(worker: pd.DataFrame, panel: pd.DataFrame, exact_soc: pd.DataFrame, diagnostics: dict[str, object]) -> None:
    worker_diag = (
        worker.groupby(["year", "soc_cluster_source_level"], dropna=False)
        .agg(sample_count=("id", "count"), weighted_count=("weight", "sum"))
        .reset_index()
    )
    worker_diag.to_csv(TABLES / "soc_kmeans_mapping_diagnostics_worker.csv", index=False, encoding="utf-8-sig")

    panel_diag = (
        panel.groupby(["year", "soc_cluster_source_level"], dropna=False)
        .agg(panel_cells=("current_isco08", "count"), total_weighted_count=("weighted_count", "sum"))
        .reset_index()
    )
    panel_diag.to_csv(TABLES / "soc_kmeans_mapping_diagnostics_panel.csv", index=False, encoding="utf-8-sig")

    exact_soc.to_csv(TABLES / "isco_to_soc_kmeans_crosswalk.csv", index=False, encoding="utf-8-sig")

    diagnostics["worker_soc_cluster_match_rate"] = float(worker["soc_cluster_type"].notna().mean())
    diagnostics["panel_soc_cluster_match_rate"] = float(panel["soc_cluster_type"].notna().mean())
    (TABLES / "run_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")


def descriptive_tables(worker: pd.DataFrame, panel: pd.DataFrame) -> None:
    valid_worker = worker.loc[(worker["has_ai_exposure"] == 1) & worker["soc_cluster_type"].notna()].copy()
    exposure_by_year = (
        valid_worker.groupby("year")
        .apply(
            lambda g: pd.Series(
                {
                    "样本量": len(g),
                    "加权平均暴露度": weighted_mean(g["ai_exposure"], g["weight"]),
                    "加权平均基础暴露度": weighted_mean(g["ai_exposure_base"], g["weight"]),
                    "高暴露四分位加权占比": weighted_share(g["high_ai_exposure_q4"] == 1, g["weight"]),
                    "劳动收入有效样本": int((g["income_analysis_sample"] == 1).sum()),
                    "加权平均log劳动收入": weighted_mean(g["log_labor_income"], g["weight"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    exposure_by_year.to_csv(TABLES / "descriptive_by_year.csv", index=False, encoding="utf-8-sig")

    cluster_desc = (
        valid_worker.groupby(["year", "soc_cluster_type"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "样本量": len(g),
                    "加权人数": g["weight"].sum(),
                    "加权平均基础暴露度": weighted_mean(g["ai_exposure_base"], g["weight"]),
                    "加权平均log劳动收入": weighted_mean(g["log_labor_income"], g["weight"]),
                    "大专及以上加权比例": weighted_share(g["college_plus"] == 1, g["weight"]),
                    "女性加权比例": weighted_share(g["female"] == 1, g["weight"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    cluster_desc.to_csv(TABLES / "descriptive_by_soc_cluster.csv", index=False, encoding="utf-8-sig")

    panel_desc = (
        panel.loc[panel["soc_cluster_type"].notna()]
        .groupby(["year", "soc_cluster_type"], dropna=False)
        .agg(
            职业单元数=("current_isco08", "nunique"),
            加权就业份额=("weighted_employment_share", "sum"),
            加权就业规模=("weighted_count", "sum"),
            平均基础暴露度=("ai_exposure_base", "mean"),
            平均log劳动收入=("weighted_mean_log_labor_income", "mean"),
        )
        .reset_index()
    )
    panel_desc.to_csv(TABLES / "employment_share_by_soc_cluster.csv", index=False, encoding="utf-8-sig")


def fit_glm_clustered(
    formula: str,
    data: pd.DataFrame,
    family,
    cluster_col: str | None = None,
    offset=None,
    var_weights=None,
):
    model = smf.glm(formula=formula, data=data, family=family, offset=offset, var_weights=var_weights)
    result = model.fit(maxiter=200, disp=False)
    if cluster_col and data[cluster_col].nunique() > 1:
        try:
            result = model.fit(maxiter=200, disp=False, cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
        except Exception:
            pass
    return result


def extract_terms(result, terms: list[str], model_name: str) -> pd.DataFrame:
    rows = []
    for term in terms:
        if term not in result.params.index:
            rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "coef": np.nan,
                    "std_err": np.nan,
                    "z": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
            continue
        coef = float(result.params[term])
        se = float(result.bse[term])
        rows.append(
            {
                "model": model_name,
                "term": term,
                "coef": coef,
                "std_err": se,
                "z": coef / se if se > 0 else np.nan,
                "p_value": float(result.pvalues[term]),
                "ci_low": coef - 1.96 * se,
                "ci_high": coef + 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


def employment_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = panel.loc[
        panel["weighted_employment_share"].notna()
        & panel["weighted_count"].notna()
        & panel["ai_exposure_base"].notna()
        & panel["soc_cluster_type"].notna()
        & (panel["sample_count"] >= MIN_OCC_YEAR_N)
    ].copy()
    data["year_str"] = data["year"].astype(str)
    data["current_isco08"] = data["current_isco08"].map(normalize_isco)
    data["log_total_weighted_current_isco"] = np.log(data["year_total_weighted_current_isco"].clip(lower=1e-9))
    data["base_x_2021"] = data["ai_exposure_base"] * data["event_2021"]
    data["base_x_2023"] = data["ai_exposure_base"] * data["event_2023"]
    data["slope_x_2021"] = data["ai_exposure_slope_2018_2024"] * data["event_2021"]
    data["slope_x_2023"] = data["ai_exposure_slope_2018_2024"] * data["event_2023"]
    data["share_clip"] = data["weighted_employment_share"].clip(1e-6, 1 - 1e-6)

    formula_base = (
        "share_clip ~ base_x_2021 + base_x_2023 + slope_x_2021 + slope_x_2023 "
        "+ mean_age + college_plus_share + female_share + C(current_isco08) + C(year_str)"
    )
    frac = fit_glm_clustered(
        formula_base,
        data,
        sm.families.Binomial(),
        cluster_col="current_isco08",
    )

    ppml = fit_glm_clustered(
        "weighted_count ~ base_x_2021 + base_x_2023 + slope_x_2021 + slope_x_2023 "
        "+ mean_age + college_plus_share + female_share + C(current_isco08) + C(year_str)",
        data,
        sm.families.Poisson(),
        cluster_col="current_isco08",
        offset=data["log_total_weighted_current_isco"],
    )

    terms = ["base_x_2021", "base_x_2023", "slope_x_2021", "slope_x_2023"]
    coef = pd.concat(
        [
            extract_terms(frac, terms, "分数Logit：连续暴露"),
            extract_terms(ppml, terms, "PPML：加权就业规模"),
        ],
        ignore_index=True,
    )
    coef.to_csv(TABLES / "employment_event_models.csv", index=False, encoding="utf-8-sig")

    # K-means descriptive panel: compare all matched cells, not only the model sample.
    # The normalized share sums to one within the SOC K-means-covered occupation cells.
    cluster_panel = (
        panel.loc[panel["soc_cluster_type"].notna() & panel["weighted_employment_share"].notna()]
        .groupby(["year", "soc_cluster_type"], dropna=False)
        .agg(
            weighted_employment_share=("weighted_employment_share", "sum"),
            weighted_count=("weighted_count", "sum"),
            occupation_units=("current_isco08", "nunique"),
            mean_base_exposure=("ai_exposure_base", "mean"),
        )
        .reset_index()
    )
    totals = (
        cluster_panel.groupby("year", as_index=False)["weighted_employment_share"]
        .sum()
        .rename(columns={"weighted_employment_share": "soc_kmeans_covered_share_total"})
    )
    cluster_panel = cluster_panel.merge(totals, on="year", how="left")
    cluster_panel["normalized_employment_share"] = (
        cluster_panel["weighted_employment_share"] / cluster_panel["soc_kmeans_covered_share_total"]
    )
    cluster_panel.to_csv(TABLES / "employment_kmeans_cluster_panel.csv", index=False, encoding="utf-8-sig")
    return coef, cluster_panel


def build_ml_preprocessor(data: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    categorical = [col for col in features if data[col].dtype == "object" or str(data[col].dtype) == "category"]
    numeric = [col for col in features if col not in categorical]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical),
        ],
        remainder="drop",
    )


def dml_partialling_out(
    data: pd.DataFrame,
    y_col: str,
    t_col: str,
    features: list[str],
    learner_name: str = "gbrt",
    n_splits: int = 3,
    weight_col: str = "weight",
    cluster_col: str = "current_isco08",
) -> dict[str, float]:
    columns = list(dict.fromkeys([y_col, t_col, weight_col, cluster_col] + features))
    sample = data[columns].dropna(subset=[y_col, t_col, weight_col, cluster_col]).copy()
    sample = sample.loc[pd.to_numeric(sample[weight_col], errors="coerce") > 0].copy()
    sample = sample.reset_index(drop=True)
    y = sample[y_col].to_numpy(dtype=float)
    t = sample[t_col].to_numpy(dtype=float)
    weights = sample[weight_col].to_numpy(dtype=float)
    weights = weights / np.nanmean(weights)
    clusters = sample[cluster_col].astype(str).to_numpy()
    x = sample[features]

    if learner_name == "lasso":
        y_model = Pipeline([("prep", build_ml_preprocessor(sample, features)), ("model", LassoCV(cv=3, random_state=RANDOM_STATE, max_iter=20000))])
        t_model = Pipeline([("prep", build_ml_preprocessor(sample, features)), ("model", LassoCV(cv=3, random_state=RANDOM_STATE, max_iter=20000))])
    elif learner_name == "rf":
        base = RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        y_model = Pipeline([("prep", build_ml_preprocessor(sample, features)), ("model", base)])
        t_model = Pipeline(
            [
                ("prep", build_ml_preprocessor(sample, features)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        min_samples_leaf=20,
                        random_state=RANDOM_STATE + 1,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        y_model = Pipeline(
            [
                ("prep", build_ml_preprocessor(sample, features)),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE, max_depth=2, n_estimators=80, learning_rate=0.05)),
            ]
        )
        t_model = Pipeline(
            [
                ("prep", build_ml_preprocessor(sample, features)),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE + 1, max_depth=2, n_estimators=80, learning_rate=0.05)),
            ]
        )

    y_hat = np.zeros_like(y, dtype=float)
    t_hat = np.zeros_like(t, dtype=float)
    n_splits = min(n_splits, max(2, len(sample) // 300))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in kfold.split(sample):
        try:
            y_model.fit(x.iloc[train_idx], y[train_idx], model__sample_weight=weights[train_idx])
        except TypeError:
            y_model.fit(x.iloc[train_idx], y[train_idx])
        try:
            t_model.fit(x.iloc[train_idx], t[train_idx], model__sample_weight=weights[train_idx])
        except TypeError:
            t_model.fit(x.iloc[train_idx], t[train_idx])
        y_hat[test_idx] = y_model.predict(x.iloc[test_idx])
        t_hat[test_idx] = t_model.predict(x.iloc[test_idx])

    y_res = y - y_hat
    t_res = t - t_hat
    denom = float(np.sum(weights * t_res**2))
    theta = float(np.sum(weights * t_res * y_res) / denom) if denom > 0 else np.nan
    residual = y_res - theta * t_res
    n = len(sample)
    psi = weights * t_res * residual
    cluster_score = pd.DataFrame({"cluster": clusters, "psi": psi}).groupby("cluster")["psi"].sum()
    g = int(cluster_score.shape[0])
    if denom > 0 and g > 1:
        finite_correction = g / (g - 1)
        se = float(np.sqrt(finite_correction * np.sum(cluster_score.to_numpy() ** 2) / (denom**2)))
    elif denom > 0:
        se = float(np.sqrt(np.sum(psi**2) / (denom**2)))
    else:
        se = np.nan
    z = theta / se if se and se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if pd.notna(z) else np.nan
    return {
        "outcome": y_col,
        "treatment": t_col,
        "learner": learner_name,
        "n": int(n),
        "theta": theta,
        "std_err": se,
        "z": z,
        "p_value": p,
        "ci_low": theta - 1.96 * se if pd.notna(se) else np.nan,
        "ci_high": theta + 1.96 * se if pd.notna(se) else np.nan,
        "y_residual_sd": float(np.std(y_res)),
        "t_residual_sd": float(np.std(t_res)),
        "cluster_count": g,
        "weighted": True,
        "inference": f"clustered_by_{cluster_col}",
    }


def prepare_income_sample(worker: pd.DataFrame) -> pd.DataFrame:
    sample = worker.loc[
        (worker["income_analysis_sample"] == 1)
        & (worker["has_ai_exposure"] == 1)
        & worker["soc_cluster_type"].notna()
        & worker["log_labor_income"].notna()
        & worker["weight"].notna()
    ].copy()
    for col in [
        "province_code",
        "current_isco08",
        "current_isco_major",
        "year",
        "education_group",
        "exposure_source_level",
        "soc_cluster_type",
        "age_group",
    ]:
        sample[col] = sample[col].astype("string").fillna("缺失").astype(str)
    sample["t_base_post"] = sample["ai_exposure_base"] * sample["post2023"]
    sample["t_high_post"] = sample["high_ai_exposure_q4"] * sample["post2023"]
    sample["t_slope_post"] = sample["ai_exposure_slope_2018_2024"] * sample["post2023"]
    sample["t_change_post"] = sample["ai_exposure_change_2018_2024"] * sample["post2023"]
    cluster_dummies = pd.get_dummies(sample["soc_cluster_type"], prefix="cluster", dtype=float)
    for col in cluster_dummies.columns:
        if "低暴露稳定型" in col:
            continue
        sample[f"t_{col}_post"] = cluster_dummies[col] * sample["post2023"]
    return sample


def income_dml_models(worker: pd.DataFrame) -> pd.DataFrame:
    sample = prepare_income_sample(worker)
    features = [
        "age",
        "age_sq",
        "female",
        "college_plus",
        "urban",
        "health_code",
        "marriage_code",
        "province_code",
        "current_isco08",
        "current_isco_major",
        "year",
        "education_group",
        "exposure_source_level",
        "ai_exposure_base",
        "high_ai_exposure_q4",
        "ai_exposure_change_2018_2024",
        "ai_exposure_slope_2018_2024",
        "ai_exposure_volatility",
        "soc_cluster_type",
        "soc_cluster_assignment_share",
    ]
    treatments = ["t_base_post", "t_high_post", "t_slope_post", "t_change_post"]
    treatments += [col for col in sample.columns if col.startswith("t_cluster_") and col.endswith("_post")]

    rows = []
    labor_sample = sample.dropna(subset=["log_labor_income"]).copy()
    for treatment in treatments:
        if labor_sample[treatment].std(skipna=True) <= 1e-12:
            continue
        rows.append(dml_partialling_out(labor_sample, "log_labor_income", treatment, features, "gbrt"))
    for treatment in ["t_base_post", "t_high_post"]:
        if labor_sample[treatment].std(skipna=True) > 1e-12:
            rows.append(dml_partialling_out(labor_sample, "log_labor_income", treatment, features, "lasso"))

    total_sample = sample.dropna(subset=["log_total_income"]).copy()
    for treatment in ["t_base_post", "t_high_post"]:
        if total_sample[treatment].std(skipna=True) > 1e-12:
            rows.append(dml_partialling_out(total_sample, "log_total_income", treatment, features, "gbrt"))
    results = pd.DataFrame(rows)
    results.to_csv(TABLES / "income_dml_results.csv", index=False, encoding="utf-8-sig")
    return results


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum_weights = np.cumsum(weights)
    cutoff = quantile * cum_weights[-1]
    return float(np.interp(cutoff, cum_weights, values))


def rif_value(y: np.ndarray, tau: float, weights: np.ndarray) -> tuple[np.ndarray, float, float]:
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.nanmean(weights)
    q = weighted_quantile(y, tau, weights)
    try:
        kde = stats.gaussian_kde(y, weights=weights)
    except TypeError:
        kde = stats.gaussian_kde(y)
    density = float(kde.evaluate([q])[0])
    density = max(density, 1e-8)
    rif = q + (tau - (y <= q).astype(float)) / density
    return rif, q, density


def rif_models(worker: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = prepare_income_sample(worker)
    sample = sample.dropna(subset=["log_labor_income", "t_base_post"]).copy()
    formula_controls = (
        "t_base_post + t_high_post + t_slope_post + age + age_sq + female + college_plus + urban "
        "+ C(province_code) + C(year) + C(current_isco08) + C(education_group) + C(exposure_source_level)"
    )

    rows = []
    for tau in [0.25, 0.5, 0.75]:
        rif, q, density = rif_value(
            sample["log_labor_income"].to_numpy(dtype=float),
            tau,
            sample["weight"].to_numpy(dtype=float),
        )
        temp = sample.copy()
        temp["rif"] = rif
        try:
            model = smf.wls(f"rif ~ {formula_controls}", data=temp, weights=temp["weight"].clip(lower=1e-6))
            result = model.fit(cov_type="cluster", cov_kwds={"groups": temp["current_isco08"]})
        except Exception:
            result = smf.wls(f"rif ~ {formula_controls}", data=temp, weights=temp["weight"].clip(lower=1e-6)).fit()
        for term in ["t_base_post", "t_high_post", "t_slope_post"]:
            if term in result.params.index:
                coef = float(result.params[term])
                se = float(result.bse[term])
                rows.append(
                    {
                        "quantile": tau,
                        "income_quantile_value": q,
                        "density_at_quantile": density,
                        "term": term,
                        "coef": coef,
                        "std_err": se,
                        "t": coef / se if se > 0 else np.nan,
                        "p_value": float(result.pvalues[term]),
                        "ci_low": coef - 1.96 * se,
                        "ci_high": coef + 1.96 * se,
                        "n": int(len(temp)),
                    }
                )
    rif_results = pd.DataFrame(rows)
    rif_results.to_csv(TABLES / "income_rif_results.csv", index=False, encoding="utf-8-sig")

    hetero_rows = []
    group_specs = {
        "大专及以上": "college_plus",
        "女性": "female",
        "城市": "urban",
    }
    base_controls = [
        "age",
        "age_sq",
        "female",
        "college_plus",
        "urban",
        "C(province_code)",
        "C(year)",
        "C(current_isco08)",
        "C(education_group)",
        "C(exposure_source_level)",
    ]
    for group_label, group_col in group_specs.items():
        temp = sample.copy()
        temp["group"] = temp[group_col].fillna(0).astype(float)
        temp["t_group"] = temp["t_base_post"] * temp["group"]
        control_terms = [term for term in base_controls if term != group_col]
        hetero_formula = "log_labor_income ~ t_base_post + t_group + group + " + " + ".join(control_terms)
        try:
            model = smf.wls(hetero_formula, data=temp, weights=temp["weight"].clip(lower=1e-6))
            result = model.fit(cov_type="cluster", cov_kwds={"groups": temp["current_isco08"]})
        except Exception:
            result = smf.wls(hetero_formula, data=temp, weights=temp["weight"].clip(lower=1e-6)).fit()
        for term, label in [("t_base_post", "基准组"), ("t_group", f"{group_label}差异")]:
            if term in result.params.index:
                coef = float(result.params[term])
                se = float(result.bse[term])
                hetero_rows.append(
                    {
                        "group": group_label,
                        "term": term,
                        "label": label,
                        "coef": coef,
                        "std_err": se,
                        "p_value": float(result.pvalues[term]),
                        "ci_low": coef - 1.96 * se,
                        "ci_high": coef + 1.96 * se,
                        "n": int(len(temp)),
                    }
                )
    hetero = pd.DataFrame(hetero_rows)
    hetero.to_csv(TABLES / "heterogeneity_group_interactions.csv", index=False, encoding="utf-8-sig")
    return rif_results, hetero


def plot_outputs(worker: pd.DataFrame, panel: pd.DataFrame, employment_coef: pd.DataFrame, dml: pd.DataFrame, rif: pd.DataFrame, hetero: pd.DataFrame) -> None:
    valid_worker = worker.loc[(worker["has_ai_exposure"] == 1) & worker["ai_exposure"].notna()].copy()

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=valid_worker, x="ai_exposure", hue="year", common_norm=False, fill=False, linewidth=2)
    plt.title("不同年份的 AI-LLM 暴露度分布")
    plt.xlabel("AI-LLM 暴露度")
    plt.ylabel("密度")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_cgss_exposure_distribution.png")
    plt.close()

    trend = (
        panel.loc[panel["ai_exposure_quartile"].isin([1, 4])]
        .groupby(["year", "ai_exposure_quartile"], as_index=False)["weighted_employment_share"]
        .sum()
    )
    trend["暴露组"] = trend["ai_exposure_quartile"].map({1: "低暴露四分位", 4: "高暴露四分位"})
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=trend, x="year", y="weighted_employment_share", hue="暴露组", marker="o", linewidth=2)
    plt.title("高低暴露职业的加权就业份额变化")
    plt.xlabel("年份")
    plt.ylabel("加权就业份额")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_employment_share_trends.png")
    plt.close()

    event_plot = employment_coef.loc[employment_coef["term"].isin(["base_x_2021", "base_x_2023"])].copy()
    event_plot["年份"] = event_plot["term"].map({"base_x_2021": "2021", "base_x_2023": "2023"})
    plt.figure(figsize=(8, 5))
    for model_name, group in event_plot.groupby("model"):
        plt.errorbar(
            group["年份"],
            group["coef"],
            yerr=1.96 * group["std_err"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=model_name,
        )
    plt.axhline(0, color="black", linewidth=1)
    plt.title("连续处理事件研究系数")
    plt.xlabel("年份")
    plt.ylabel("暴露度交互项系数")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_event_study_coefficients.png")
    plt.close()

    cluster_trend = (
        panel.loc[panel["soc_cluster_type"].notna()]
        .groupby(["year", "soc_cluster_type"], as_index=False)["weighted_employment_share"]
        .sum()
    )
    cluster_trend["soc_kmeans_covered_share_total"] = cluster_trend.groupby("year")["weighted_employment_share"].transform("sum")
    cluster_trend["normalized_employment_share"] = (
        cluster_trend["weighted_employment_share"] / cluster_trend["soc_kmeans_covered_share_total"]
    )
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=cluster_trend, x="year", y="normalized_employment_share", hue="soc_cluster_type", marker="o", linewidth=2)
    plt.title("SOC K-means 类型的归一化就业份额变化")
    plt.xlabel("年份")
    plt.ylabel("归一化加权就业份额")
    plt.legend(title="SOC K-means 类型")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_soc_kmeans_employment_share.png")
    plt.close()

    dml_plot = dml.loc[
        (dml["outcome"] == "log_labor_income")
        & (dml["learner"] == "lasso")
        & (dml["treatment"].isin(["t_base_post", "t_high_post"]))
    ].copy()
    dml_plot["处理变量"] = dml_plot["treatment"].map(
        {
            "t_base_post": "基础暴露度×2023",
            "t_high_post": "高暴露四分位×2023",
            "t_slope_post": "暴露斜率×2023",
            "t_change_post": "暴露变化×2023",
        }
    ).fillna(dml_plot["treatment"].str.replace("t_cluster_", "K-means：", regex=False).str.replace("_post", "×2023", regex=False))
    plt.figure(figsize=(9, 5))
    plt.errorbar(dml_plot["theta"], dml_plot["处理变量"], xerr=1.96 * dml_plot["std_err"], fmt="o", capsize=4)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("DML：核心暴露变量与劳动收入")
    plt.xlabel("估计系数")
    plt.ylabel("处理变量")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_income_dml_effects.png")
    plt.close()

    rif_plot = rif.loc[rif["term"] == "t_base_post"].copy()
    plt.figure(figsize=(8, 5))
    plt.errorbar(rif_plot["quantile"].astype(str), rif_plot["coef"], yerr=1.96 * rif_plot["std_err"], marker="o", capsize=4)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("RIF：不同收入分位的暴露度关系")
    plt.xlabel("劳动收入分位")
    plt.ylabel("基础暴露度×2023 系数")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_rif_quantile_effects.png")
    plt.close()

    hetero_plot = hetero.loc[hetero["term"] == "t_group"].copy()
    if not hetero_plot.empty:
        plt.figure(figsize=(8, 5))
        plt.errorbar(hetero_plot["coef"], hetero_plot["group"], xerr=1.96 * hetero_plot["std_err"], fmt="o", capsize=4)
        plt.axvline(0, color="black", linewidth=1)
        plt.title("DML替代：群体异质性交互项")
        plt.xlabel("暴露度×2023×群体 系数")
        plt.ylabel("群体")
        plt.tight_layout()
        plt.savefig(FIGURES / "fig_heterogeneity_effects.png")
        plt.close()


def star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def write_report(worker: pd.DataFrame, panel: pd.DataFrame, employment_coef: pd.DataFrame, dml: pd.DataFrame, rif: pd.DataFrame, hetero: pd.DataFrame) -> None:
    diag = json.loads((TABLES / "run_diagnostics.json").read_text(encoding="utf-8"))
    run_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    income_year_map = (
        worker[["year", "income_reference_year"]]
        .dropna()
        .drop_duplicates()
        .sort_values("year")
        .astype(int)
    )
    income_year_text = "；".join(
        [f"{row.year} 年调查对应 {row.income_reference_year} 年收入" for row in income_year_map.itertuples()]
    )
    emp_main = employment_coef.loc[
        (employment_coef["model"] == "分数Logit：连续暴露") & (employment_coef["term"] == "base_x_2023")
    ].iloc[0]
    emp_pre = employment_coef.loc[
        (employment_coef["model"] == "分数Logit：连续暴露") & (employment_coef["term"] == "base_x_2021")
    ].iloc[0]
    ppml_main = employment_coef.loc[
        (employment_coef["model"] == "PPML：加权就业规模") & (employment_coef["term"] == "base_x_2023")
    ].iloc[0]
    dml_main = dml.loc[
        (dml["outcome"] == "log_labor_income")
        & (dml["treatment"] == "t_base_post")
        & (dml["learner"] == "lasso")
    ].iloc[0]
    high_main = dml.loc[
        (dml["outcome"] == "log_labor_income")
        & (dml["treatment"] == "t_high_post")
        & (dml["learner"] == "lasso")
    ].iloc[0]
    gbrt_base = dml.loc[
        (dml["outcome"] == "log_labor_income")
        & (dml["treatment"] == "t_base_post")
        & (dml["learner"] == "gbrt")
    ].iloc[0]
    gbrt_high = dml.loc[
        (dml["outcome"] == "log_labor_income")
        & (dml["treatment"] == "t_high_post")
        & (dml["learner"] == "gbrt")
    ].iloc[0]

    rif_main = rif.loc[rif["term"] == "t_base_post"].copy()
    rif_lines = []
    for _, row in rif_main.iterrows():
        rif_lines.append(
            f"- 第 {int(row['quantile'] * 100)} 分位：系数 {row['coef']:.4f}，"
            f"p={row['p_value']:.3f}{star(row['p_value'])}。"
        )

    cluster_panel = read_csv(TABLES / "employment_kmeans_cluster_panel.csv")
    cluster_wide = cluster_panel.pivot(index="soc_cluster_type", columns="year", values="normalized_employment_share")
    cluster_wide["2023较2018变化"] = cluster_wide.get(2023, np.nan) - cluster_wide.get(2018, np.nan)
    cluster_summary = cluster_wide.reset_index()
    cluster_summary.to_csv(TABLES / "employment_kmeans_cluster_change_summary.csv", index=False, encoding="utf-8-sig")

    cluster_lines = []
    for _, row in cluster_summary.iterrows():
        cluster_lines.append(
            f"- {row['soc_cluster_type']}：2018 年归一化份额 {row.get(2018, np.nan):.4f}，"
            f"2023 年归一化份额 {row.get(2023, np.nan):.4f}，变化 {row['2023较2018变化']:.4f}。"
        )

    report = f"""# CGSS 扩展分析报告：SOC 职业暴露度、就业份额与工资收入

生成时间：{run_date}

## 1. 分析目标与数据口径

本报告聚焦 SOC 职业口径下的 AI-LLM 暴露度。由于 CGSS 使用 ISCO-08 职业编码，本文先通过 `CGSS_clean/isco_soc_mapping_long.csv` 将 ISCO 映射到 SOC minor group，再接入 `basic-expansion` 中已经按最佳 K 重新估计的 SOC 职业组别 K-means 结果。SOC 职业组别的最佳聚类数为 `K=3`，三类分别为低暴露稳定型、中等暴露稳定型和中等暴露急剧下降型。

本文回答两个问题：

1. 高 AI-LLM 暴露职业是否在 2023 年出现 CGSS 加权就业份额相对下降？
2. 高 AI-LLM 暴露职业中的劳动者是否在 2023 年出现劳动收入相对下降？

需要强调的是，CGSS 是抽样调查，不能把样本中的职业人数直接解释为全国真实职业人数。因此，就业部分使用“加权就业份额”而不是“职业人数”作为主结果。

收入变量也需要谨慎解释：{income_year_text}。因此，2023 年调查中的劳动收入主要对应 2022 年，最多只能反映 GPT 发布前后极早期的收入关联信号，不能解释为完整的 GPT 后长期工资效应。

## 2. 样本与 SOC 映射质量

- 个体层面原始清洗数据：{diag['worker_rows']} 条。
- 职业-年份面板：{diag['panel_rows']} 个职业-年份单元。
- 个体样本中可接入 SOC K-means 类型的记录：{diag['worker_with_soc_cluster']} 条，匹配率 {diag['worker_soc_cluster_match_rate']:.3f}。
- 职业-年份面板中可接入 SOC K-means 类型的单元：{diag['panel_with_soc_cluster']} 个，匹配率 {diag['panel_soc_cluster_match_rate']:.3f}。

SOC K-means 的映射采用“精确 ISCO 优先、三位/二位/一位前缀回退”的规则。每条记录保留 `soc_cluster_source_level` 与 `soc_cluster_assignment_share`，用于后续稳健性检验。

## 3. 就业份额模型结果

就业份额部分使用职业-年份面板。主模型为分数 Logit QMLE，因变量为 `weighted_employment_share`；辅助模型为 PPML，因变量为 `weighted_count`，并加入年度总加权就业规模 offset。核心变量是基础暴露度与 2021、2023 年份虚拟变量的交互项。

主模型结果显示：

- GPT 前弱趋势项 `基础暴露度 × 2021` 的系数为 {emp_pre['coef']:.4f}，p={emp_pre['p_value']:.3f}{star(emp_pre['p_value'])}。
- GPT 后第一期项 `基础暴露度 × 2023` 的系数为 {emp_main['coef']:.4f}，p={emp_main['p_value']:.3f}{star(emp_main['p_value'])}。
- PPML 中 `基础暴露度 × 2023` 的系数为 {ppml_main['coef']:.4f}，p={ppml_main['p_value']:.3f}{star(ppml_main['p_value'])}。

本轮估计没有发现“高基础暴露度职业在 2023 年就业份额显著下降”的稳定证据。分数 Logit 与 PPML 中，`基础暴露度 × 2023` 均为正但远未显著，且置信区间很宽；因此更严谨的表述应是：现有 CGSS 三期截面尚不足以支持 AI-LLM 暴露度压低职业加权就业份额的结论。由于只有三期截面，这一结果也不能被解释为强因果估计。

SOC K-means 类型的就业份额变化如下。这里使用的是 SOC K-means 覆盖职业内部的归一化加权份额，避免因职业映射覆盖率或低样本职业过滤造成份额合计不等于 1：

{chr(10).join(cluster_lines)}

K-means 描述性结果提示，不同 SOC 动态暴露类型的就业结构变化并不符合简单的“高暴露必然收缩”叙事。中等暴露急剧下降型的份额下降较明显，但低暴露稳定型也下降；相反，中等暴露稳定型的份额上升。因此，K-means 类型更适合用来刻画职业结构重组，而不宜单独作为替代效应的因果证据。

对应图表：

- `results/figures/fig_employment_share_trends.png`
- `results/figures/fig_event_study_coefficients.png`
- `results/figures/fig_soc_kmeans_employment_share.png`

## 4. 工资收入 DML 结果

工资收入部分使用个体层面数据，主结果变量为 `log_labor_income`。优化后的 DML 使用 CGSS 调查权重，并按 4 位 ISCO 职业聚类计算标准误。控制变量包括年龄、年龄平方、性别、教育、城乡、健康、婚姻、省份、年份、4 位 ISCO 职业、ISCO 职业大类、暴露度映射质量、基础暴露度主效应、暴露度动态变量和 SOC K-means 类型。由于加入 4 位职业固定效应后处理变量的剩余变异较少，本文将 LassoCV-DML 作为高维固定效应主口径，梯度提升树 DML 作为灵活学习器敏感性结果。

主结果显示：

- Lasso-DML 中，`基础暴露度 × 2023调查` 的系数为 {dml_main['theta']:.4f}，职业聚类标准误 {dml_main['std_err']:.4f}，p={dml_main['p_value']:.3f}{star(dml_main['p_value'])}，样本量 {int(dml_main['n'])}，聚类数 {int(dml_main['cluster_count'])}。
- Lasso-DML 中，`高暴露四分位 × 2023调查` 的系数为 {high_main['theta']:.4f}，职业聚类标准误 {high_main['std_err']:.4f}，p={high_main['p_value']:.3f}{star(high_main['p_value'])}。
- GBRT-DML 敏感性结果中，`基础暴露度 × 2023调查` 的 p 值为 {gbrt_base['p_value']:.3f}，`高暴露四分位 × 2023调查` 的 p 值为 {gbrt_high['p_value']:.3f}，均未达到常规显著性水平。

本轮 DML 结果经过权重、4 位职业控制和职业聚类标准误修正后，解释应更加保守。Lasso-DML 仍显示负向且 10% 水平附近的边际信号，但 GBRT-DML 不支持稳定显著性。因此，更稳妥的表述是：高暴露职业在 2023 调查对应的 2022 年劳动收入中存在一定负向早期信号，但该信号对学习器设定敏感，尚不能构成稳定收入下降证据，更不能解释为强因果结论。

对应图表：

- `results/figures/fig_income_dml_effects.png`

## 5. 收入分布与异质性

RIF 回归用于检验 AI-LLM 暴露度是否只影响平均收入，还是改变不同收入分位的相对位置。优化后 RIF 使用加权分位数和加权密度估计，主处理变量仍为 `基础暴露度 × 2023调查`。

{chr(10).join(rif_lines)}

RIF 结果没有显示基础暴露度在第 25、50、75 分位上存在一致稳定关系。第 75 分位出现 10% 水平附近的负向边际信号，但第 25 分位和中位数并不显著，高暴露四分位项也不显著。因此，目前不能认为 AI-LLM 基础暴露度已经明显改变劳动收入分布。

由于当前环境没有安装 `econml`，异质性分析采用加权分组交互模型作为因果森林的可复现替代，重点比较教育、性别和城乡差异。探索性结果显示，女性和城市样本的交互项为正，意味着基础暴露度负向关系在这些群体中较弱；但该部分不是主识别模型，应作为异质性提示而非核心因果结论。相关结果见：

- `results/tables/heterogeneity_group_interactions.csv`
- `results/figures/fig_heterogeneity_effects.png`

## 6. 自我审查与结论边界

本轮分析做了三点审查：

1. **职业口径审查**：没有直接使用智联二级职业作为 CGSS 主口径，因为 CGSS 职业编码与智联分类没有稳定 crosswalk；主口径改用 SOC 职业组别 K=3。
2. **就业结果审查**：没有把 CGSS 样本数写成全国职业人数，而是使用加权就业份额和加权样本规模；K-means 类型份额在展示时使用覆盖职业内部归一化份额。
3. **工资收入审查**：DML 已加入调查权重、4 位职业控制和职业聚类标准误；2023 年调查收入按 2022 年收入解释。
4. **因果表述审查**：由于只有 2018、2021、2023 三期截面，且 2023 调查收入主要对应 2022 年，因此所有结果均表述为相关关系或弱准实验信号，不表述为强因果结论。

因此，本文最稳妥的结论框架是：

1. 就业结构方面，现有 CGSS 三期数据没有发现高基础暴露职业在 2023 年就业份额显著下降的稳定证据。
2. 工资收入方面，Lasso-DML 存在负向边际信号，但 GBRT-DML 不稳健，因此只能解释为 2022 收入口径下的早期弱信号，不能作为稳定收入下降证据。
3. 收入分布方面，RIF 模型暂未发现基础暴露度对不同收入分位的稳定影响。
4. SOC K-means 类型有助于描述职业结构重组，但不能单独解释为 AI 替代劳动的因果证据。

总体而言，AI-LLM 暴露度可以被用于解释 CGSS 中职业就业结构和个体收入的差异，但现有数据只能提供早期、有限和需要稳健性检验支持的证据。若后续补充更多 CGSS 年份、CPI 平减和外部职业就业统计，可以进一步增强识别力度。
"""
    (REPORT_DIR / "CGSS扩展分析报告.md").write_text(report, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    ensure_dirs()
    setup_plot_style()
    worker, panel, exact_soc, diagnostics = prepare_data()
    save_mapping_diagnostics(worker, panel, exact_soc, diagnostics)
    descriptive_tables(worker, panel)
    employment_coef, _ = employment_models(panel)
    dml = income_dml_models(worker)
    rif, hetero = rif_models(worker)
    plot_outputs(worker, panel, employment_coef, dml, rif, hetero)
    write_report(worker, panel, employment_coef, dml, rif, hetero)
    print("CGSS SOC analysis completed.")
    print(f"Tables: {TABLES}")
    print(f"Figures: {FIGURES}")
    print(f"Report: {REPORT_DIR / 'CGSS扩展分析报告.md'}")


if __name__ == "__main__":
    main()
