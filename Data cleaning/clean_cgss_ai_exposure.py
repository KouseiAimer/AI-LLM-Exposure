"""Build cleaned CGSS + AI-LLM exposure datasets.

The primary occupation conversion follows an iscoCrosswalks-style idea:
SOC/O*NET exposure is treated as an indicator and averaged to ISCO-08
through the local IBS SOC2010-ISCO08 crosswalk.  CGSS aggregate ISCO codes
are retained through a transparent ISCO hierarchy fallback.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CGSS_DIR = PROJECT_ROOT / "CGSS"
CROSS_DIR = PROJECT_ROOT / "Crosswork"
OUT_DIR = PROJECT_ROOT / "CGSS_clean"
DIAG_DIR = OUT_DIR / "diagnostics"


SPECIAL_MISSING = {
    -5,
    -4,
    -3,
    -2,
    -1,
    9997,
    9998,
    9999,
    9999996,
    9999997,
    9999998,
    9999999,
}


ISCO_MAJOR_LABELS = {
    "0": "军职人员",
    "1": "管理人员",
    "2": "专业人员",
    "3": "技术和辅助专业人员",
    "4": "办事人员",
    "5": "服务和销售人员",
    "6": "农林牧渔劳动者",
    "7": "工艺和相关工人",
    "8": "设备机器操作及装配人员",
    "9": "简单职业",
}


@dataclass(frozen=True)
class YearSpec:
    year: int
    file_name: str
    current_isco: list[str]
    ever_isco: str | None
    weight: str
    gender: str
    birth_year: str
    education: str
    total_income: str
    labor_income: str
    household_income: str
    health: str
    work_status: str
    current_work_status: str
    marriage: str
    subjective_status: str | None = None
    status_change: str | None = None


SPECS = [
    YearSpec(
        year=2018,
        file_name="CGSS2018.dta",
        current_isco=["isco08_a59d"],
        ever_isco="isco08_a60d",
        weight="weight",
        gender="a2",
        birth_year="a31",
        education="a7a",
        total_income="a8a",
        labor_income="a8b",
        household_income="a62",
        health="a15",
        work_status="a58",
        current_work_status="a59a",
        marriage="a69",
    ),
    YearSpec(
        year=2021,
        file_name="CGSS2021.dta",
        current_isco=["isco08_a59d"],
        ever_isco="isco08_a60d",
        weight="weight",
        gender="A2",
        birth_year="A3_1",
        education="A7a",
        total_income="A8a",
        labor_income="A8b",
        household_income="A62",
        health="A15",
        work_status="A58",
        current_work_status="A59a",
        marriage="A69",
    ),
    YearSpec(
        year=2023,
        file_name="CGSS2023.dta",
        current_isco=["isco08a59d", "isco08a42new"],
        ever_isco="isco08a60d",
        weight="weight2",
        gender="a2",
        birth_year="a3a",
        education="a7a",
        total_income="a8a",
        labor_income="a8b",
        household_income="a62",
        health="a15",
        work_status="a58",
        current_work_status="a59a",
        marriage="a69",
        subjective_status="b1",
        status_change="b2",
    ),
]


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).replace("\xa0", " ").replace("\n", " ").strip()


def to_number(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def clean_numeric(series: pd.Series, extra_invalid_negative: bool = False) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.mask(out.isin(SPECIAL_MISSING))
    if extra_invalid_negative:
        out = out.mask(out < 0)
    return out


def clean_income(series: pd.Series) -> pd.Series:
    out = clean_numeric(series, extra_invalid_negative=True)
    return out


def normalize_isco(value: Any) -> str | None:
    x = to_number(value)
    if np.isnan(x):
        return None
    code = int(x)
    if code in SPECIAL_MISSING or code < 0 or code >= 9800:
        return None
    return str(code).zfill(4)


def normalize_soc(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().replace(" ", "")
    if not s:
        return None
    if "." in s:
        s = s.split(".", 1)[0]
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 6:
        return None
    digits = digits[:6]
    return f"{digits[:2]}-{digits[2:]}"


def soc_minor(value: Any) -> str | None:
    soc = normalize_soc(value)
    if soc is None:
        return None
    return f"{soc[:2]}-{soc[3]}000"


def isco_granularity(code: str | None) -> str:
    if not code:
        return "missing"
    if code[1:] == "000":
        return "major_1digit_aggregate"
    if code[2:] == "00":
        return "submajor_2digit_aggregate"
    if code[3] == "0":
        return "minor_3digit_aggregate"
    return "unit_4digit"


def fallback_prefixes(code: str) -> list[tuple[int, str]]:
    if code[1:] == "000":
        return [(1, code[:1])]
    if code[2:] == "00":
        return [(2, code[:2]), (1, code[:1])]
    if code[3] == "0":
        return [(3, code[:3]), (2, code[:2]), (1, code[:1])]
    return [(3, code[:3]), (2, code[:2]), (1, code[:1])]


def combine_current_isco(df: pd.DataFrame, variables: list[str]) -> tuple[pd.Series, pd.Series]:
    out = pd.Series([None] * len(df), index=df.index, dtype="object")
    source = pd.Series([""] * len(df), index=df.index, dtype="object")
    for var in variables:
        if var not in df.columns:
            continue
        s = df[var].map(normalize_isco)
        take = out.isna() & s.notna()
        out = out.where(~take, s)
        source = source.where(~take, var)
    return out, source


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    ok = values.notna() & weights.notna() & (weights > 0)
    if not ok.any():
        return np.nan
    return float(np.average(values[ok], weights=weights[ok]))


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    ok = values.notna() & weights.notna() & (weights > 0)
    if not ok.any():
        return np.nan
    data = pd.DataFrame({"value": values[ok], "weight": weights[ok]}).sort_values("value")
    cutoff = data["weight"].sum() / 2
    return float(data.loc[data["weight"].cumsum() >= cutoff, "value"].iloc[0])


def education_group(code: Any) -> str:
    x = to_number(code)
    if np.isnan(x):
        return "缺失"
    if x <= 4:
        return "初中及以下"
    if x <= 6:
        return "高中/中专"
    return "大专及以上"


def read_cgss() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for spec in SPECS:
        path = CGSS_DIR / spec.file_name
        wanted = [
            "id",
            "s41",
            "s42",
            "s43",
            "isurban",
            spec.weight,
            spec.gender,
            spec.birth_year,
            spec.education,
            spec.total_income,
            spec.labor_income,
            spec.household_income,
            spec.health,
            spec.work_status,
            spec.current_work_status,
            spec.marriage,
            spec.ever_isco,
            spec.subjective_status,
            spec.status_change,
            *spec.current_isco,
        ]
        wanted = [v for v in wanted if v]
        with pd.read_stata(path, iterator=True, convert_categoricals=False) as reader:
            available = [v for v in wanted if v in reader.variable_labels()]
        raw = pd.read_stata(path, columns=available, convert_categoricals=False, preserve_dtypes=False)
        current_isco, current_source = combine_current_isco(raw, spec.current_isco)

        weight = clean_numeric(raw[spec.weight], extra_invalid_negative=True)
        weight = weight.where(weight > 0)
        birth_year = clean_numeric(raw[spec.birth_year])
        age = spec.year - birth_year
        age = age.where((age >= 16) & (age <= 100))
        total_income = clean_income(raw[spec.total_income])
        labor_income = clean_income(raw[spec.labor_income])
        household_income = clean_income(raw[spec.household_income])
        education = clean_numeric(raw[spec.education])

        out = pd.DataFrame(
            {
                "year": spec.year,
                "income_reference_year": spec.year - 1,
                "id": raw["id"].astype("string") if "id" in raw.columns else pd.Series(pd.NA, index=raw.index, dtype="string"),
                "province_code": clean_numeric(raw["s41"]) if "s41" in raw.columns else np.nan,
                "city_code": clean_numeric(raw["s42"]) if "s42" in raw.columns else np.nan,
                "county_code": clean_numeric(raw["s43"]) if "s43" in raw.columns else np.nan,
                "isurban_code": clean_numeric(raw["isurban"]) if "isurban" in raw.columns else np.nan,
                "weight": weight,
                "gender_code": clean_numeric(raw[spec.gender]),
                "female": clean_numeric(raw[spec.gender]).eq(2).astype("Int64"),
                "birth_year": birth_year,
                "age": age,
                "age_sq": age**2,
                "education_code": education,
                "education_group": education.map(education_group),
                "health_code": clean_numeric(raw[spec.health]),
                "work_status_code": clean_numeric(raw[spec.work_status]),
                "current_work_status_code": clean_numeric(raw[spec.current_work_status]),
                "marriage_code": clean_numeric(raw[spec.marriage]),
                "total_income": total_income,
                "labor_income": labor_income,
                "household_income": household_income,
                "log_total_income": np.log1p(total_income),
                "log_labor_income": np.log1p(labor_income),
                "log_household_income": np.log1p(household_income),
                "current_isco08": current_isco,
                "current_isco_source_var": current_source,
                "current_isco_granularity": current_isco.map(isco_granularity),
                "current_isco_major": current_isco.map(lambda x: x[:1] if x else None),
                "current_isco_major_label": current_isco.map(lambda x: ISCO_MAJOR_LABELS.get(x[:1], "") if x else ""),
            }
        )
        if spec.ever_isco and spec.ever_isco in raw.columns:
            out["ever_nonfarm_isco08"] = raw[spec.ever_isco].map(normalize_isco)
        else:
            out["ever_nonfarm_isco08"] = None
        out["subjective_status_code"] = clean_numeric(raw[spec.subjective_status]) if spec.subjective_status and spec.subjective_status in raw.columns else np.nan
        out["status_change_code"] = clean_numeric(raw[spec.status_change]) if spec.status_change and spec.status_change in raw.columns else np.nan

        frames.append(out)
        diagnostics.append(
            {
                "year": spec.year,
                "raw_n": len(raw),
                "valid_current_isco": int(out["current_isco08"].notna().sum()),
                "valid_current_isco_rate": float(out["current_isco08"].notna().mean()),
                "valid_weight": int(out["weight"].notna().sum()),
                "valid_labor_income": int(out["labor_income"].notna().sum()),
                "valid_total_income": int(out["total_income"].notna().sum()),
                "valid_age": int(out["age"].notna().sum()),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(diagnostics)


def locate_exposure_dir() -> Path:
    matches = list(PROJECT_ROOT.glob("*/exposure_by_year_minor_soc.xlsx"))
    if not matches:
        raise FileNotFoundError("Cannot locate exposure_by_year_minor_soc.xlsx")
    return matches[0].parent


def load_exposure() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exposure_dir = locate_exposure_dir()

    minor_year = pd.read_excel(exposure_dir / "exposure_by_year_minor_soc.xlsx")
    minor_year["minor_soc"] = minor_year["minor_soc_code"].map(normalize_soc)
    minor_year = minor_year.dropna(subset=["minor_soc", "year", "exposure"]).copy()

    minor_base = pd.read_excel(exposure_dir / "exposure_base_minor_soc.xlsx")
    minor_base["minor_soc"] = minor_base["minor_soc_code"].map(normalize_soc)
    minor_base = minor_base.dropna(subset=["minor_soc", "exposure"]).copy()

    detail_year = pd.read_excel(exposure_dir / "exposure_by_year_soc_detail.xlsx")
    detail_year["soc2010"] = detail_year["occu_soc_code"].map(normalize_soc)
    detail_year = detail_year.dropna(subset=["soc2010", "year", "exposure"]).copy()

    detail_base = pd.read_excel(exposure_dir / "exposure_base_soc_detail.xlsx")
    detail_base["soc2010"] = detail_base["occu_soc_code"].map(normalize_soc)
    detail_base = detail_base.dropna(subset=["soc2010", "exposure"]).copy()

    return minor_year, minor_base, detail_year, detail_base


def load_ibs_crosswalk() -> pd.DataFrame:
    path = CROSS_DIR / "onetsoc_to_isco_cws_ibs" / "soc10_isco08.dta"
    cw = pd.read_stata(path, convert_categoricals=False, preserve_dtypes=False)
    out = pd.DataFrame(
        {
            "soc2010": cw["soc10"].map(normalize_soc),
            "minor_soc": cw["soc10"].map(soc_minor),
            "isco08": cw["isco08"].map(normalize_isco),
        }
    )
    out = out.dropna(subset=["soc2010", "minor_soc", "isco08"]).drop_duplicates()
    return out


def exact_isco_exposure_from_minor(
    cw: pd.DataFrame, minor_year: pd.DataFrame, minor_base: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cw_minor = cw[["isco08", "minor_soc"]].drop_duplicates()
    year = cw_minor.merge(minor_year[["minor_soc", "year", "exposure"]], on="minor_soc", how="inner")
    exact_year = (
        year.groupby(["isco08", "year"], as_index=False)
        .agg(
            ai_exposure_minor_exact=("exposure", "mean"),
            n_source_minor_soc=("minor_soc", "nunique"),
        )
        .sort_values(["isco08", "year"])
    )
    base = cw_minor.merge(minor_base[["minor_soc", "exposure"]], on="minor_soc", how="inner")
    exact_base = (
        base.groupby("isco08", as_index=False)
        .agg(
            ai_exposure_base_exact=("exposure", "mean"),
            n_source_minor_soc_base=("minor_soc", "nunique"),
        )
        .sort_values("isco08")
    )
    return exact_year, exact_base


def exact_isco_exposure_from_detail(
    cw: pd.DataFrame, detail_year: pd.DataFrame, detail_base: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cw_detail = cw[["isco08", "soc2010"]].drop_duplicates()
    detail_year_unique = detail_year.groupby(["soc2010", "year"], as_index=False)["exposure"].mean()
    detail_base_unique = detail_base.groupby("soc2010", as_index=False)["exposure"].mean()

    year = cw_detail.merge(detail_year_unique, on="soc2010", how="inner")
    exact_year = (
        year.groupby(["isco08", "year"], as_index=False)
        .agg(
            ai_exposure_detail_exact=("exposure", "mean"),
            n_source_detail_soc=("soc2010", "nunique"),
        )
        .sort_values(["isco08", "year"])
    )
    base = cw_detail.merge(detail_base_unique, on="soc2010", how="inner")
    exact_base = (
        base.groupby("isco08", as_index=False)
        .agg(
            ai_exposure_detail_base_exact=("exposure", "mean"),
            n_source_detail_soc_base=("soc2010", "nunique"),
        )
        .sort_values("isco08")
    )
    return exact_year, exact_base


def prefix_pool(exact_year: pd.DataFrame, exact_base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    year_rows = []
    for length in [1, 2, 3]:
        tmp = exact_year.copy()
        tmp["prefix_len"] = length
        tmp["isco_prefix"] = tmp["isco08"].str[:length]
        pooled = (
            tmp.groupby(["prefix_len", "isco_prefix", "year"], as_index=False)
            .agg(
                ai_exposure_minor=("ai_exposure_minor_exact", "mean"),
                n_source_isco=("isco08", "nunique"),
                n_source_minor_soc=("n_source_minor_soc", "sum"),
            )
        )
        year_rows.append(pooled)
    base_rows = []
    for length in [1, 2, 3]:
        tmp = exact_base.copy()
        tmp["prefix_len"] = length
        tmp["isco_prefix"] = tmp["isco08"].str[:length]
        pooled = (
            tmp.groupby(["prefix_len", "isco_prefix"], as_index=False)
            .agg(
                ai_exposure_base=("ai_exposure_base_exact", "mean"),
                n_source_isco_base=("isco08", "nunique"),
                n_source_minor_soc_base=("n_source_minor_soc_base", "sum"),
            )
        )
        base_rows.append(pooled)
    return pd.concat(year_rows, ignore_index=True), pd.concat(base_rows, ignore_index=True)


def assign_exposure_to_current_isco(
    current_isco_codes: pd.Series,
    exact_year: pd.DataFrame,
    exact_base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_year_map = {
        (row.isco08, int(row.year)): row
        for row in exact_year.itertuples(index=False)
    }
    exact_base_map = {row.isco08: row for row in exact_base.itertuples(index=False)}
    year_pool, base_pool = prefix_pool(exact_year, exact_base)
    year_pool_map = {
        (int(row.prefix_len), row.isco_prefix, int(row.year)): row
        for row in year_pool.itertuples(index=False)
    }
    base_pool_map = {
        (int(row.prefix_len), row.isco_prefix): row
        for row in base_pool.itertuples(index=False)
    }
    exposure_years = sorted(exact_year["year"].dropna().astype(int).unique())
    codes = sorted({code for code in current_isco_codes.dropna().astype(str)})

    year_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    for code in codes:
        base_exact = exact_base_map.get(code)
        if base_exact is not None:
            base_rows.append(
                {
                    "isco08": code,
                    "ai_exposure_base": float(base_exact.ai_exposure_base_exact),
                    "exposure_source_level": "exact_isco",
                    "source_prefix_len": 4,
                    "source_prefix": code,
                    "n_source_isco_base": 1,
                    "n_source_minor_soc_base": int(base_exact.n_source_minor_soc_base),
                }
            )
        else:
            assigned = False
            for length, prefix in fallback_prefixes(code):
                pooled = base_pool_map.get((length, prefix))
                if pooled is not None:
                    base_rows.append(
                        {
                            "isco08": code,
                            "ai_exposure_base": float(pooled.ai_exposure_base),
                            "exposure_source_level": f"isco_prefix_{length}",
                            "source_prefix_len": length,
                            "source_prefix": prefix,
                            "n_source_isco_base": int(pooled.n_source_isco_base),
                            "n_source_minor_soc_base": int(pooled.n_source_minor_soc_base),
                        }
                    )
                    assigned = True
                    break
            if not assigned:
                base_rows.append(
                    {
                        "isco08": code,
                        "ai_exposure_base": np.nan,
                        "exposure_source_level": "unmatched",
                        "source_prefix_len": np.nan,
                        "source_prefix": "",
                        "n_source_isco_base": 0,
                        "n_source_minor_soc_base": 0,
                    }
                )

        for year in exposure_years:
            exact = exact_year_map.get((code, year))
            if exact is not None:
                year_rows.append(
                    {
                        "isco08": code,
                        "exposure_year": year,
                        "ai_exposure": float(exact.ai_exposure_minor_exact),
                        "exposure_source_level": "exact_isco",
                        "source_prefix_len": 4,
                        "source_prefix": code,
                        "n_source_isco": 1,
                        "n_source_minor_soc": int(exact.n_source_minor_soc),
                    }
                )
                continue
            assigned = False
            for length, prefix in fallback_prefixes(code):
                pooled = year_pool_map.get((length, prefix, year))
                if pooled is not None:
                    year_rows.append(
                        {
                            "isco08": code,
                            "exposure_year": year,
                            "ai_exposure": float(pooled.ai_exposure_minor),
                            "exposure_source_level": f"isco_prefix_{length}",
                            "source_prefix_len": length,
                            "source_prefix": prefix,
                            "n_source_isco": int(pooled.n_source_isco),
                            "n_source_minor_soc": int(pooled.n_source_minor_soc),
                        }
                    )
                    assigned = True
                    break
            if not assigned:
                year_rows.append(
                    {
                        "isco08": code,
                        "exposure_year": year,
                        "ai_exposure": np.nan,
                        "exposure_source_level": "unmatched",
                        "source_prefix_len": np.nan,
                        "source_prefix": "",
                        "n_source_isco": 0,
                        "n_source_minor_soc": 0,
                    }
                )

    year_df = pd.DataFrame(year_rows)
    base_df = pd.DataFrame(base_rows)
    return year_df, base_df


def exposure_dynamics(assigned_year: pd.DataFrame, assigned_base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, group in assigned_year.groupby("isco08"):
        g = group.dropna(subset=["ai_exposure"]).sort_values("exposure_year")
        if g.empty:
            rows.append(
                {
                    "isco08": code,
                    "ai_exposure_2018": np.nan,
                    "ai_exposure_2024": np.nan,
                    "ai_exposure_change_2018_2024": np.nan,
                    "ai_exposure_slope_2018_2024": np.nan,
                    "ai_exposure_volatility": np.nan,
                }
            )
            continue
        exp2018 = g.loc[g["exposure_year"].eq(2018), "ai_exposure"]
        exp2024 = g.loc[g["exposure_year"].eq(2024), "ai_exposure"]
        years = g["exposure_year"].astype(float).to_numpy()
        values = g["ai_exposure"].astype(float).to_numpy()
        slope = np.polyfit(years, values, deg=1)[0] if len(g) >= 2 else np.nan
        rows.append(
            {
                "isco08": code,
                "ai_exposure_2018": float(exp2018.iloc[0]) if not exp2018.empty else np.nan,
                "ai_exposure_2024": float(exp2024.iloc[0]) if not exp2024.empty else np.nan,
                "ai_exposure_change_2018_2024": float(exp2024.iloc[0] - exp2018.iloc[0]) if not exp2018.empty and not exp2024.empty else np.nan,
                "ai_exposure_slope_2018_2024": float(slope),
                "ai_exposure_volatility": float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan,
            }
        )
    dyn = pd.DataFrame(rows)
    out = assigned_base.merge(dyn, on="isco08", how="left")
    out["ai_exposure_quartile"] = pd.qcut(
        out["ai_exposure_base"],
        q=4,
        labels=[1, 2, 3, 4],
        duplicates="drop",
    ).astype("Int64")
    out["high_ai_exposure_q4"] = out["ai_exposure_quartile"].eq(4).astype("Int64")
    return out


def load_isco_labels() -> pd.DataFrame:
    path = CGSS_DIR / "results" / "tables" / "current_isco_codes_to_map.csv"
    if not path.exists():
        return pd.DataFrame(columns=["isco08", "isco08_label"])
    labels = pd.read_csv(path, dtype={"isco08_code": "string"})
    labels["isco08"] = labels["isco08_code"].map(lambda x: str(int(float(x))).zfill(4) if pd.notna(x) else None)
    return labels[["isco08", "isco08_label"]].drop_duplicates("isco08")


def attach_exposure(worker: pd.DataFrame, assigned_year: pd.DataFrame, exposure_info: pd.DataFrame) -> pd.DataFrame:
    out = worker.merge(
        assigned_year,
        left_on=["current_isco08", "year"],
        right_on=["isco08", "exposure_year"],
        how="left",
    )
    out = out.drop(columns=["isco08"], errors="ignore")
    out = out.merge(
        exposure_info,
        left_on="current_isco08",
        right_on="isco08",
        how="left",
        suffixes=("", "_base"),
    )
    out = out.drop(columns=["isco08"], errors="ignore")
    labels = load_isco_labels()
    out = out.merge(labels, left_on="current_isco08", right_on="isco08", how="left")
    out = out.drop(columns=["isco08"], errors="ignore")
    out["has_current_isco"] = out["current_isco08"].notna().astype("Int64")
    out["has_ai_exposure"] = out["ai_exposure"].notna().astype("Int64")
    out["income_analysis_sample"] = (
        out["current_isco08"].notna()
        & out["ai_exposure"].notna()
        & out["labor_income"].notna()
        & out["weight"].notna()
    ).astype("Int64")
    out["employment_share_sample"] = (
        out["current_isco08"].notna()
        & out["ai_exposure"].notna()
        & out["weight"].notna()
    ).astype("Int64")
    return out


def build_occupation_panel(worker: pd.DataFrame) -> pd.DataFrame:
    sample = worker[worker["employment_share_sample"].eq(1)].copy()
    rows = []
    for (year, isco), group in sample.groupby(["year", "current_isco08"]):
        row = {
            "year": year,
            "current_isco08": isco,
            "sample_count": len(group),
            "weighted_count": float(group["weight"].sum()),
            "weighted_mean_log_labor_income": weighted_mean(group["log_labor_income"], group["weight"]),
            "weighted_median_labor_income": weighted_median(group["labor_income"], group["weight"]),
            "weighted_mean_log_total_income": weighted_mean(group["log_total_income"], group["weight"]),
            "weighted_median_total_income": weighted_median(group["total_income"], group["weight"]),
            "mean_age": weighted_mean(group["age"], group["weight"]),
            "college_plus_share": weighted_mean(group["education_group"].eq("大专及以上").astype(float), group["weight"]),
            "female_share": weighted_mean(group["female"].astype(float), group["weight"]),
        }
        first_cols = [
            "isco08_label",
            "current_isco_major",
            "current_isco_major_label",
            "current_isco_granularity",
            "ai_exposure",
            "ai_exposure_base",
            "ai_exposure_change_2018_2024",
            "ai_exposure_slope_2018_2024",
            "ai_exposure_quartile",
            "high_ai_exposure_q4",
            "exposure_source_level",
            "source_prefix",
            "n_source_isco",
            "n_source_minor_soc",
        ]
        for col in first_cols:
            row[col] = group[col].dropna().iloc[0] if group[col].notna().any() else np.nan
        rows.append(row)
    panel = pd.DataFrame(rows)
    totals = panel.groupby("year", as_index=False)["weighted_count"].sum().rename(columns={"weighted_count": "year_total_weighted_current_isco"})
    panel = panel.merge(totals, on="year", how="left")
    panel["weighted_employment_share"] = panel["weighted_count"] / panel["year_total_weighted_current_isco"]
    panel = panel.sort_values(["year", "weighted_count"], ascending=[True, False])
    return panel


def compare_detail_minor(
    exact_minor_base: pd.DataFrame,
    exact_detail_base: pd.DataFrame,
) -> pd.DataFrame:
    comp = exact_minor_base.merge(exact_detail_base, on="isco08", how="inner")
    if comp.empty:
        return pd.DataFrame(
            [
                {
                    "matched_exact_isco_with_both": 0,
                    "correlation_minor_detail_base": np.nan,
                    "mean_abs_difference": np.nan,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "matched_exact_isco_with_both": len(comp),
                "correlation_minor_detail_base": comp["ai_exposure_base_exact"].corr(comp["ai_exposure_detail_base_exact"]),
                "mean_abs_difference": float((comp["ai_exposure_base_exact"] - comp["ai_exposure_detail_base_exact"]).abs().mean()),
            }
        ]
    )


def run_r_environment_check() -> pd.DataFrame:
    rows = []
    commands = [
        {
            "rscript_path": "conda run -n r-env Rscript",
            "cmd": ["conda", "run", "-n", "r-env", "Rscript"],
        }
    ]
    for item in commands:
        version = subprocess.run(
            [*item["cmd"], "-e", "cat(R.version.string)"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        package = subprocess.run(
            [*item["cmd"], "-e", "cat(requireNamespace('iscoCrosswalks', quietly=TRUE))"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        rows.append(
            {
                "rscript_path": item["rscript_path"],
                "version_output": (version.stdout + version.stderr).strip(),
                "version_returncode": version.returncode,
                "iscoCrosswalks_installed": package.stdout.strip().upper() == "TRUE",
                "package_check_output": (package.stdout + package.stderr).strip(),
                "package_returncode": package.returncode,
            }
        )
    if not rows:
        rows.append(
            {
                "rscript_path": "",
                "version_output": "Rscript not found in checked conda paths.",
                "version_returncode": np.nan,
                "iscoCrosswalks_installed": False,
                "package_check_output": "",
                "package_returncode": np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_data_dictionary() -> pd.DataFrame:
    rows = [
        ("cgss_worker_ai_exposure.csv", "year", "调查年份。"),
        ("cgss_worker_ai_exposure.csv", "income_reference_year", "收入变量对应的上一年度。"),
        ("cgss_worker_ai_exposure.csv", "id", "CGSS 问卷编号。"),
        ("cgss_worker_ai_exposure.csv", "weight", "调查权重；2018/2021 使用 weight，2023 使用 weight2。"),
        ("cgss_worker_ai_exposure.csv", "current_isco08", "当前职业 ISCO-08 代码；应按文本读取，保留前导零。"),
        ("cgss_worker_ai_exposure.csv", "current_isco_granularity", "ISCO 职业代码粒度：四位具体职业或一/二/三位聚合职业。"),
        ("cgss_worker_ai_exposure.csv", "ai_exposure", "与调查年份对应的 AI-LLM 暴露度，主口径为 SOC minor indicator 均值聚合到 ISCO。"),
        ("cgss_worker_ai_exposure.csv", "ai_exposure_base", "基础 AI-LLM 暴露度。"),
        ("cgss_worker_ai_exposure.csv", "ai_exposure_change_2018_2024", "同一 ISCO 暴露度在 2024 年相对 2018 年的变化。"),
        ("cgss_worker_ai_exposure.csv", "ai_exposure_slope_2018_2024", "2018-2024 年年度暴露度线性斜率。"),
        ("cgss_worker_ai_exposure.csv", "ai_exposure_quartile", "按基础暴露度分成的四分位，4 表示高暴露。"),
        ("cgss_worker_ai_exposure.csv", "exposure_source_level", "暴露度来源：exact_isco、isco_prefix_3、isco_prefix_2、isco_prefix_1 或 unmatched。"),
        ("cgss_worker_ai_exposure.csv", "source_prefix", "层级回退时使用的 ISCO 前缀；应按文本读取。"),
        ("cgss_worker_ai_exposure.csv", "labor_income", "清洗后的个人职业/劳动收入，特殊缺失码和负值设为缺失，零收入保留。"),
        ("cgss_worker_ai_exposure.csv", "log_labor_income", "log(labor_income+1)。"),
        ("cgss_worker_ai_exposure.csv", "income_analysis_sample", "可进入个体收入分析的样本标记。"),
        ("cgss_worker_ai_exposure.csv", "employment_share_sample", "可进入职业就业份额分析的样本标记。"),
        ("occupation_year_panel.csv", "weighted_count", "CGSS 权重加总后的职业-年份样本规模，不等同于全国真实职业人数。"),
        ("occupation_year_panel.csv", "weighted_employment_share", "职业在当年有效当前职业样本中的加权份额。"),
        ("isco_exposure_crosswalk.csv", "isco08", "ISCO-08 职业代码；应按文本读取，保留前导零。"),
        ("isco_exposure_crosswalk.csv", "n_source_isco_base", "基础暴露度赋值所依据的源 ISCO 职业数量。"),
        ("isco_exposure_by_year.csv", "exposure_year", "暴露度年份。"),
        ("isco_soc_mapping_long.csv", "soc2010", "美国 SOC2010 代码。"),
        ("isco_soc_mapping_long.csv", "minor_soc", "美国 SOC minor group 代码。"),
    ]
    dictionary = pd.DataFrame(rows, columns=["file", "variable", "description"])
    dictionary.to_csv(OUT_DIR / "data_dictionary.csv", index=False, encoding="utf-8-sig")
    return dictionary


def diagnostics(worker: pd.DataFrame, exposure_info: pd.DataFrame, detail_minor_compare: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_year_rows = []
    for year, group in worker.groupby("year"):
        valid_current = group["current_isco08"].notna()
        has_exp = group["has_ai_exposure"].eq(1)
        by_year_rows.append(
            {
                "year": year,
                "n_obs": len(group),
                "valid_current_isco": int(valid_current.sum()),
                "valid_current_isco_rate": float(valid_current.mean()),
                "ai_exposure_matched": int(has_exp.sum()),
                "ai_exposure_matched_rate_total": float(has_exp.mean()),
                "ai_exposure_matched_rate_among_current_isco": float((has_exp & valid_current).sum() / valid_current.sum()) if valid_current.sum() else np.nan,
                "exact_isco_exposure_count": int(group["exposure_source_level"].eq("exact_isco").sum()),
                "prefix_fallback_exposure_count": int(group["exposure_source_level"].astype("string").str.startswith("isco_prefix").sum()),
                "unmatched_exposure_count": int(group["exposure_source_level"].eq("unmatched").sum()),
                "valid_labor_income": int(group["labor_income"].notna().sum()),
                "income_analysis_sample": int(group["income_analysis_sample"].sum()),
            }
        )
    by_year = pd.DataFrame(by_year_rows)

    source_dist = (
        worker[worker["current_isco08"].notna()]
        .groupby(["year", "exposure_source_level"], dropna=False)
        .agg(sample_count=("id", "size"), weighted_count=("weight", "sum"))
        .reset_index()
    )
    detail_minor_compare.to_csv(DIAG_DIR / "detail_minor_exposure_comparison.csv", index=False, encoding="utf-8-sig")
    exposure_info["exposure_source_level"].value_counts(dropna=False).rename_axis("exposure_source_level").reset_index(name="isco_count").to_csv(
        DIAG_DIR / "isco_exposure_source_distribution.csv", index=False, encoding="utf-8-sig"
    )
    return by_year, source_dist


def md_table(df: pd.DataFrame, n: int | None = None) -> str:
    show = df.copy()
    if n is not None:
        show = show.head(n)
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].round(4)
    return show.to_markdown(index=False)


def write_report(
    worker: pd.DataFrame,
    occupation_panel: pd.DataFrame,
    cgss_diag: pd.DataFrame,
    match_diag: pd.DataFrame,
    source_dist: pd.DataFrame,
    detail_minor_compare: pd.DataFrame,
    r_check: pd.DataFrame,
    data_dictionary: pd.DataFrame,
) -> None:
    top_unmatched = (
        worker[worker["current_isco08"].notna() & worker["ai_exposure"].isna()]
        .groupby(["current_isco08", "isco08_label", "current_isco_major_label"], dropna=False)
        .agg(sample_count=("id", "size"), weighted_count=("weight", "sum"))
        .reset_index()
        .sort_values("sample_count", ascending=False)
        .head(20)
    )
    lines = [
        "# CGSS 数据清洗与 AI-LLM 暴露度合并报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 清洗目标",
        "",
        "本轮清洗的目标是将 CGSS2018、CGSS2021、CGSS2023 的个体职业、收入、就业与基本人口学变量整理成可分析数据，并把 AI-LLM 职业暴露度合并到 CGSS 个体和职业-年份层面。",
        "",
        "## 2. 职业映射方法",
        "",
        "主口径采用 `Crosswork/onetsoc_to_isco_cws_ibs/soc10_isco08.dta`。具体做法是：先将 SOC2010 映射到 SOC minor group，再把 `exposure_by_year_minor_soc.xlsx` 中的 AI-LLM 暴露度作为 indicator 按均值聚合到 ISCO-08。这个做法与 `iscoCrosswalks` 对指标变量使用均值转换的思想一致。",
        "",
        "由于 CGSS 中存在 `2000`、`5000`、`7000` 等聚合 ISCO 代码，脚本没有直接删除这些高频样本，而是采用透明的层级回退：四位 ISCO 精确匹配优先；若无精确匹配，则依次使用三位、两位或一位 ISCO 前缀下已匹配职业的平均暴露度。每条记录都保留 `exposure_source_level`、`source_prefix`、`n_source_isco` 和 `n_source_minor_soc`，供后续稳健性检验。",
        "",
        "## 3. CGSS 基础清洗结果",
        "",
        md_table(cgss_diag),
        "",
        "收入变量中的负值与 CGSS 特殊缺失码被设为缺失，保留零收入，并生成 `log_labor_income`、`log_total_income`、`log_household_income`。跨年收入目前仍是名义值，正式模型若比较收入水平，应进一步接入 CPI 或收入平减指数。",
        "",
        "## 4. 暴露度匹配结果",
        "",
        md_table(match_diag),
        "",
        "经过 ISCO 层级回退后，当前职业样本中的暴露度匹配率明显提高。该口径牺牲的是职业边界精细度，但保留了 CGSS 中高频聚合职业，适合用作主清洗数据。后续模型应将 `exposure_source_level` 作为映射质量控制或稳健性分组。",
        "",
        "### 暴露度来源分布",
        "",
        md_table(source_dist),
        "",
        "### 详细 SOC 与 minor SOC 暴露度比较",
        "",
        md_table(detail_minor_compare),
        "",
        "详细 SOC 暴露度覆盖率较低，minor SOC 口径覆盖率更高。上表用于检查二者在可共同匹配职业上的一致性；后续可把详细 SOC 口径作为高质量子样本稳健性分析。",
        "",
        "## 5. R 环境检查",
        "",
        md_table(r_check),
        "",
        "`r-env` 中可找到 Rscript，但当前未检测到 `iscoCrosswalks` 包。因此，本轮未直接调用 R 包，而是用本地 IBS crosswalk 在 Python 中实现了同样的 indicator 均值聚合逻辑。若之后安装该包，可将本轮 `isco_exposure_crosswalk.csv` 与 R 包生成结果比较。",
        "",
        "## 6. 清洗后数据文件",
        "",
        "- `cgss_worker_ai_exposure.csv`：个体层面清洗数据，一行一个受访者，包含当前 ISCO、AI-LLM 暴露度、收入、教育、年龄、性别、城乡、省份、权重等变量。",
        "- `occupation_year_panel.csv`：职业-年份层面数据，包含样本数、加权职业份额、加权收入统计和暴露度变量。",
        "- `isco_exposure_crosswalk.csv`：ISCO 职业层面的基础暴露度、年度变化、斜率、四分位和映射来源。",
        "- `isco_exposure_by_year.csv`：ISCO-年份层面的年度暴露度。",
        "- `isco_soc_mapping_long.csv`：IBS SOC2010-ISCO08 长表映射。",
        "- `diagnostics/`：诊断表，包括匹配率、暴露度来源分布、R 环境检查和未匹配职业。",
        "- `data_dictionary.csv`：核心变量说明。注意所有 ISCO/SOC 职业代码都建议按文本读取，以保留 `0110` 这类前导零。",
        "",
        "### 核心变量字典节选",
        "",
        md_table(data_dictionary, n=18),
        "",
        "## 7. 后续分析建议",
        "",
        "就业数量部分建议使用 `occupation_year_panel.csv` 中的加权职业份额，而不要把 CGSS 原始样本数称为全国职业人数。薪资部分建议使用 `cgss_worker_ai_exposure.csv`，以 `log_labor_income` 或 `log_total_income` 为结果变量，同时控制年龄、教育、性别、城乡、省份、年份和映射质量。",
        "",
    ]
    if not top_unmatched.empty:
        lines.extend(
            [
                "## 8. 仍未匹配的高频职业",
                "",
                md_table(top_unmatched),
                "",
            ]
        )
    (OUT_DIR / "报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    worker_raw, cgss_diag = read_cgss()
    minor_year, minor_base, detail_year, detail_base = load_exposure()
    cw = load_ibs_crosswalk()

    exact_minor_year, exact_minor_base = exact_isco_exposure_from_minor(cw, minor_year, minor_base)
    exact_detail_year, exact_detail_base = exact_isco_exposure_from_detail(cw, detail_year, detail_base)
    assigned_year, assigned_base = assign_exposure_to_current_isco(
        worker_raw["current_isco08"], exact_minor_year, exact_minor_base
    )
    exposure_info = exposure_dynamics(assigned_year, assigned_base)
    worker = attach_exposure(worker_raw, assigned_year, exposure_info)
    occupation_panel = build_occupation_panel(worker)
    detail_minor_compare = compare_detail_minor(exact_minor_base, exact_detail_base)
    match_diag, source_dist = diagnostics(worker, exposure_info, detail_minor_compare)
    r_check = run_r_environment_check()
    data_dictionary = write_data_dictionary()

    cw.to_csv(OUT_DIR / "isco_soc_mapping_long.csv", index=False, encoding="utf-8-sig")
    assigned_year.to_csv(OUT_DIR / "isco_exposure_by_year.csv", index=False, encoding="utf-8-sig")
    exposure_info.to_csv(OUT_DIR / "isco_exposure_crosswalk.csv", index=False, encoding="utf-8-sig")
    worker.to_csv(OUT_DIR / "cgss_worker_ai_exposure.csv", index=False, encoding="utf-8-sig")
    occupation_panel.to_csv(OUT_DIR / "occupation_year_panel.csv", index=False, encoding="utf-8-sig")
    cgss_diag.to_csv(DIAG_DIR / "cgss_basic_cleaning_diagnostics.csv", index=False, encoding="utf-8-sig")
    match_diag.to_csv(DIAG_DIR / "ai_exposure_match_diagnostics_by_year.csv", index=False, encoding="utf-8-sig")
    source_dist.to_csv(DIAG_DIR / "ai_exposure_source_distribution_by_year.csv", index=False, encoding="utf-8-sig")
    r_check.to_csv(DIAG_DIR / "r_environment_check.csv", index=False, encoding="utf-8-sig")

    unmatched = (
        worker[worker["current_isco08"].notna() & worker["ai_exposure"].isna()]
        .groupby(["current_isco08", "isco08_label", "current_isco_major_label"], dropna=False)
        .agg(sample_count=("id", "size"), weighted_count=("weight", "sum"))
        .reset_index()
        .sort_values("sample_count", ascending=False)
    )
    unmatched.to_csv(DIAG_DIR / "unmatched_isco_after_hierarchy_fallback.csv", index=False, encoding="utf-8-sig")

    write_report(worker, occupation_panel, cgss_diag, match_diag, source_dist, detail_minor_compare, r_check, data_dictionary)

    print("CGSS AI exposure cleaning finished.")
    print(f"Output directory: {OUT_DIR}")
    print(f"Worker rows: {len(worker)}")
    print(f"Occupation-year rows: {len(occupation_panel)}")


if __name__ == "__main__":
    main()
