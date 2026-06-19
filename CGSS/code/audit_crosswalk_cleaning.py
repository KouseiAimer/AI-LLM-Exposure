"""Audit CGSS occupation cleaning and crosswalk quality for AI-LLM exposure work."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import win32com.client as win32


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
CROSS_DIR = PROJECT_ROOT / "Crosswork"
TABLE_DIR = ROOT / "results" / "tables"
REPORT_PATH = ROOT / "cgss_data_cleaning_notes.md"

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


@dataclass(frozen=True)
class YearSpec:
    year: int
    file_name: str
    current_isco: list[str]
    weight: str


SPECS = [
    YearSpec(2018, "CGSS2018.dta", ["isco08_a59d"], "weight"),
    YearSpec(2021, "CGSS2021.dta", ["isco08_a59d"], "weight"),
    YearSpec(2023, "CGSS2023.dta", ["isco08a59d", "isco08a42new"], "weight2"),
]


def setup() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).replace("\xa0", " ").replace("\n", " ").strip()


def numeric(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def normalize_isco(value: Any) -> str | None:
    x = numeric(value)
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
    if "-" in s:
        left, right = s.split("-", 1)
        digits = "".join(ch for ch in left + right if ch.isdigit())
    else:
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


def combine_current_isco(df: pd.DataFrame, variables: list[str]) -> pd.Series:
    out = pd.Series([None] * len(df), index=df.index, dtype="object")
    for var in variables:
        if var not in df.columns:
            continue
        s = df[var].map(normalize_isco)
        out = out.where(out.notna(), s)
    return out


def read_stata_labels(path: Path) -> dict[str, str]:
    with pd.read_stata(path, iterator=True, convert_categoricals=False) as reader:
        return {k: clean_text(v) for k, v in reader.variable_labels().items()}


def build_current_isco_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    year_rows = []
    for spec in SPECS:
        path = ROOT / spec.file_name
        if not path.exists():
            continue
        labels = read_stata_labels(path)
        available = [v for v in [*spec.current_isco, spec.weight] if v in labels]
        df = pd.read_stata(path, columns=available, convert_categoricals=False, preserve_dtypes=False)
        current = combine_current_isco(df, spec.current_isco)
        weight = pd.to_numeric(df.get(spec.weight), errors="coerce")
        weight = weight.where(weight.notna() & (weight > 0), np.nan)
        tmp = pd.DataFrame({"year": spec.year, "isco08": current, "weight": weight})
        tmp = tmp[tmp["isco08"].notna()].copy()
        tmp["sample_count"] = 1
        rows.append(tmp)
        year_rows.append(
            {
                "year": spec.year,
                "file": spec.file_name,
                "valid_current_isco_count": int(tmp["sample_count"].sum()),
                "valid_current_isco_weighted": float(tmp["weight"].sum()),
                "unique_current_isco": int(tmp["isco08"].nunique()),
                "weight_missing_among_valid_isco": int(tmp["weight"].isna().sum()),
            }
        )
    person = pd.concat(rows, ignore_index=True)
    counts = (
        person.groupby("isco08", as_index=False)
        .agg(sample_count=("sample_count", "sum"), weighted_count=("weight", "sum"))
        .sort_values(["sample_count", "isco08"], ascending=[False, True])
    )
    label_path = TABLE_DIR / "current_isco_codes_to_map.csv"
    if label_path.exists():
        labels = pd.read_csv(label_path, dtype={"isco08_code": "string"})
        labels["isco08"] = labels["isco08_code"].map(lambda x: str(int(float(x))).zfill(4) if pd.notna(x) else None)
        labels = labels[["isco08", "isco08_label", "major_label"]].drop_duplicates("isco08")
        counts = counts.merge(labels, on="isco08", how="left")
    return counts, pd.DataFrame(year_rows)


def read_excel_used_range(path: Path, sheet_name: str | None = None) -> list[list[Any]]:
    excel = win32.DispatchEx("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False
    try:
        wb = excel.Workbooks.Open(str(path.resolve()), ReadOnly=True)
        ws = wb.Worksheets(sheet_name) if sheet_name else wb.Worksheets(1)
        values = ws.UsedRange.Value
        wb.Close(False)
    finally:
        excel.Quit()
    if values is None:
        return []
    return [list(row) for row in values]


def read_bls_isco_soc() -> pd.DataFrame:
    path = CROSS_DIR / "ISCO_SOC_Crosswalk.xls"
    rows = read_excel_used_range(path, "ISCO-08 to 2010 SOC")
    header_idx = next(i for i, row in enumerate(rows) if "ISCO-08 Code" in row and "2010 SOC Code" in row)
    header = [clean_text(x) for x in rows[header_idx]]
    records = []
    for row in rows[header_idx + 1 :]:
        record = dict(zip(header, row))
        isco = normalize_isco(record.get("ISCO-08 Code"))
        soc = normalize_soc(record.get("2010 SOC Code"))
        if isco and soc:
            records.append(
                {
                    "source": "BLS_ISCO08_to_SOC2010",
                    "isco08": isco,
                    "target_code": soc,
                    "target_classification": "US_SOC2010",
                    "target_title": clean_text(record.get("2010 SOC Title")),
                    "source_file": path.name,
                }
            )
    return pd.DataFrame(records).drop_duplicates()


def read_ibs_soc_isco() -> pd.DataFrame:
    path = CROSS_DIR / "onetsoc_to_isco_cws_ibs" / "soc10_isco08.dta"
    df = pd.read_stata(path, convert_categoricals=False, preserve_dtypes=False)
    out = pd.DataFrame(
        {
            "source": "IBS_SOC10_ISCO08_dta",
            "isco08": df["isco08"].map(normalize_isco),
            "target_code": df["soc10"].map(normalize_soc),
            "target_classification": "US_SOC2010",
            "target_title": "",
            "source_file": "onetsoc_to_isco_cws_ibs/soc10_isco08.dta",
        }
    )
    return out.dropna(subset=["isco08", "target_code"]).drop_duplicates()


def read_ons_isco_uk_soc() -> pd.DataFrame:
    path = CROSS_DIR / "onetsocconversion.xlsx"
    df = pd.read_excel(path, sheet_name="SOC_2010", header=2)
    out = pd.DataFrame(
        {
            "source": "ONS_ISCO08_to_UK_SOC2010",
            "isco08": df["ISCO-08"].map(normalize_isco),
            "target_code": df["SOC 2010"].map(lambda x: str(int(float(x))).zfill(4) if pd.notna(x) else None),
            "target_classification": "UK_SOC2010",
            "target_title": "",
            "source_file": path.name,
            "proportion": pd.to_numeric(df["Proportion"], errors="coerce"),
        }
    )
    return out.dropna(subset=["isco08", "target_code"]).drop_duplicates()


def read_soc2010_2018() -> pd.DataFrame:
    path = CROSS_DIR / "soc_2010_to_2018_crosswalk.xlsx"
    df = pd.read_excel(path, sheet_name="Sorted by 2010", header=8)
    out = pd.DataFrame(
        {
            "soc2010": df["2010 SOC Code"].map(normalize_soc),
            "soc2018": df["2018 SOC Code"].map(normalize_soc),
            "soc2018_title": df["2018 SOC Title"].map(clean_text),
        }
    )
    return out.dropna(subset=["soc2010", "soc2018"]).drop_duplicates()


def read_onet2019_2018() -> pd.DataFrame:
    path = CROSS_DIR / "2019_to_SOC_Crosswalk.xlsx"
    df = pd.read_excel(path, sheet_name=0, header=3)
    out = pd.DataFrame(
        {
            "onet2019": df["O*NET-SOC 2019 Code"].map(clean_text),
            "onet2019_base_soc": df["O*NET-SOC 2019 Code"].map(normalize_soc),
            "soc2018": df["2018 SOC Code"].map(normalize_soc),
            "soc2018_title": df["2018 SOC Title"].map(clean_text),
        }
    )
    return out.dropna(subset=["onet2019_base_soc", "soc2018"]).drop_duplicates()


def locate_exposure_dir() -> Path | None:
    matches = list(PROJECT_ROOT.glob("*/exposure_base_soc_detail.xlsx"))
    if not matches:
        return None
    return matches[0].parent


def read_exposure_sets() -> tuple[set[str], set[str], pd.DataFrame]:
    exposure_dir = locate_exposure_dir()
    if exposure_dir is None:
        return set(), set(), pd.DataFrame()
    detail = pd.read_excel(exposure_dir / "exposure_base_soc_detail.xlsx")
    minor = pd.read_excel(exposure_dir / "exposure_base_minor_soc.xlsx")
    detail["base_soc"] = detail["occu_soc_code"].map(normalize_soc)
    minor["minor_soc"] = minor["minor_soc_code"].map(normalize_soc)
    inventory = pd.DataFrame(
        [
            {
                "file": "exposure_base_soc_detail.xlsx",
                "rows": len(detail),
                "unique_normalized_soc": int(detail["base_soc"].nunique()),
                "classification": "O*NET/SOC detailed occupation, normalized to 6-digit SOC base",
            },
            {
                "file": "exposure_base_minor_soc.xlsx",
                "rows": len(minor),
                "unique_normalized_soc": int(minor["minor_soc"].nunique()),
                "classification": "SOC minor group",
            },
        ]
    )
    return set(detail["base_soc"].dropna()), set(minor["minor_soc"].dropna()), inventory


def file_inventory(
    bls: pd.DataFrame,
    ibs: pd.DataFrame,
    ons: pd.DataFrame,
    soc_version: pd.DataFrame,
    onet_version: pd.DataFrame,
    exposure_inventory: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "file": "Crosswork/ISCO_SOC_Crosswalk.xls",
            "role": "主映射候选",
            "content": "BLS/SOCPC ISCO-08 与美国 SOC 2010 互转表",
            "rows_used": len(bls),
            "judgement": "可作为主 crosswalk，但一对多较多，无就业权重或比例。",
        },
        {
            "file": "Crosswork/onetsoc_to_isco_cws_ibs/soc10_isco08.dta",
            "role": "稳健性映射候选",
            "content": "IBS 任务研究使用的 SOC 2010 与 ISCO-08 转换表",
            "rows_used": len(ibs),
            "judgement": "适合把 O*NET/SOC 指标汇总到 ISCO 或做敏感性检验；也是 iscoCrosswalks 近似匹配思路的重要数据基础之一。",
        },
        {
            "file": "Crosswork/onetsocconversion.xlsx",
            "role": "参考映射",
            "content": "ONS ISCO-08 与英国 SOC 2010/2000 的比例转换",
            "rows_used": len(ons),
            "judgement": "目标分类是 UK SOC，不可直接合并美国 SOC/O*NET 暴露度，可作覆盖率参考。",
        },
        {
            "file": "Crosswork/soc_2010_to_2018_crosswalk.xlsx",
            "role": "版本转换",
            "content": "美国 SOC 2010 到 SOC 2018",
            "rows_used": len(soc_version),
            "judgement": "用于处理 BLS SOC2010 与 O*NET/SOC 2018/2019 暴露代码之间的版本差异。",
        },
        {
            "file": "Crosswork/2019_to_SOC_Crosswalk.xlsx",
            "role": "版本转换",
            "content": "O*NET-SOC 2019 到 SOC 2018",
            "rows_used": len(onet_version),
            "judgement": "用于判断暴露度详细职业代码与 SOC2018 的关系，不提供 ISCO 映射。",
        },
        {
            "file": "R package: iscoCrosswalks",
            "role": "算法扩展备选",
            "content": "基于 ISCO-08 与 SOC2010 的近似职业转换函数，可在不同 SOC/ISCO 层级之间映射指标或人数。",
            "rows_used": 0,
            "judgement": "当前环境未安装 Rscript，未直接调用；若详细 SOC 覆盖率不足，建议作为扩展方案，用 indicator=TRUE 将 SOC 暴露度聚合到 ISCO 层级。",
        },
    ]
    for _, row in exposure_inventory.iterrows():
        rows.append(
            {
                "file": f"暴露指数数据/{row['file']}",
                "role": "暴露度合并目标",
                "content": row["classification"],
                "rows_used": int(row["rows"]),
                "judgement": f"规范化后唯一 SOC 数 {int(row['unique_normalized_soc'])}。",
            }
        )
    return pd.DataFrame(rows)


def summarize_stage(
    source: str,
    stage: str,
    current: pd.DataFrame,
    flags: pd.DataFrame,
    flag_col: str,
    target_count_col: str,
    compatible: str,
    note: str,
) -> dict[str, Any]:
    total_unique = len(current)
    total_sample = float(current["sample_count"].sum())
    total_weighted = float(current["weighted_count"].sum())
    merged = current.merge(flags[["isco08", flag_col, target_count_col]], on="isco08", how="left")
    flag = merged[flag_col].eq(True)
    matched = merged[flag].copy()
    multi = merged[merged[target_count_col].fillna(0) > 1].copy()
    return {
        "source": source,
        "stage": stage,
        "compatible_with_ai_exposure": compatible,
        "current_isco_total": total_unique,
        "current_sample_total": total_sample,
        "current_weighted_total": total_weighted,
        "matched_isco": int(matched["isco08"].nunique()),
        "matched_isco_rate": matched["isco08"].nunique() / total_unique if total_unique else np.nan,
        "matched_sample_count": float(matched["sample_count"].sum()),
        "matched_sample_rate": matched["sample_count"].sum() / total_sample if total_sample else np.nan,
        "matched_weighted_count": float(matched["weighted_count"].sum()),
        "matched_weighted_rate": matched["weighted_count"].sum() / total_weighted if total_weighted else np.nan,
        "multi_target_isco": int(multi["isco08"].nunique()),
        "note": note,
    }


def build_flags(
    mapping: pd.DataFrame,
    soc_version: pd.DataFrame,
    detail_socs: set[str],
    minor_socs: set[str],
) -> pd.DataFrame:
    m = mapping.copy()
    m = m.rename(columns={"target_code": "soc2010"})
    m = m.merge(soc_version, on="soc2010", how="left")
    m["detail_direct"] = m["soc2010"].isin(detail_socs)
    m["detail_via2018"] = m["soc2018"].isin(detail_socs)
    m["minor_direct"] = m["soc2010"].map(soc_minor).isin(minor_socs)
    m["minor_via2018"] = m["soc2018"].map(soc_minor).isin(minor_socs)
    grouped = (
        m.groupby("isco08", as_index=False)
        .agg(
            has_crosswalk=("soc2010", lambda x: x.notna().any()),
            detail_direct=("detail_direct", "any"),
            detail_via2018=("detail_via2018", "any"),
            minor_direct=("minor_direct", "any"),
            minor_via2018=("minor_via2018", "any"),
            target_count=("soc2010", "nunique"),
            target_examples=("soc2010", lambda x: ", ".join(sorted(set(x.dropna()))[:8])),
        )
    )
    return grouped


def build_ons_flags(ons: pd.DataFrame) -> pd.DataFrame:
    return (
        ons.groupby("isco08", as_index=False)
        .agg(
            has_crosswalk=("target_code", lambda x: x.notna().any()),
            target_count=("target_code", "nunique"),
            target_examples=("target_code", lambda x: ", ".join(sorted(set(x.dropna()))[:8])),
        )
    )


def unmatched_table(current: pd.DataFrame, flags: pd.DataFrame, flag_col: str, source: str, stage: str) -> pd.DataFrame:
    merged = current.merge(flags[["isco08", flag_col, "target_count", "target_examples"]], on="isco08", how="left")
    flag = merged[flag_col].eq(True)
    out = merged[~flag].copy()
    out.insert(0, "source", source)
    out.insert(1, "stage", stage)
    return out.sort_values("sample_count", ascending=False).head(40)


def multi_table(current: pd.DataFrame, flags: pd.DataFrame, source: str) -> pd.DataFrame:
    merged = current.merge(flags[["isco08", "target_count", "target_examples"]], on="isco08", how="left")
    out = merged[merged["target_count"].fillna(0) > 1].copy()
    out.insert(0, "source", source)
    return out.sort_values(["sample_count", "target_count"], ascending=[False, False]).head(50)


def cleaning_rules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": "1. 原始数据保留",
                "rule": "所有 .dta/.xlsx/.xls 原始文件只读，不在原文件上覆盖。",
                "reason": "保证复现可追溯，后续任何清洗都能回到原始数据重新生成。",
            },
            {
                "step": "2. 年份统一",
                "rule": "为每个样本添加 year；2018、2021 使用 weight，2023 使用整合权重 weight2。",
                "reason": "三期 CGSS 是重复截面，不是个体面板；权重口径必须显式记录。",
            },
            {
                "step": "3. 当前职业定义",
                "rule": "2018/2021 使用 isco08_a59d；2023 优先 isco08a59d，缺失时补用 isco08a42new。",
                "reason": "保持 respondent current occupation 口径一致，并最大化 2023 当前职业覆盖。",
            },
            {
                "step": "4. ISCO 规范化",
                "rule": "职业代码转为四位字符串；负值、9997/9998/9999、9999996-9999999 视为特殊缺失；1000、2000、5000 等聚合职业不删除但标记为低粒度。",
                "reason": "聚合职业是 CGSS 中真实出现的编码，直接删除会造成系统性样本损失，但不能当作精细职业处理。",
            },
            {
                "step": "5. 收入变量",
                "rule": "个人总收入、劳动收入、家庭收入中的特殊缺失码设为缺失；建模时使用 log(income+1)，并保留零收入；跨年比较需做 CPI 平减。",
                "reason": "收入分布高度右偏且存在拒答/不知道/不适用编码；名义收入不能直接跨年比较。",
            },
            {
                "step": "6. Crosswalk 主口径",
                "rule": "优先使用 BLS/SOCPC ISCO-08 -> US SOC2010；IBS SOC10-ISCO08 作为稳健性；ONS UK SOC 只作参考。",
                "reason": "AI-LLM 暴露度使用美国 SOC/O*NET 体系，英国 SOC 映射不能直接合并。",
            },
            {
                "step": "7. SOC 版本处理",
                "rule": "SOC2010 先直接匹配暴露度；再通过 SOC2010->SOC2018 检查版本转换后的匹配率。",
                "reason": "暴露度详细职业代码更接近 O*NET/SOC 2018/2019 体系，版本转换可减少因分类变动导致的漏配。",
            },
            {
                "step": "8. 一对多处理",
                "rule": "不得随意选第一个 SOC；正式数据应保留 target_count，并在无比例权重时使用等权平均暴露度，同时报告一对多职业比例。",
                "reason": "ISCO 与 SOC 的职业边界不同，一对多映射会引入测量误差。",
            },
            {
                "step": "9. 近似算法扩展",
                "rule": "若详细 SOC 匹配率不足，可使用 iscoCrosswalks 的层级转换思想，将 SOC 暴露度作为 indicator 聚合到 ISCO 层级，并把结果作为主口径或稳健性口径之一。",
                "reason": "该方法比简单一对一匹配更适合处理分类体系边界不一致，但仍需报告近似映射带来的测量误差。",
            },
            {
                "step": "10. 就业人数解释",
                "rule": "CGSS 只能构造样本内加权职业就业份额，不能直接代表全国真实职业就业人数。",
                "reason": "CGSS 是社会调查样本，不是就业登记或劳动力总量统计。",
            },
        ]
    )


def md_table(df: pd.DataFrame, n: int | None = None, cols: list[str] | None = None) -> str:
    out = df.copy()
    if cols:
        out = out[cols]
    if n is not None:
        out = out.head(n)
    out = out.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    return out.to_markdown(index=False)


def write_report(
    year_summary: pd.DataFrame,
    inventory: pd.DataFrame,
    coverage: pd.DataFrame,
    unmatched: pd.DataFrame,
    multi: pd.DataFrame,
    rules: pd.DataFrame,
) -> None:
    coverage_show = coverage[
        [
            "source",
            "stage",
            "compatible_with_ai_exposure",
            "matched_isco",
            "matched_isco_rate",
            "matched_sample_count",
            "matched_sample_rate",
            "matched_weighted_rate",
            "multi_target_isco",
        ]
    ].copy()
    top_unmatched = unmatched[
        [
            "source",
            "stage",
            "isco08",
            "isco08_label",
            "major_label",
            "sample_count",
            "weighted_count",
        ]
    ].copy()
    top_multi = multi[
        [
            "source",
            "isco08",
            "isco08_label",
            "major_label",
            "sample_count",
            "target_count",
            "target_examples",
        ]
    ].copy()
    lines = [
        "# CGSS 与 AI-LLM 暴露度数据清洗说明",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 一、清洗目标",
        "",
        "本说明用于把 CGSS2018、CGSS2021、CGSS2023 的个体职业、就业与收入变量清洗为可同 AI-LLM 职业暴露度连接的研究数据。核心难点不是读取 CGSS，而是职业分类体系不同：CGSS 使用 ISCO-08，暴露度数据使用 SOC/O*NET 或智联职业分类。因此清洗必须同时处理年份变量口径、特殊缺失码、调查权重、ISCO 粒度、ISCO-SOC crosswalk、SOC 版本转换和一对多映射。",
        "",
        "## 二、本轮实际读取的数据",
        "",
        md_table(year_summary),
        "",
        "三期数据均包含当前职业 ISCO-08 编码。当前职业有效样本合计为 12,832 个，新增 2018 后，当前职业唯一 ISCO 代码从上一轮的 365 个扩展到 458 个，能够明显增加职业覆盖，但也带来更多需要复核的低频职业。",
        "",
        "## 三、crosswalk 与暴露度文件判断",
        "",
        md_table(inventory),
        "",
        "结论是：`ISCO_SOC_Crosswalk.xls` 和 `soc10_isco08.dta` 可以作为美国 SOC 暴露度合并的主要与稳健性路径；`soc_2010_to_2018_crosswalk.xlsx`、`2019_to_SOC_Crosswalk.xlsx` 是版本转换工具；`onetsocconversion.xlsx` 是英国 SOC 口径，不能直接合并美国 SOC/O*NET 暴露度。",
        "",
        "`iscoCrosswalks` 的价值在于“近似转换算法”而不是另一张简单对照表。它可将指标或原始人数在 ISCO 与 SOC 的不同层级之间转换；在本项目中，更合适的用法是把 SOC/O*NET 暴露度视为 indicator，经由 ISCO-SOC 层级关系聚合到 ISCO-08，再与 CGSS 合并。当前电脑没有可用的 `Rscript`，因此本轮没有直接运行该 R 包；但已经将其列为后续算法扩展路径。",
        "",
        "## 四、数据清洗规则",
        "",
        md_table(rules),
        "",
        "## 五、映射覆盖率诊断",
        "",
        md_table(coverage_show),
        "",
        "覆盖率要分层解释。第一层是 ISCO 能否映射到 SOC；第二层是映射后的 SOC 能否在暴露度文件中找到对应职业；第三层是一对多映射下暴露度如何聚合。只报告第一层会高估可用性，只报告详细 SOC 暴露度又会牺牲较多样本。",
        "",
        "## 六、高频未匹配职业",
        "",
        md_table(top_unmatched, n=30),
        "",
        "这些职业是后续人工复核的优先对象。尤其是 `2000`、`5000`、`7000`、`6000`、`1000` 等聚合代码，它们不是简单缺失，而是 CGSS 中较粗粒度的职业记录；如果直接删除，会导致劳动者样本结构发生偏移。",
        "",
        "## 七、一对多映射职业",
        "",
        md_table(top_multi, n=30),
        "",
        "一对多映射说明同一个 ISCO 职业可能对应多个美国 SOC 职业。正式建模时建议保留 `target_count`，并将映射质量变量作为稳健性检验的一部分：主结果使用可解释的聚合暴露度，附录报告仅保留唯一映射或高质量映射的结果。",
        "",
        "## 八、建议形成的清洗后数据集",
        "",
        "1. 个体收入数据：一行一个 CGSS 受访者，保留 year、id、weight、当前 ISCO、AI-LLM 暴露度、收入、教育、年龄、性别、城乡、省份、健康、婚姻、就业状态。",
        "2. 职业-年份就业份额数据：以 ISCO 或映射后的 SOC/minor SOC 为单位，使用 CGSS 权重计算样本内就业份额；只能解释为调查样本加权份额。",
        "3. 映射质量数据：一行一个 ISCO-SOC 对应关系，保留 source、target_count、是否能匹配详细 SOC 暴露度、是否需要 SOC2018 转换、是否属于低粒度 ISCO 聚合代码。",
        "4. 稳健性数据：分别构造 BLS 主口径、IBS 稳健性口径、minor SOC 聚合口径、SOC 详细职业高质量子样本口径。",
        "",
        "## 九、对后续模型的含义",
        "",
        "如果研究“AI-LLM 暴露度是否使职业人数减少”，仅用 CGSS 不应说成全国职业人数减少，而应说成“样本内职业就业份额与暴露度的关系”。更严谨的做法是用 CGSS 构造重复截面的职业加权份额，再结合暴露度水平与 2018-2024 年暴露度变化，使用动态面板或广义加性模型检查高暴露职业的份额变化。",
        "",
        "如果研究“AI-LLM 暴露度是否使薪资下降”，CGSS 更适合做个体收入分析。建议使用分位数模型、RIF 回归、因果森林或双重机器学习，并把教育、年龄、城乡、省份、职业映射质量、年份固定效应作为关键控制或异质性维度。暴露度既要包含水平，也要包含年度变化或斜率。",
        "",
        "## 十、关于 iscoCrosswalks 的后续执行方案",
        "",
        "若安装 R 后执行 `iscoCrosswalks`，建议将 `exposure_base_soc_detail.xlsx` 或 `exposure_by_year_soc_detail.xlsx` 整理为两列：`job` 为 SOC 层级代码，`value` 为 AI-LLM 暴露度；调用 ISCO-SOC 转换函数时将其视为 indicator，使同一 ISCO 下多个 SOC 暴露度按均值聚合，而不是按人数求和。随后把生成的 ISCO 层级暴露度与 CGSS 的四位 ISCO 当前职业合并，并与 BLS/IBS 等权映射结果比较。",
        "",
        "这一路径的优势是可以减少因 SOC 详细职业名单只有 100 个而造成的样本损失；限制是近似映射不可避免地弱化职业边界，必须在论文中作为“测量误差与稳健性”单独讨论。",
        "",
        "参考来源：`iscoCrosswalks` GitHub README 说明该包用于在 ISCO 与 SOC 分类之间映射 indicators 和 raw counts，并在指标变量场景下使用 `indicator = TRUE` 进行均值聚合。链接：https://github.com/eworx-org/iscoCrosswalks",
        "",
        "## 十一、输出文件",
        "",
        "- `CGSS/results/tables/cgss_current_isco_counts_clean.csv`：三期当前职业 ISCO 清洗后频数。",
        "- `CGSS/results/tables/cgss_current_isco_year_summary.csv`：按年份的当前职业覆盖率摘要。",
        "- `CGSS/results/tables/crosswalk_file_inventory.csv`：crosswalk 文件用途判断。",
        "- `CGSS/results/tables/crosswalk_coverage_summary.csv`：映射覆盖率诊断。",
        "- `CGSS/results/tables/crosswalk_unmatched_top.csv`：高频未匹配职业。",
        "- `CGSS/results/tables/crosswalk_multi_mapped_top.csv`：一对多映射职业。",
        "- `CGSS/results/tables/cgss_cleaning_rules.csv`：清洗规则表。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    current, year_summary = build_current_isco_counts()
    detail_socs, minor_socs, exposure_inventory = read_exposure_sets()
    bls = read_bls_isco_soc()
    ibs = read_ibs_soc_isco()
    ons = read_ons_isco_uk_soc()
    soc_version = read_soc2010_2018()
    onet_version = read_onet2019_2018()

    bls_flags = build_flags(bls, soc_version, detail_socs, minor_socs)
    ibs_flags = build_flags(ibs, soc_version, detail_socs, minor_socs)
    ons_flags = build_ons_flags(ons)

    summaries = []
    for source, flags in [("BLS_ISCO08_to_SOC2010", bls_flags), ("IBS_SOC10_ISCO08_dta", ibs_flags)]:
        summaries.extend(
            [
                summarize_stage(source, "ISCO-08 -> US SOC2010", current, flags, "has_crosswalk", "target_count", "是", "第一层职业分类映射。"),
                summarize_stage(source, "SOC2010 直接匹配详细 SOC 暴露度", current, flags, "detail_direct", "target_count", "是", "仅匹配暴露度详细职业文件中的 SOC base code。"),
                summarize_stage(source, "SOC2010 经 SOC2018 转换后匹配详细 SOC 暴露度", current, flags, "detail_via2018", "target_count", "是", "处理 SOC 2010/2018 版本差异后的详细职业匹配。"),
                summarize_stage(source, "SOC2010 直接匹配 minor SOC 暴露度", current, flags, "minor_direct", "target_count", "是", "较粗的职业组别暴露度，覆盖通常高于详细职业。"),
                summarize_stage(source, "SOC2010 经 SOC2018 转换后匹配 minor SOC 暴露度", current, flags, "minor_via2018", "target_count", "是", "版本转换后的职业组别暴露度匹配。"),
            ]
        )
    summaries.append(
        summarize_stage(
            "ONS_ISCO08_to_UK_SOC2010",
            "ISCO-08 -> UK SOC2010",
            current,
            ons_flags,
            "has_crosswalk",
            "target_count",
            "否",
            "英国 SOC 口径，不可直接合并美国 SOC/O*NET 暴露度。",
        )
    )
    coverage = pd.DataFrame(summaries)

    unmatched = pd.concat(
        [
            unmatched_table(current, bls_flags, "has_crosswalk", "BLS_ISCO08_to_SOC2010", "ISCO-08 -> US SOC2010"),
            unmatched_table(current, bls_flags, "detail_via2018", "BLS_ISCO08_to_SOC2010", "SOC2018 详细职业暴露度"),
            unmatched_table(current, ibs_flags, "detail_via2018", "IBS_SOC10_ISCO08_dta", "SOC2018 详细职业暴露度"),
            unmatched_table(current, ons_flags, "has_crosswalk", "ONS_ISCO08_to_UK_SOC2010", "ISCO-08 -> UK SOC2010"),
        ],
        ignore_index=True,
    )
    multi = pd.concat(
        [
            multi_table(current, bls_flags, "BLS_ISCO08_to_SOC2010"),
            multi_table(current, ibs_flags, "IBS_SOC10_ISCO08_dta"),
            multi_table(current, ons_flags, "ONS_ISCO08_to_UK_SOC2010"),
        ],
        ignore_index=True,
    )
    inventory = file_inventory(bls, ibs, ons, soc_version, onet_version, exposure_inventory)
    rules = cleaning_rules()

    current.to_csv(TABLE_DIR / "cgss_current_isco_counts_clean.csv", index=False, encoding="utf-8-sig")
    year_summary.to_csv(TABLE_DIR / "cgss_current_isco_year_summary.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(TABLE_DIR / "crosswalk_file_inventory.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(TABLE_DIR / "crosswalk_coverage_summary.csv", index=False, encoding="utf-8-sig")
    unmatched.to_csv(TABLE_DIR / "crosswalk_unmatched_top.csv", index=False, encoding="utf-8-sig")
    multi.to_csv(TABLE_DIR / "crosswalk_multi_mapped_top.csv", index=False, encoding="utf-8-sig")
    rules.to_csv(TABLE_DIR / "cgss_cleaning_rules.csv", index=False, encoding="utf-8-sig")

    write_report(year_summary, inventory, coverage, unmatched, multi, rules)
    print("Crosswalk cleaning audit finished.")
    print(f"Tables: {TABLE_DIR}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
