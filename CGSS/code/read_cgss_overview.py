"""CGSS 数据初步读取与 AI-LLM 暴露度研究可行性摸底。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
TABLE_DIR = RESULT_DIR / "tables"
REPORT_PATH = ROOT / "cgss_initial_report.md"


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
    respondent_current_isco: list[str]
    respondent_ever_isco: str | None
    spouse_isco: str | None
    father_isco: str | None
    mother_isco: str | None
    weight: str | None
    gender: str | None
    birth_year: str | None
    education: str | None
    total_income: str | None
    labor_income: str | None
    household_income: str | None
    health: str | None
    work_status: str | None
    current_work_status: str | None
    marriage: str | None
    subjective_status: str | None = None
    status_change: str | None = None


SPECS = [
    YearSpec(
        year=2018,
        file_name="CGSS2018.dta",
        respondent_current_isco=["isco08_a59d"],
        respondent_ever_isco="isco08_a60d",
        spouse_isco="isco08_sp",
        father_isco="isco08_f",
        mother_isco="isco08_m",
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
        respondent_current_isco=["isco08_a59d"],
        respondent_ever_isco="isco08_a60d",
        spouse_isco="isco08_sp",
        father_isco="isco08_f",
        mother_isco="isco08_m",
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
        respondent_current_isco=["isco08a59d", "isco08a42new"],
        respondent_ever_isco="isco08a60d",
        spouse_isco="isco08sp",
        father_isco="isco08f",
        mother_isco="isco08m",
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


CORE_ROLES = {
    "gender": "性别",
    "birth_year": "出生年份",
    "education": "最高教育程度",
    "total_income": "个人全年总收入",
    "labor_income": "个人全年职业/劳动收入",
    "household_income": "家庭全年总收入",
    "health": "主观健康",
    "work_status": "工作经历及状况",
    "current_work_status": "当前工作状况",
    "marriage": "婚姻状况",
    "subjective_status": "主观社会经济地位",
    "status_change": "相对三年前社会经济地位变化",
    "weight": "调查权重",
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


def setup() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).replace("\xa0", " ").replace("\n", " ").strip()


def read_stata_with_labels(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    with pd.read_stata(path, iterator=True, convert_categoricals=False) as reader:
        labels = {k: clean_text(v) for k, v in reader.variable_labels().items()}
    df = pd.read_stata(path, convert_categoricals=False, preserve_dtypes=False)
    return df, labels


def numeric_series(df: pd.DataFrame, var: str | None) -> pd.Series:
    if var is None or var not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[var], errors="coerce")


def is_valid_isco(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
        code = int(float(value))
    except (TypeError, ValueError):
        return False
    return code not in SPECIAL_MISSING and 0 <= code < 9800


def normalize_isco(value: Any) -> int | float:
    if not is_valid_isco(value):
        return np.nan
    return int(float(value))


def isco_major(value: Any) -> str:
    if not is_valid_isco(value):
        return ""
    code = int(float(value))
    return str(code).zfill(4)[0]


def combine_current_isco(df: pd.DataFrame, variables: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for var in variables:
        if var not in df.columns:
            continue
        s = pd.to_numeric(df[var], errors="coerce").map(normalize_isco)
        out = out.where(out.notna(), s)
    return out


def describe_numeric(s: pd.Series) -> dict[str, float]:
    valid = pd.to_numeric(s, errors="coerce")
    valid = valid[valid.notna() & ~valid.isin(SPECIAL_MISSING)]
    if valid.empty:
        return {"mean": np.nan, "median": np.nan, "p25": np.nan, "p75": np.nan}
    return {
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "p25": float(valid.quantile(0.25)),
        "p75": float(valid.quantile(0.75)),
    }


def variable_inventory(df: pd.DataFrame, labels: dict[str, str], year: int) -> pd.DataFrame:
    rows = []
    n = len(df)
    for var in df.columns:
        non_missing = int(df[var].notna().sum())
        rows.append(
            {
                "year": year,
                "variable": var,
                "label": labels.get(var, ""),
                "dtype": str(df[var].dtype),
                "non_missing": non_missing,
                "missing": int(n - non_missing),
                "missing_rate": 1 - non_missing / n if n else np.nan,
                "n_unique": int(df[var].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def core_variable_summary(df: pd.DataFrame, labels: dict[str, str], spec: YearSpec) -> pd.DataFrame:
    rows = []
    n = len(df)
    for attr, cn_name in CORE_ROLES.items():
        var = getattr(spec, attr)
        if var is None or var not in df.columns:
            rows.append(
                {
                    "year": spec.year,
                    "role": attr,
                    "role_cn": cn_name,
                    "variable": var or "",
                    "label": "",
                    "non_missing": 0,
                    "missing_rate": 1.0,
                    "special_missing_count": np.nan,
                    "valid_non_special": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                }
            )
            continue
        s = numeric_series(df, var)
        non_missing = int(df[var].notna().sum())
        special_count = int(s.isin(SPECIAL_MISSING).sum())
        valid_count = int((s.notna() & ~s.isin(SPECIAL_MISSING)).sum())
        stats = describe_numeric(s)
        rows.append(
            {
                "year": spec.year,
                "role": attr,
                "role_cn": cn_name,
                "variable": var,
                "label": labels.get(var, ""),
                "non_missing": non_missing,
                "missing_rate": 1 - non_missing / n if n else np.nan,
                "special_missing_count": special_count,
                "valid_non_special": valid_count,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def isco_coverage(df: pd.DataFrame, labels: dict[str, str], spec: YearSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    role_to_vars = {
        "respondent_current_isco": spec.respondent_current_isco,
        "respondent_ever_nonfarm_isco": [spec.respondent_ever_isco],
        "spouse_isco": [spec.spouse_isco],
        "father_isco": [spec.father_isco],
        "mother_isco": [spec.mother_isco],
    }
    rows = []
    code_rows = []
    major_rows = []
    n = len(df)
    for role, variables in role_to_vars.items():
        variables = [v for v in variables if v and v in df.columns]
        if not variables:
            continue
        if role == "respondent_current_isco":
            s = combine_current_isco(df, variables)
            label = " + ".join(labels.get(v, v) for v in variables)
            variable = " / ".join(variables)
        else:
            variable = variables[0]
            s = numeric_series(df, variable).map(normalize_isco)
            label = labels.get(variable, "")
        valid = s.dropna().astype(int)
        rows.append(
            {
                "year": spec.year,
                "isco_role": role,
                "variable": variable,
                "label": label,
                "n_obs": n,
                "valid_isco_count": int(valid.size),
                "valid_isco_rate": valid.size / n if n else np.nan,
                "unique_isco_codes": int(valid.nunique()),
            }
        )
        counts = valid.value_counts().rename_axis("isco08_code").reset_index(name="count")
        counts["year"] = spec.year
        counts["isco_role"] = role
        code_rows.append(counts)
        majors = valid.map(lambda x: str(int(x)).zfill(4)[0])
        major_count = majors.value_counts().rename_axis("isco_major").reset_index(name="count")
        major_count["year"] = spec.year
        major_count["isco_role"] = role
        major_count["major_label"] = major_count["isco_major"].map(ISCO_MAJOR_LABELS)
        major_rows.append(major_count)
    return pd.DataFrame(rows), pd.concat(code_rows, ignore_index=True), pd.concat(major_rows, ignore_index=True)


def read_isco_labels() -> pd.DataFrame:
    files = list(ROOT.glob("*编码表.xlsx"))
    if not files:
        return pd.DataFrame(columns=["isco08_code", "isco08_label"])
    xlsx = files[0]
    xl = pd.ExcelFile(xlsx)
    sheet = next((s for s in xl.sheet_names if "isco" in s.lower()), None)
    if sheet is None:
        return pd.DataFrame(columns=["isco08_code", "isco08_label"])
    labels = pd.read_excel(xlsx, sheet_name=sheet)
    labels = labels.rename(columns={labels.columns[0]: "isco08_code", labels.columns[1]: "isco08_label"})
    labels["isco08_code"] = pd.to_numeric(labels["isco08_code"], errors="coerce")
    labels = labels.dropna(subset=["isco08_code"]).copy()
    labels["isco08_code"] = labels["isco08_code"].astype(int)
    labels["isco08_label"] = labels["isco08_label"].map(clean_text)
    return labels


def exposure_inventory() -> pd.DataFrame:
    project_root = ROOT.parent
    exposure_dirs = [p for p in project_root.iterdir() if p.is_dir() and "暴露指数数据" in p.name]
    if not exposure_dirs:
        return pd.DataFrame()
    exposure_dir = exposure_dirs[0]
    rows = []
    for file in sorted(exposure_dir.glob("*.xlsx")):
        try:
            sample = pd.read_excel(file, nrows=3)
            rows.append(
                {
                    "file": file.name,
                    "columns": ", ".join(map(str, sample.columns)),
                    "rows_read_sample": len(sample),
                    "suggested_merge_level": suggest_exposure_level(file.name, sample.columns),
                }
            )
        except Exception as exc:
            rows.append({"file": file.name, "columns": "", "rows_read_sample": 0, "suggested_merge_level": f"读取失败: {exc}"})
    return pd.DataFrame(rows)


def suggest_exposure_level(file_name: str, columns: pd.Index) -> str:
    cols = set(map(str, columns))
    if "occu_soc_code" in cols:
        return "SOC 详细职业；需要 ISCO-08 到 SOC/O*NET crosswalk"
    if "minor_soc_code" in cols:
        return "SOC 职业组别；需要 ISCO-08 到 SOC 大类或职业组 crosswalk"
    if {"jd_class1", "jd_class2"}.issubset(cols):
        return "智联二级职业；CGSS 无直接智联分类，需要文本/职业名称映射"
    return "待判断"


def build_research_notes() -> list[dict[str, str]]:
    return [
        {
            "topic": "职业 AI-LLM 暴露度与个人收入",
            "question": "高 AI-LLM 暴露职业中的劳动者，其个人总收入或职业/劳动收入是否更高或更低？",
            "model": "加权 OLS / 分位数回归；因变量为 log(个人劳动收入+1) 或 log(个人总收入+1)。",
            "key_variables": "AI-LLM 暴露度、个人收入、教育、年龄、性别、城乡、省份、调查年份、权重。",
            "caution": "横截面关联不能解释为因果效应；当前职业只在就业者中观测，存在样本选择。",
        },
        {
            "topic": "教育对 AI-LLM 暴露的调节作用",
            "question": "教育程度是否改变 AI-LLM 暴露度与收入/就业状态之间的关系？",
            "model": "交互项模型：Outcome = Exposure + Education + Exposure×Education + Controls。",
            "key_variables": "AI-LLM 暴露度、最高教育程度、收入、就业状态、人口学控制变量。",
            "caution": "教育既可能提高进入高暴露职业的概率，也可能提高适应技术变化的能力，需要区分选择效应与缓冲效应。",
        },
        {
            "topic": "AI-LLM 暴露度与就业状态",
            "question": "高暴露职业劳动者是否在就业状态、非农就业、工作经历上呈现不同特征？",
            "model": "Logit / 多项 Logit；因变量为是否有当前工作、工作经历类型或当前工作状态。",
            "key_variables": "工作经历及状况、当前工作状况、当前职业 ISCO、AI-LLM 暴露度。",
            "caution": "未就业者可能缺少当前职业编码，可考虑使用曾经职业或限制在就业样本。",
        },
        {
            "topic": "职业暴露度的代际传递",
            "question": "父母职业的 AI-LLM 暴露度是否与子代当前职业暴露度、收入或教育相关？",
            "model": "代际相关模型 / mobility transition：子代暴露度 = 父母暴露度 + 控制变量。",
            "key_variables": "被访者当前职业、父亲职业、母亲职业、教育、收入、城乡、省份。",
            "caution": "父母职业为回溯变量，且职业编码缺失较多；适合做结构性关联分析。",
        },
        {
            "topic": "夫妻职业暴露匹配",
            "question": "夫妻双方职业的 AI-LLM 暴露度是否存在同质匹配？高暴露家庭收入是否更集中？",
            "model": "夫妻暴露度相关、家庭收入回归、分组比较。",
            "key_variables": "被访者当前职业、配偶职业、个人收入、配偶收入、家庭收入。",
            "caution": "仅适用于有配偶且配偶职业编码有效的样本。",
        },
        {
            "topic": "地区与城乡差异",
            "question": "不同省份、城乡劳动者的职业 AI-LLM 暴露度分布是否不同？",
            "model": "描述统计、加权均值比较、地区固定效应回归。",
            "key_variables": "省市县编码、城乡、职业暴露度、收入和教育。",
            "caution": "CGSS 不是以职业-地区市场份额为核心设计，地区细分后样本量需要检查。",
        },
        {
            "topic": "主观社会经济地位与 AI-LLM 暴露",
            "question": "高暴露职业劳动者是否拥有更高的主观社会经济地位，或是否感到地位改善/下降？",
            "model": "有序 Logit / OLS；2023 年可用 b1、b2。",
            "key_variables": "AI-LLM 暴露度、主观社会经济地位、地位变化、收入、教育。",
            "caution": "该方向 2023 数据更完整；2021 对应变量需进一步核对。",
        },
    ]


def build_data_needs() -> list[dict[str, str]]:
    return [
        {
            "priority": "必须",
            "data": "ISCO-08 到 SOC/O*NET 的职业 crosswalk",
            "why_needed": "CGSS 使用 ISCO-08，AI-LLM 暴露指数使用 SOC/O*NET；没有 crosswalk 就不能把个体职业与暴露度严谨合并。",
            "suggested_source": "BLS ISCO-08 x SOC 2010 crosswalk；O*NET Resource Center crosswalks；必要时人工复核高频职业。",
        },
        {
            "priority": "必须",
            "data": "各年份 CGSS 问卷、编码表、权重说明",
            "why_needed": "跨年份变量名和取值标签会变化，必须用问卷和编码表统一收入、教育、就业状态、职业编码和权重口径。",
            "suggested_source": "CNSDA/CGSS 官方发布页面；本地已有 2023 编码表，其他年份需补齐。",
        },
        {
            "priority": "强烈建议",
            "data": "更多 CGSS 年份：优先 2017、2015，之后再考虑 2013、2012、2011、2010",
            "why_needed": "2018 已经纳入当前工作底稿；更多年份可扩大职业样本、构造重复截面，并观察 AI-LLM 暴露度与收入/就业结构的长期关联。",
            "suggested_source": "CNSDA 高级搜索显示 CGSS 有 2023、2021、2018、2017、2015 等项目；具体下载需注册。",
        },
        {
            "priority": "强烈建议",
            "data": "CPI 或居民收入价格平减指数",
            "why_needed": "跨年份比较收入时必须把名义收入调整为同一价格基期，否则年份差异会混入通货膨胀因素。",
            "suggested_source": "国家统计局年度 CPI 或居民消费价格指数。",
        },
        {
            "priority": "可选增强",
            "data": "省级或城市层面数字经济、产业结构、失业率、平均工资等宏观变量",
            "why_needed": "若研究地区差异，可控制地区经济发展、产业结构和数字化程度，避免把地区结构差异误归因于职业暴露度。",
            "suggested_source": "中国统计年鉴、各省统计年鉴、城市统计年鉴或公开数字经济指数。",
        },
        {
            "priority": "可选增强",
            "data": "职业名称人工校验表",
            "why_needed": "ISCO 到 SOC 是多对多映射，部分高频职业需要人工确认匹配，避免因职业分类差异造成暴露度测量误差。",
            "suggested_source": "基于 `results/tables/current_isco_codes_to_map.csv` 建立人工复核表。",
        },
    ]


def write_report(
    dataset_inventory: pd.DataFrame,
    core_summary: pd.DataFrame,
    isco_cov: pd.DataFrame,
    current_codes: pd.DataFrame,
    major_dist: pd.DataFrame,
    exposure_inv: pd.DataFrame,
    research_notes: pd.DataFrame,
    data_needs: pd.DataFrame,
) -> None:
    def md_table(df: pd.DataFrame, n: int | None = None) -> str:
        if n is not None:
            df = df.head(n)
        return df.to_markdown(index=False)

    current_cov = isco_cov[isco_cov["isco_role"] == "respondent_current_isco"].copy()
    income_rows = core_summary[core_summary["role"].isin(["total_income", "labor_income", "household_income"])].copy()
    edu_rows = core_summary[core_summary["role"].isin(["education", "work_status", "current_work_status", "health"])].copy()
    current_major = major_dist[major_dist["isco_role"] == "respondent_current_isco"].copy()
    current_major = current_major.sort_values(["year", "isco_major"])

    lines = [
        "# CGSS 数据初步阅读报告",
        "",
        "## 1. 数据文件概况",
        "",
        "本报告由 `CGSS/code/read_cgss_overview.py` 自动读取 `CGSS2018.dta`、`CGSS2021.dta` 与 `CGSS2023.dta` 后生成，目的在于判断 CGSS 能否与 AI-LLM 暴露指数结合开展后续研究。",
        "",
        md_table(dataset_inventory),
        "",
        "## 2. 与 AI-LLM 暴露度连接的关键变量",
        "",
        "两期 CGSS 均包含 ISCO-08 职业编码，这是将个体数据与职业 AI-LLM 暴露指数连接的关键入口。需要注意的是，当前 AI-LLM 暴露指数使用 SOC/O*NET 或智联职业分类，而 CGSS 使用 ISCO-08，因此下一步必须构造或引入 `ISCO-08 -> SOC/O*NET` 的职业 crosswalk。",
        "",
        "### 当前职业编码覆盖率",
        "",
        md_table(current_cov[["year", "variable", "valid_isco_count", "valid_isco_rate", "unique_isco_codes"]].round(4)),
        "",
        "### 当前职业 ISCO 大类分布",
        "",
        md_table(current_major[["year", "isco_major", "major_label", "count"]]),
        "",
        "当前职业唯一代码清单已输出到 `results/tables/current_isco_codes_to_map.csv`，可作为后续职业映射的工作底稿。",
        "",
        "## 3. 核心结果变量与控制变量可用性",
        "",
        "### 收入变量",
        "",
        md_table(income_rows[["year", "role_cn", "variable", "valid_non_special", "mean", "median", "p25", "p75"]].round(2)),
        "",
        "收入变量已经剔除 CGSS 中的特殊编码（如 9999996=收入高于百万位数、9999997=不适用、9999998=不知道、9999999=拒绝回答）。后续正式回归中，建议使用 `log(收入+1)`，并报告缩尾或分位数回归作为稳健性检验。",
        "",
        "### 教育、工作与健康变量",
        "",
        md_table(edu_rows[["year", "role_cn", "variable", "valid_non_special", "label"]]),
        "",
        "这些变量说明，CGSS 不仅能提供职业编码，还能提供个人收入、劳动收入、家庭收入、教育、健康、工作状态、婚姻、城乡和省份等变量，适合做个体层面的劳动者异质性分析。",
        "",
        "## 4. AI-LLM 暴露指数文件对接情况",
        "",
        md_table(exposure_inv) if not exposure_inv.empty else "未在项目根目录下找到暴露指数数据文件夹。",
        "",
        "从对接角度看，最推荐的路径是：先将 CGSS 的 ISCO-08 职业编码映射到 SOC/O*NET 详细职业，再合并 `exposure_base_soc_detail.xlsx` 或年度暴露指数。若只能做探索性分析，也可以先将 ISCO-08 大类粗略映射到 SOC 大类，但这一口径较粗，不宜作为主结果。",
        "",
        "## 5. 可开展的研究方向",
        "",
        md_table(research_notes),
        "",
        "## 6. 还需要补充的数据",
        "",
        md_table(data_needs),
        "",
        "其中最关键的是 `ISCO-08 -> SOC/O*NET` crosswalk。没有这一步，CGSS 只能停留在职业大类描述，无法严谨合并 AI-LLM 暴露指数。更多 CGSS 年份不是绝对必要，但如果希望把研究从 2021/2023 两期横截面扩展为重复截面分析，建议优先补充 2018、2017、2015 三期。",
        "",
        "## 7. 初步判断",
        "",
        "CGSS 数据与 AI-LLM 暴露度结合是可行的，但前提是解决职业分类映射问题。最有价值的研究方向不是复现原文的招聘需求回归，而是从劳动者个体角度补充原文：不同 AI-LLM 暴露职业中的劳动者，在收入、教育、就业状态、主观社会经济地位、代际职业背景和地区分布上是否存在系统差异。",
        "",
        "在研究设计上，建议下一步优先完成三件事：",
        "",
        "1. 构造 `ISCO-08 -> SOC/O*NET` crosswalk，并评估匹配率。",
        "2. 将 CGSS2018、CGSS2021 与 CGSS2023 处理为统一变量口径的个体层面分析数据。",
        "3. 先做描述统计和加权回归，再考虑教育异质性、代际传递和地区差异等扩展模型。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    dataset_rows = []
    variable_tables = []
    core_tables = []
    isco_cov_tables = []
    code_tables = []
    major_tables = []

    for spec in SPECS:
        path = ROOT / spec.file_name
        if not path.exists():
            continue
        df, labels = read_stata_with_labels(path)
        dataset_rows.append(
            {
                "year": spec.year,
                "file": spec.file_name,
                "n_obs": len(df),
                "n_variables": df.shape[1],
                "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            }
        )
        inv = variable_inventory(df, labels, spec.year)
        inv.to_csv(TABLE_DIR / f"variable_inventory_{spec.year}.csv", index=False, encoding="utf-8-sig")
        variable_tables.append(inv)
        core_tables.append(core_variable_summary(df, labels, spec))
        cov, codes, majors = isco_coverage(df, labels, spec)
        isco_cov_tables.append(cov)
        code_tables.append(codes)
        major_tables.append(majors)

    dataset_inventory = pd.DataFrame(dataset_rows)
    variable_inventory_all = pd.concat(variable_tables, ignore_index=True)
    core_summary = pd.concat(core_tables, ignore_index=True)
    isco_cov = pd.concat(isco_cov_tables, ignore_index=True)
    isco_codes = pd.concat(code_tables, ignore_index=True)
    major_dist = pd.concat(major_tables, ignore_index=True)

    isco_labels = read_isco_labels()
    current_codes = (
        isco_codes[isco_codes["isco_role"] == "respondent_current_isco"]
        .groupby("isco08_code", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )
    current_codes = current_codes.merge(isco_labels, on="isco08_code", how="left")
    current_codes["isco_major"] = current_codes["isco08_code"].map(lambda x: str(int(x)).zfill(4)[0])
    current_codes["major_label"] = current_codes["isco_major"].map(ISCO_MAJOR_LABELS)

    exposure_inv = exposure_inventory()
    research_notes = pd.DataFrame(build_research_notes())
    data_needs = pd.DataFrame(build_data_needs())

    dataset_inventory.to_csv(TABLE_DIR / "dataset_inventory.csv", index=False, encoding="utf-8-sig")
    variable_inventory_all.to_csv(TABLE_DIR / "variable_inventory_all.csv", index=False, encoding="utf-8-sig")
    core_summary.to_csv(TABLE_DIR / "core_variable_summary.csv", index=False, encoding="utf-8-sig")
    isco_cov.to_csv(TABLE_DIR / "isco_coverage.csv", index=False, encoding="utf-8-sig")
    isco_codes.to_csv(TABLE_DIR / "isco_code_counts_long.csv", index=False, encoding="utf-8-sig")
    major_dist.to_csv(TABLE_DIR / "isco_major_distribution.csv", index=False, encoding="utf-8-sig")
    current_codes.to_csv(TABLE_DIR / "current_isco_codes_to_map.csv", index=False, encoding="utf-8-sig")
    exposure_inv.to_csv(TABLE_DIR / "ai_exposure_inventory_for_merge.csv", index=False, encoding="utf-8-sig")
    research_notes.to_csv(TABLE_DIR / "possible_research_designs.csv", index=False, encoding="utf-8-sig")
    data_needs.to_csv(TABLE_DIR / "additional_data_needs.csv", index=False, encoding="utf-8-sig")

    write_report(dataset_inventory, core_summary, isco_cov, current_codes, major_dist, exposure_inv, research_notes, data_needs)

    print("CGSS overview finished.")
    print(f"Tables: {TABLE_DIR}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
