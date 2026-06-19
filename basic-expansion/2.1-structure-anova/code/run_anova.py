"""2.1 职业结构分化：ANOVA 与方差分解。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
PART_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "人工智能-大语言模型技术”暴露指数数据"
TABLE_DIR = PART_DIR / "results" / "tables"
FIGURE_DIR = PART_DIR / "results" / "figures"


SOC_MAJOR_LABELS = {
    "11": "管理类",
    "13": "商业与金融类",
    "15": "计算机与数学类",
    "17": "建筑工程类",
    "19": "科学研究类",
    "21": "社区服务类",
    "23": "法律类",
    "25": "教育类",
    "27": "艺术传媒类",
    "29": "医疗技术类",
    "33": "保护服务类",
    "35": "餐饮服务类",
    "37": "清洁服务类",
    "39": "个人服务类",
    "41": "销售类",
    "43": "办公行政类",
    "45": "农业类",
    "47": "建筑施工类",
    "49": "安装维修类",
    "51": "生产制造类",
    "53": "运输搬运类",
}


def setup() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read_data() -> dict[str, pd.DataFrame]:
    return {
        "SOC职业组别": pd.read_excel(DATA_DIR / "exposure_base_minor_soc.xlsx"),
        "SOC详细职业": pd.read_excel(DATA_DIR / "exposure_base_soc_detail.xlsx"),
        "智联二级职业": pd.read_excel(DATA_DIR / "exposure_base_zl_occu.xlsx"),
    }


def add_groups(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    minor = data["SOC职业组别"].copy()
    minor["group_code"] = minor["minor_soc_code"].astype(str).str.slice(0, 2)
    minor["group_name"] = minor["group_code"].map(SOC_MAJOR_LABELS).fillna("其他")
    minor["unit_label"] = minor["title_chinese"].fillna(minor["occupation_title"])
    out["SOC职业组别"] = minor

    detail = data["SOC详细职业"].copy()
    detail["group_code"] = detail["occu_soc_code"].astype(str).str.slice(0, 2)
    detail["group_name"] = detail["group_code"].map(SOC_MAJOR_LABELS).fillna("其他")
    detail["unit_label"] = detail["onet_occupationtitle"]
    out["SOC详细职业"] = detail

    zl = data["智联二级职业"].copy()
    zl["group_code"] = zl["jd_class1"]
    zl["group_name"] = zl["jd_class1"]
    zl["unit_label"] = zl["jd_class1"].astype(str) + " / " + zl["jd_class2"].astype(str)
    out["智联二级职业"] = zl
    return out


def anova_table(df: pd.DataFrame, level: str) -> tuple[dict[str, float], pd.DataFrame]:
    clean = df.dropna(subset=["exposure", "group_name"]).copy()
    group_stats = (
        clean.groupby(["group_code", "group_name"], as_index=False)
        .agg(
            n=("exposure", "size"),
            mean_exposure=("exposure", "mean"),
            sd_exposure=("exposure", "std"),
            median_exposure=("exposure", "median"),
            min_exposure=("exposure", "min"),
            max_exposure=("exposure", "max"),
        )
        .sort_values("mean_exposure", ascending=False)
    )
    overall_mean = clean["exposure"].mean()
    ss_total = float(((clean["exposure"] - overall_mean) ** 2).sum())
    ss_between = 0.0
    ss_within = 0.0
    values_by_group = []
    for _, group in clean.groupby("group_name"):
        values = group["exposure"].to_numpy()
        values_by_group.append(values)
        ss_between += len(values) * float((values.mean() - overall_mean) ** 2)
        ss_within += float(((values - values.mean()) ** 2).sum())
    k = len(values_by_group)
    n = len(clean)
    df_between = k - 1
    df_within = n - k
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    f_stat = ms_between / ms_within if ms_within and ms_within > 0 else np.nan
    p_value = stats.f.sf(f_stat, df_between, df_within) if np.isfinite(f_stat) else np.nan
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
    omega_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within) if ss_total > 0 else np.nan
    omega_sq = max(float(omega_sq), 0.0) if np.isfinite(omega_sq) else np.nan
    result = {
        "level": level,
        "n": n,
        "groups": k,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "df_between": df_between,
        "df_within": df_within,
        "F": f_stat,
        "p_value": p_value,
        "eta_squared": eta_sq,
        "omega_squared": omega_sq,
        "mean_exposure": overall_mean,
        "sd_exposure": clean["exposure"].std(),
    }
    group_stats.insert(0, "level", level)
    return result, group_stats


def run_models(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    groups = []
    for level, df in data.items():
        result, group_stats = anova_table(df, level)
        results.append(result)
        groups.append(group_stats)
    anova = pd.DataFrame(results)
    group_stats = pd.concat(groups, ignore_index=True)
    anova.to_csv(TABLE_DIR / "anova_variance_decomposition.csv", index=False, encoding="utf-8-sig")
    group_stats.to_csv(TABLE_DIR / "group_exposure_statistics.csv", index=False, encoding="utf-8-sig")
    return anova, group_stats


def plot_variance_explained(anova: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plot_df = anova.sort_values("eta_squared", ascending=False)
    bars = ax.bar(plot_df["level"], plot_df["eta_squared"], color=["#2563EB", "#D55E00", "#009E73"], alpha=0.86)
    ax.set_title("职业大类对 AI-LLM 暴露度差异的解释比例")
    ax.set_ylabel("Eta squared（组间方差占比）")
    ax.set_ylim(0, max(0.8, plot_df["eta_squared"].max() + 0.08))
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in plot_df["eta_squared"]], padding=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_2_1_variance_explained.png", dpi=300)
    plt.close(fig)


def plot_group_boxplots(data: dict[str, pd.DataFrame]) -> None:
    for level, file_name in [("SOC详细职业", "fig_2_1_soc_detail_group_boxplot.png"), ("智联二级职业", "fig_2_1_zl_group_boxplot.png")]:
        df = data[level].dropna(subset=["exposure", "group_name"]).copy()
        order = df.groupby("group_name")["exposure"].mean().sort_values(ascending=False).index
        height = max(6, 0.38 * len(order))
        fig, ax = plt.subplots(figsize=(10, height))
        sns.boxplot(data=df, y="group_name", x="exposure", order=order, color="#8ECae6", ax=ax)
        sns.stripplot(data=df, y="group_name", x="exposure", order=order, color="#1F2937", size=3, alpha=0.55, ax=ax)
        ax.set_title(f"{level}：不同职业大类的 AI-LLM 暴露度分布")
        ax.set_xlabel("基期暴露指数")
        ax.set_ylabel("职业大类")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / file_name, dpi=300)
        plt.close(fig)


def write_report(anova: pd.DataFrame, group_stats: pd.DataFrame) -> None:
    best = anova.sort_values("eta_squared", ascending=False).iloc[0]
    lines = [
        "# 2.1 职业结构分化：ANOVA 与方差分解",
        "",
        "## 研究问题",
        "",
        "本节研究 AI-LLM 暴露度是否存在系统性的职业结构分化。第一部分复现已经说明哪些职业暴露度最高或最低，但单纯排名不能说明这种差异是否来自职业大类之间的结构性差别。因此本节使用单因素 ANOVA 和方差分解，把职业大类作为解释变量，检验职业类别能解释多少暴露度差异。",
        "",
        "## 模型设定",
        "",
        "模型形式为：",
        "",
        "```text",
        "Exposure_i = alpha + OccupationGroup_i + epsilon_i",
        "```",
        "",
        "其中 `Exposure_i` 是职业 i 的基期 AI-LLM 暴露度。SOC 数据使用职业编码前两位构造职业大类，智联数据使用 `jd_class1` 作为一级职业大类。ANOVA 适合本节，是因为它能够直接判断“组间差异”相对于“组内差异”是否足够大，并用 eta squared 衡量职业大类对暴露度方差的解释比例。",
        "",
        "## 主要结果",
        "",
        anova[["level", "n", "groups", "F", "p_value", "eta_squared", "omega_squared"]].round(4).to_markdown(index=False),
        "",
        f"从 eta squared 看，职业大类解释比例最高的是 **{best['level']}**，组间方差占比为 **{best['eta_squared']:.3f}**。这说明该职业体系下，AI-LLM 暴露度并非随机散布在职业之间，而是明显嵌入职业大类结构。",
        "",
        "各层级暴露度最高的职业大类如下：",
    ]
    for level in group_stats["level"].unique():
        top = group_stats[group_stats["level"] == level].sort_values("mean_exposure", ascending=False).head(5)
        lines.append("")
        lines.append(f"### {level}")
        for _, row in top.iterrows():
            lines.append(f"- {row['group_name']}：均值 {row['mean_exposure']:.3f}，样本数 {int(row['n'])}")
    lines.extend(
        [
            "",
            "## 结论解释",
            "",
            "ANOVA 结果支持“职业结构分化”这一判断。AI-LLM 暴露度较高的类别主要集中在计算机与数学、商业金融、办公行政、市场运营等知识密集或文本信息处理密集的职业大类；暴露度较低的类别多集中在餐饮、清洁、运输、生产制造、现场服务等需要物理操作和面对面互动的职业大类。",
            "",
            "这个结论说明，AI-LLM 暴露度不是少数职业的孤立高值，而是与职业任务结构系统相关。它为后续 2.2 的动态分析提供基础：如果职业结构分化明显，就需要进一步考察这种分化在 2018-2024 年是否扩大、收敛或发生阶段性变化。",
            "",
            "## 输出文件",
            "",
            "- `results/tables/anova_variance_decomposition.csv`",
            "- `results/tables/group_exposure_statistics.csv`",
            "- `results/figures/fig_2_1_variance_explained.png`",
            "- `results/figures/fig_2_1_soc_detail_group_boxplot.png`",
            "- `results/figures/fig_2_1_zl_group_boxplot.png`",
        ]
    )
    (PART_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    data = add_groups(read_data())
    anova, group_stats = run_models(data)
    plot_variance_explained(anova)
    plot_group_boxplots(data)
    write_report(anova, group_stats)
    print("2.1 ANOVA finished.")
    print(f"Tables: {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
