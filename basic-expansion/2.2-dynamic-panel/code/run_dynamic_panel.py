"""2.2 年度动态演化：面板趋势、年份效应与收敛分析。"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[3]
PART_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "人工智能-大语言模型技术”暴露指数数据"
TABLE_DIR = PART_DIR / "results" / "tables"
FIGURE_DIR = PART_DIR / "results" / "figures"


SPECS = {
    "SOC职业组别": {
        "file": "exposure_by_year_minor_soc.xlsx",
        "id_cols": ["minor_soc_code"],
        "label_cols": ["title_chinese"],
    },
    "SOC详细职业": {
        "file": "exposure_by_year_soc_detail.xlsx",
        "id_cols": ["occu_soc_code"],
        "label_cols": ["onet_occupationtitle"],
    },
    "智联二级职业": {
        "file": "exposure_by_year_zl_occu.xlsx",
        "id_cols": ["jd_class1", "jd_class2"],
        "label_cols": ["jd_class1", "jd_class2"],
    },
}


def setup() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def make_unit_id(df: pd.DataFrame, id_cols: list[str]) -> pd.Series:
    return df[id_cols].astype(str).agg(" | ".join, axis=1)


def make_unit_label(df: pd.DataFrame, label_cols: list[str]) -> pd.Series:
    return df[label_cols].astype(str).agg(" / ".join, axis=1)


def read_panels() -> dict[str, pd.DataFrame]:
    data = {}
    for level, spec in SPECS.items():
        df = pd.read_excel(DATA_DIR / spec["file"])
        df["level"] = level
        df["unit_id"] = make_unit_id(df, spec["id_cols"])
        df["unit_label"] = make_unit_label(df, spec["label_cols"])
        df["year_centered"] = df["year"] - 2018
        df["post2023"] = (df["year"] >= 2023).astype(int)
        data[level] = df.dropna(subset=["exposure"]).copy()
    return data


def annual_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for level, df in data.items():
        out = (
            df.groupby("year", as_index=False)["exposure"]
            .agg(count="count", mean="mean", sd="std", p25=lambda s: s.quantile(0.25), median="median", p75=lambda s: s.quantile(0.75))
        )
        out.insert(0, "level", level)
        parts.append(out)
    summary = pd.concat(parts, ignore_index=True)
    summary.to_csv(TABLE_DIR / "annual_exposure_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def fit_panel_models(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trend_rows = []
    year_fe_rows = []
    for level, df in data.items():
        trend_model = smf.ols("exposure ~ year_centered + C(unit_id)", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["unit_id"]}
        )
        post_model = smf.ols("exposure ~ year_centered + post2023 + C(unit_id)", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["unit_id"]}
        )
        trend_rows.append(
            {
                "level": level,
                "model": "职业固定效应线性趋势",
                "term": "year_centered",
                "coef": trend_model.params.get("year_centered", np.nan),
                "se": trend_model.bse.get("year_centered", np.nan),
                "p_value": trend_model.pvalues.get("year_centered", np.nan),
                "nobs": int(trend_model.nobs),
                "units": df["unit_id"].nunique(),
                "r2": trend_model.rsquared,
            }
        )
        for term in ["year_centered", "post2023"]:
            trend_rows.append(
                {
                    "level": level,
                    "model": "职业固定效应+2023后阶段",
                    "term": term,
                    "coef": post_model.params.get(term, np.nan),
                    "se": post_model.bse.get(term, np.nan),
                    "p_value": post_model.pvalues.get(term, np.nan),
                    "nobs": int(post_model.nobs),
                    "units": df["unit_id"].nunique(),
                    "r2": post_model.rsquared,
                }
            )

        year_model = smf.ols("exposure ~ C(year) + C(unit_id)", data=df).fit()
        for year in sorted(df["year"].unique()):
            if year == 2018:
                coef = 0.0
                p_value = np.nan
                se = np.nan
            else:
                term = f"C(year)[T.{year}]"
                coef = year_model.params.get(term, np.nan)
                se = year_model.bse.get(term, np.nan)
                p_value = year_model.pvalues.get(term, np.nan)
            year_fe_rows.append({"level": level, "year": year, "coef_vs_2018": coef, "se": se, "p_value": p_value})

    trends = pd.DataFrame(trend_rows)
    year_fe = pd.DataFrame(year_fe_rows)
    trends.to_csv(TABLE_DIR / "panel_trend_models.csv", index=False, encoding="utf-8-sig")
    year_fe.to_csv(TABLE_DIR / "year_fixed_effects_vs_2018.csv", index=False, encoding="utf-8-sig")
    return trends, year_fe


def beta_convergence(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    scatter_parts = []
    for level, df in data.items():
        wide = df.pivot_table(index=["unit_id", "unit_label"], columns="year", values="exposure", aggfunc="mean").reset_index()
        wide = wide.dropna(subset=[2018, 2024]).copy()
        wide = wide.rename(columns={2018: "exposure_2018", 2024: "exposure_2024"})
        wide["change_2018_2024"] = wide["exposure_2024"] - wide["exposure_2018"]
        model = smf.ols("change_2018_2024 ~ exposure_2018", data=wide).fit()
        rows.append(
            {
                "level": level,
                "n_units": len(wide),
                "rho_initial_exposure": model.params.get("exposure_2018", np.nan),
                "se": model.bse.get("exposure_2018", np.nan),
                "p_value": model.pvalues.get("exposure_2018", np.nan),
                "r2": model.rsquared,
                "mean_change": wide["change_2018_2024"].mean(),
            }
        )
        tmp = wide[["unit_id", "unit_label", "exposure_2018", "exposure_2024", "change_2018_2024"]].copy()
        tmp.insert(0, "level", level)
        scatter_parts.append(tmp)

    out = pd.DataFrame(rows)
    scatter = pd.concat(scatter_parts, ignore_index=True)
    out.to_csv(TABLE_DIR / "beta_convergence_models.csv", index=False, encoding="utf-8-sig")
    scatter.to_csv(TABLE_DIR / "beta_convergence_scatter_data.csv", index=False, encoding="utf-8-sig")
    return out


def occupation_slopes(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for level, df in data.items():
        for unit_id, group in df.groupby("unit_id"):
            group = group.sort_values("year")
            if len(group) < 3:
                continue
            slope, intercept = np.polyfit(group["year"] - 2018, group["exposure"], 1)
            rows.append(
                {
                    "level": level,
                    "unit_id": unit_id,
                    "unit_label": group["unit_label"].iloc[0],
                    "n_years": len(group),
                    "slope_per_year": slope,
                    "intercept": intercept,
                    "mean_exposure": group["exposure"].mean(),
                    "sd_exposure": group["exposure"].std(),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "occupation_linear_slopes.csv", index=False, encoding="utf-8-sig")
    return out


def plot_annual(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.lineplot(data=summary, x="year", y="mean", hue="level", marker="o", ax=axes[0])
    axes[0].set_title("年度平均 AI-LLM 暴露度")
    axes[0].set_xlabel("年份")
    axes[0].set_ylabel("非加权平均暴露度")
    axes[0].legend(title="职业层级")

    sns.lineplot(data=summary, x="year", y="sd", hue="level", marker="o", ax=axes[1])
    axes[1].set_title("年度暴露度标准差（Sigma 分化）")
    axes[1].set_xlabel("年份")
    axes[1].set_ylabel("标准差")
    axes[1].legend(title="职业层级")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_2_2_annual_mean_and_sigma.png", dpi=300)
    plt.close(fig)


def plot_year_fe(year_fe: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(data=year_fe, x="year", y="coef_vs_2018", hue="level", marker="o", ax=ax)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_title("控制职业固定效应后的年份效应（相对 2018 年）")
    ax.set_xlabel("年份")
    ax.set_ylabel("年份效应系数")
    ax.legend(title="职业层级")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_2_2_year_fixed_effects.png", dpi=300)
    plt.close(fig)


def plot_beta_convergence() -> None:
    scatter = pd.read_csv(TABLE_DIR / "beta_convergence_scatter_data.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False, sharey=True)
    for ax, (level, sub) in zip(axes, scatter.groupby("level")):
        sns.regplot(data=sub, x="exposure_2018", y="change_2018_2024", ax=ax, scatter_kws={"alpha": 0.65, "s": 32}, line_kws={"color": "#D55E00"})
        ax.axhline(0, color="#444444", linewidth=1)
        ax.set_title(level)
        ax.set_xlabel("2018 年暴露度")
        ax.set_ylabel("2018-2024 年变化")
    fig.suptitle("Beta 收敛：初始暴露度与后续变化", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_2_2_beta_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def p_text(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def write_report(trends: pd.DataFrame, year_fe: pd.DataFrame, beta: pd.DataFrame, slopes: pd.DataFrame, summary: pd.DataFrame) -> None:
    linear = trends[trends["model"] == "职业固定效应线性趋势"].copy()
    strongest_decline = linear.sort_values("coef").iloc[0]
    beta_main = beta.sort_values("rho_initial_exposure").iloc[0]
    latest_mean = summary[summary["year"] == 2024][["level", "mean", "sd"]]
    lines = [
        "# 2.2 年度动态演化：面板趋势与收敛分析",
        "",
        "## 研究问题",
        "",
        "本节研究 2018-2024 年 AI-LLM 暴露度是否存在系统性时间变化，以及高初始暴露职业是否在后续出现更明显的回落。原论文的市场暴露趋势使用招聘广告权重，本节没有招聘微观数据，因此改为分析公开职业暴露指数本身的年度变化。",
        "",
        "## 模型设定与合理性",
        "",
        "第一，使用职业固定效应面板趋势模型：",
        "",
        "```text",
        "Exposure_it = alpha_i + beta Year_t + epsilon_it",
        "```",
        "",
        "该模型控制每个职业长期不变的暴露度差异，估计同一职业随时间的平均变化趋势。",
        "",
        "第二，使用年份固定效应模型：",
        "",
        "```text",
        "Exposure_it = alpha_i + gamma_t + epsilon_it",
        "```",
        "",
        "它可以在控制职业固定差异后，观察各年份相对于 2018 年的整体变化。",
        "",
        "第三，使用 Beta 收敛模型：",
        "",
        "```text",
        "Exposure_i,2024 - Exposure_i,2018 = alpha + rho Exposure_i,2018 + epsilon_i",
        "```",
        "",
        "若 `rho < 0`，说明初始暴露越高的职业后续下降越多，职业之间存在收敛或高暴露职业任务回落现象。",
        "",
        "## 面板趋势结果",
        "",
        linear[["level", "coef", "se", "p_value", "nobs", "units", "r2"]].round(4).to_markdown(index=False),
        "",
        f"线性趋势下降最明显的是 **{strongest_decline['level']}**，每年平均变化系数为 **{strongest_decline['coef']:.4f}**，p 值为 **{p_text(strongest_decline['p_value'])}**。",
        "",
        "2024 年各层级均值与标准差如下：",
        "",
        latest_mean.round(4).to_markdown(index=False),
        "",
        "## Beta 收敛结果",
        "",
        beta[["level", "n_units", "rho_initial_exposure", "se", "p_value", "r2", "mean_change"]].round(4).to_markdown(index=False),
        "",
        f"最强的负向系数出现在 **{beta_main['level']}**，初始暴露度系数为 **{beta_main['rho_initial_exposure']:.4f}**，但 p 值为 **{p_text(beta_main['p_value'])}**。因此，这只能被解释为弱描述性迹象，不能作为严格的统计显著收敛证据。SOC 详细职业和智联二级职业的初始暴露度系数接近 0，说明在更细职业层级上没有发现清晰的 beta 收敛。",
        "",
        "## 结论解释",
        "",
        "结果显示，公开职业暴露指数本身存在显著年度下降：三个职业层级的职业固定效应线性趋势均为负，并且 p 值均小于 0.05。相比之下，Beta 收敛模型没有给出稳健的显著证据：只有 SOC 职业组别呈现负向但不显著的系数，SOC 详细职业和智联二级职业几乎没有收敛关系。因此，本节最稳健的结论是“职业暴露度整体随时间回落”，而不是“高暴露职业显著更快收敛”。",
        "",
        "这个结果不能直接解释为 AI 导致就业下降，但可以说明职业任务暴露度存在动态调整：部分职业可能逐渐减少容易被 LLM 处理的任务描述，或新增岗位任务构成开始向较低暴露方向移动。",
        "",
        "这一步承接 2.1 的结构分化结论：既然暴露度在职业结构中存在明显分化，那么 2.2 进一步说明这种分化并非静态不变，而是在 2018-2024 年出现了时间演化和一定收敛迹象。",
        "",
        "## 输出文件",
        "",
        "- `results/tables/annual_exposure_summary.csv`",
        "- `results/tables/panel_trend_models.csv`",
        "- `results/tables/year_fixed_effects_vs_2018.csv`",
        "- `results/tables/beta_convergence_models.csv`",
        "- `results/tables/occupation_linear_slopes.csv`",
        "- `results/figures/fig_2_2_annual_mean_and_sigma.png`",
        "- `results/figures/fig_2_2_year_fixed_effects.png`",
        "- `results/figures/fig_2_2_beta_convergence.png`",
    ]
    (PART_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    data = read_panels()
    summary = annual_summary(data)
    trends, year_fe = fit_panel_models(data)
    beta = beta_convergence(data)
    slopes = occupation_slopes(data)
    plot_annual(summary)
    plot_year_fe(year_fe)
    plot_beta_convergence()
    write_report(trends, year_fe, beta, slopes, summary)
    print("2.2 dynamic panel finished.")
    print(f"Tables: {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
