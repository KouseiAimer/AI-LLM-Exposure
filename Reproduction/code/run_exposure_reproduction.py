"""
Reproduce descriptive results from the public China AI-LLM exposure index.

This script uses only the public exposure-index workbooks stored in the parent
project directory. It does not use the original Zhaopin posting microdata, so it
cannot reproduce the paper's labor-demand regressions. The outputs are saved
under Reproduction/results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
REPRO = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "人工智能-大语言模型技术”暴露指数数据"
TABLE_DIR = REPRO / "results" / "tables"
FIGURE_DIR = REPRO / "results" / "figures"
LOG_DIR = REPRO / "results" / "logs"


FILES = {
    "base_minor_soc": "exposure_base_minor_soc.xlsx",
    "base_soc_detail": "exposure_base_soc_detail.xlsx",
    "base_zl_occu": "exposure_base_zl_occu.xlsx",
    "year_minor_soc": "exposure_by_year_minor_soc.xlsx",
    "year_soc_detail": "exposure_by_year_soc_detail.xlsx",
    "year_zl_occu": "exposure_by_year_zl_occu.xlsx",
}

LABELS = {
    "base_minor_soc": "SOC minor group, baseline 2018-2021",
    "base_soc_detail": "SOC detailed occupation, baseline 2018-2021",
    "base_zl_occu": "Zhaopin level-2 occupation, baseline 2018-2021",
    "year_minor_soc": "SOC minor group, annual 2018-2024",
    "year_soc_detail": "SOC detailed occupation, annual 2018-2024",
    "year_zl_occu": "Zhaopin level-2 occupation, annual 2018-2024",
}

PLOT_LABELS = {
    "base_minor_soc": "SOC 职业组别",
    "base_soc_detail": "SOC 详细职业",
    "base_zl_occu": "智联二级职业",
    "year_minor_soc": "SOC 职业组别",
    "year_soc_detail": "SOC 详细职业",
    "year_zl_occu": "智联二级职业",
}

ID_COLUMNS = {
    "base_minor_soc": ["minor_soc_code", "occupation_title", "title_chinese"],
    "base_soc_detail": ["occu_soc_code", "onet_occupationtitle"],
    "base_zl_occu": ["jd_class1", "jd_class2"],
    "year_minor_soc": ["minor_soc_code", "occupation_title", "title_chinese"],
    "year_soc_detail": ["occu_soc_code", "onet_occupationtitle"],
    "year_zl_occu": ["jd_class1", "jd_class2"],
}

NAME_COLUMNS = {
    "base_minor_soc": "title_chinese",
    "base_soc_detail": "onet_occupationtitle",
    "base_zl_occu": "jd_class2",
    "year_minor_soc": "title_chinese",
    "year_soc_detail": "onet_occupationtitle",
    "year_zl_occu": "jd_class2",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]


def read_index_files() -> dict[str, pd.DataFrame]:
    data = {}
    for key, filename in FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing exposure index workbook: {path}")
        df = pd.read_excel(path)
        df["source_file"] = filename
        data[key] = df
    return data


def save_data_inventory(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, df in data.items():
        rows.append(
            {
                "dataset": key,
                "description": LABELS[key],
                "source_file": FILES[key],
                "rows": len(df),
                "unique_units": int(df[ID_COLUMNS[key]].drop_duplicates().shape[0]),
                "columns": len(df.columns) - 1,
                "has_year": "year" in df.columns,
                "min_year": int(df["year"].min()) if "year" in df.columns else np.nan,
                "max_year": int(df["year"].max()) if "year" in df.columns else np.nan,
                "non_missing_exposure": int(df["exposure"].notna().sum()),
                "missing_exposure": int(df["exposure"].isna().sum()),
                "mean_exposure": df["exposure"].mean(),
                "std_exposure": df["exposure"].std(),
                "min_exposure": df["exposure"].min(),
                "max_exposure": df["exposure"].max(),
            }
        )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(TABLE_DIR / "data_inventory.csv", index=False, encoding="utf-8-sig")
    return inventory


def save_missing_exposure_rows(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for key, df in data.items():
        missing = df[df["exposure"].isna()].copy()
        if missing.empty:
            continue
        keep = ID_COLUMNS[key] + (["year"] if "year" in missing.columns else [])
        missing = missing[keep]
        missing.insert(0, "dataset", key)
        parts.append(missing)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["dataset"])
    out.to_csv(TABLE_DIR / "missing_exposure_rows.csv", index=False, encoding="utf-8-sig")
    return out


def save_summary_stats(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, df in data.items():
        desc = df["exposure"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        rows.append(
            {
                "dataset": key,
                "description": LABELS[key],
                "count": desc["count"],
                "mean": desc["mean"],
                "std": desc["std"],
                "min": desc["min"],
                "p10": desc["10%"],
                "p25": desc["25%"],
                "median": desc["50%"],
                "p75": desc["75%"],
                "p90": desc["90%"],
                "max": desc["max"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "exposure_summary_statistics.csv", index=False, encoding="utf-8-sig")
    return out


def save_top_bottom_tables(data: dict[str, pd.DataFrame], n: int = 20) -> None:
    for key in ["base_minor_soc", "base_soc_detail", "base_zl_occu"]:
        df = data[key].copy()
        cols = ID_COLUMNS[key] + ["exposure"]
        high = df.sort_values("exposure", ascending=False).head(n)[cols]
        low = df.sort_values("exposure", ascending=True).head(n)[cols]
        high.to_csv(TABLE_DIR / f"{key}_top{n}_high_exposure.csv", index=False, encoding="utf-8-sig")
        low.to_csv(TABLE_DIR / f"{key}_top{n}_low_exposure.csv", index=False, encoding="utf-8-sig")
        combined = pd.concat(
            [
                high.assign(rank_type="暴露度最高", rank=np.arange(1, len(high) + 1)),
                low.assign(rank_type="暴露度最低", rank=np.arange(1, len(low) + 1)),
            ],
            ignore_index=True,
        )
        combined.to_csv(TABLE_DIR / f"{key}_top_bottom_{n}.csv", index=False, encoding="utf-8-sig")


def save_annual_trends(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for key in ["year_minor_soc", "year_soc_detail", "year_zl_occu"]:
        df = data[key].copy()
        trend = (
            df.groupby("year", as_index=False)["exposure"]
            .agg(["count", "mean", "std", "min", "max"])
        )
        trend.insert(0, "dataset", key)
        trend.insert(1, "description", LABELS[key])
        parts.append(trend)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(TABLE_DIR / "annual_unweighted_exposure_trends.csv", index=False, encoding="utf-8-sig")
    return out


def save_occupation_changes(data: dict[str, pd.DataFrame], n: int = 20) -> None:
    for key in ["year_minor_soc", "year_soc_detail", "year_zl_occu"]:
        df = data[key].copy()
        id_cols = ID_COLUMNS[key]
        wide = df.pivot_table(index=id_cols, columns="year", values="exposure", aggfunc="mean")
        wide = wide.reset_index()
        if 2018 not in wide.columns or 2024 not in wide.columns:
            continue
        wide["change_2018_2024"] = wide[2024] - wide[2018]
        wide["pct_change_2018_2024"] = wide["change_2018_2024"] / wide[2018]
        wide = wide.rename(columns={2018: "exposure_2018", 2024: "exposure_2024"})
        cols = id_cols + ["exposure_2018", "exposure_2024", "change_2018_2024", "pct_change_2018_2024"]
        wide[cols].sort_values("change_2018_2024", ascending=False).head(n).to_csv(
            TABLE_DIR / f"{key}_largest_exposure_increases_2018_2024.csv",
            index=False,
            encoding="utf-8-sig",
        )
        wide[cols].sort_values("change_2018_2024", ascending=True).head(n).to_csv(
            TABLE_DIR / f"{key}_largest_exposure_decreases_2018_2024.csv",
            index=False,
            encoding="utf-8-sig",
        )


def save_base_vs_dynamic_check(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pairs = [
        ("SOC 职业组别", "base_minor_soc", "year_minor_soc", ["minor_soc_code"]),
        ("SOC 详细职业", "base_soc_detail", "year_soc_detail", ["occu_soc_code"]),
        ("智联二级职业", "base_zl_occu", "year_zl_occu", ["jd_class1", "jd_class2"]),
    ]
    rows = []
    for level, base_key, year_key, keys in pairs:
        base = data[base_key].copy()
        yearly = data[year_key].copy()
        avg = (
            yearly[yearly["year"].between(2018, 2021)]
            .groupby(keys, dropna=False)["exposure"]
            .mean()
            .reset_index(name="avg_exposure_2018_2021")
        )
        merged = base.merge(avg, on=keys, how="left")
        merged["base_minus_simple_avg"] = merged["exposure"] - merged["avg_exposure_2018_2021"]
        rows.append(
            {
                "level": level,
                "units": len(merged),
                "matched_units": int(merged["avg_exposure_2018_2021"].notna().sum()),
                "mean_diff": merged["base_minus_simple_avg"].mean(),
                "median_diff": merged["base_minus_simple_avg"].median(),
                "max_abs_diff": merged["base_minus_simple_avg"].abs().max(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "base_vs_2018_2021_simple_average_check.csv", index=False, encoding="utf-8-sig")
    return out


def plot_base_distribution(data: dict[str, pd.DataFrame]) -> None:
    plot_df = []
    for key in ["base_minor_soc", "base_soc_detail", "base_zl_occu"]:
        tmp = data[key][["exposure"]].copy()
        tmp["dataset"] = PLOT_LABELS[key]
        plot_df.append(tmp)
    plot_df = pd.concat(plot_df, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=plot_df,
        x="exposure",
        hue="dataset",
        bins=18,
        element="step",
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set_title("基期 AI-LLM 暴露指数分布")
    ax.set_xlabel("暴露指数")
    ax.set_ylabel("密度")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_distribution_baseline_exposure.png", dpi=300)
    plt.close(fig)


def plot_top_bottom_minor_soc(data: dict[str, pd.DataFrame], n: int = 20) -> None:
    df = data["base_minor_soc"].copy()
    high = df.nlargest(n, "exposure").assign(group="暴露度最高")
    low = df.nsmallest(n, "exposure").assign(group="暴露度最低")
    plot_df = pd.concat([high, low], ignore_index=True)
    plot_df["label"] = plot_df["title_chinese"].fillna(plot_df["occupation_title"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharex=True)
    for ax, group, color in zip(axes, ["暴露度最高", "暴露度最低"], ["#D55E00", "#0072B2"]):
        sub = plot_df[plot_df["group"] == group].sort_values("exposure", ascending=True)
        ax.barh(sub["label"], sub["exposure"], color=color, alpha=0.85)
        ax.set_title(group)
        ax.set_xlabel("基期暴露指数")
        ax.set_xlim(0, 1)
        for i, value in enumerate(sub["exposure"]):
            ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=8)
    fig.suptitle("SOC 职业组别 AI-LLM 暴露度最高和最低的职业", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_top_bottom20_base_minor_soc.png", dpi=300)
    plt.close(fig)


def display_labels(df: pd.DataFrame, key: str) -> pd.Series:
    if key.endswith("zl_occu"):
        return df["jd_class1"].astype(str) + " / " + df["jd_class2"].astype(str)
    if key.endswith("minor_soc"):
        return df["title_chinese"].fillna(df["occupation_title"]).astype(str)
    return df["onet_occupationtitle"].astype(str)


def plot_top_bottom_base(
    data: dict[str, pd.DataFrame],
    key: str,
    file_name: str,
    n: int = 20,
    title_note: str | None = None,
) -> None:
    df = data[key].copy()
    df["label"] = display_labels(df, key)
    high = df.nlargest(n, "exposure").sort_values("exposure", ascending=True)
    low = df.nsmallest(n, "exposure").sort_values("exposure", ascending=False)

    height = max(6.0, n * 0.36)
    fig, axes = plt.subplots(1, 2, figsize=(16, height), sharex=True)
    for ax, sub, title, color in [
        (axes[0], high, f"暴露度最高 {n} 个", "#D55E00"),
        (axes[1], low, f"暴露度最低 {n} 个", "#0072B2"),
    ]:
        ax.barh(sub["label"], sub["exposure"], color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("基期暴露指数")
        ax.set_xlim(0, 1)
        ax.tick_params(axis="y", labelsize=8)
        for i, value in enumerate(sub["exposure"]):
            ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=7)

    note = f"（{title_note}）" if title_note else ""
    fig.suptitle(f"{PLOT_LABELS[key]}基期 AI-LLM 暴露度排序{note}", y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / file_name, dpi=300)
    plt.close(fig)


def plot_annual_trends(annual: pd.DataFrame) -> None:
    plot_df = annual.copy()
    plot_df["职业分类层级"] = plot_df["dataset"].map(PLOT_LABELS)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=plot_df, x="year", y="mean", hue="职业分类层级", marker="o", ax=ax)
    ax.set_title("不同职业分类层级的年度平均 AI-LLM 暴露度")
    ax.set_xlabel("年份")
    ax.set_ylabel("非加权平均暴露度")
    ax.legend(title="职业分类层级", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_annual_unweighted_mean_exposure.png", dpi=300)
    plt.close(fig)


def plot_quartile_trends(data: dict[str, pd.DataFrame]) -> None:
    """Show exposure trends by baseline quartile, not labor-demand shares."""
    base = data["base_minor_soc"].copy()
    yearly = data["year_minor_soc"].copy()
    base["baseline_quartile"] = pd.qcut(
        base["exposure"],
        q=4,
        labels=["Q1 最低", "Q2 较低", "Q3 较高", "Q4 最高"],
    )
    merged = yearly.merge(
        base[["minor_soc_code", "baseline_quartile"]],
        on="minor_soc_code",
        how="inner",
    )
    quartile_trend = (
        merged.groupby(["baseline_quartile", "year"], observed=True)["exposure"]
        .mean()
        .reset_index()
    )
    quartile_trend.to_csv(
        TABLE_DIR / "minor_soc_baseline_quartile_annual_exposure.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=quartile_trend, x="year", y="exposure", hue="baseline_quartile", marker="o", ax=ax)
    ax.set_title("按基期暴露度四分位分组的年度趋势")
    ax.set_xlabel("年份")
    ax.set_ylabel("非加权平均暴露度")
    ax.legend(title="基期四分位")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_minor_soc_quartile_exposure_trends.png", dpi=300)
    plt.close(fig)


def plot_selected_minor_occupation_trends(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    selected_codes = [
        "13-2000",
        "41-4000",
        "15-1200",
        "41-3000",
        "43-4000",
        "41-9000",
        "11-2000",
        "13-1000",
        "27-1000",
        "11-3000",
        "43-6000",
        "17-2000",
        "11-9000",
        "53-3000",
        "51-9000",
        "51-2000",
    ]
    base = data["base_minor_soc"][
        ["minor_soc_code", "title_chinese", "occupation_title", "exposure"]
    ].copy()
    selected = base[base["minor_soc_code"].isin(selected_codes)].copy()
    selected = selected.sort_values("exposure", ascending=False)
    order = selected["minor_soc_code"].tolist()

    yearly = data["year_minor_soc"].merge(
        selected[["minor_soc_code", "title_chinese"]],
        on="minor_soc_code",
        how="inner",
        suffixes=("", "_base"),
    )
    yearly["title_for_plot"] = yearly["title_chinese_base"].fillna(yearly["occupation_title"])
    yearly["minor_soc_code"] = pd.Categorical(yearly["minor_soc_code"], categories=order, ordered=True)
    yearly = yearly.sort_values(["minor_soc_code", "year"])

    ncols = 4
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9.5), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, code in zip(axes_flat, order):
        sub = yearly[yearly["minor_soc_code"] == code]
        if sub.empty:
            ax.axis("off")
            continue
        title = sub["title_for_plot"].iloc[0]
        ax.plot(sub["year"], sub["exposure"], marker="o", linewidth=1.8, color="#2563EB")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    for ax in axes_flat[len(order) :]:
        ax.axis("off")

    fig.suptitle("部分 SOC 职业组别年度 AI-LLM 暴露指数变化", y=1.02)
    fig.supxlabel("年份", y=0.02)
    fig.supylabel("暴露指数", x=0.01)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_selected_minor_occupation_trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    out = yearly[
        ["minor_soc_code", "title_for_plot", "year", "exposure"]
    ].rename(columns={"title_for_plot": "title_chinese"})
    out.to_csv(
        TABLE_DIR / "selected_minor_occupation_yearly_trends.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return out


def write_run_metadata(data: dict[str, pd.DataFrame]) -> None:
    metadata = {
        "project_root": str(ROOT),
        "reproduction_dir": str(REPRO),
        "data_dir": str(DATA_DIR),
        "input_files": FILES,
        "note": (
            "Outputs use public exposure-index workbooks only. The paper's "
            "labor-demand regressions require original Zhaopin posting microdata, "
            "occupation/city weights, Felten-LLM IV, and city controls."
        ),
        "datasets": {
            key: {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "year_min": int(df["year"].min()) if "year" in df.columns else None,
                "year_max": int(df["year"].max()) if "year" in df.columns else None,
            }
            for key, df in data.items()
        },
    }
    with open(LOG_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def write_summary_report(data: dict[str, pd.DataFrame], annual: pd.DataFrame) -> None:
    minor = data["base_minor_soc"].sort_values("exposure", ascending=False)
    soc_detail = data["base_soc_detail"].sort_values("exposure", ascending=False)
    zl = data["base_zl_occu"].sort_values("exposure", ascending=False)

    minor_trend = annual[annual["dataset"] == "year_minor_soc"].sort_values("year")
    soc_trend = annual[annual["dataset"] == "year_soc_detail"].sort_values("year")
    zl_trend = annual[annual["dataset"] == "year_zl_occu"].sort_values("year")
    missing_path = TABLE_DIR / "missing_exposure_rows.csv"
    check_path = TABLE_DIR / "base_vs_2018_2021_simple_average_check.csv"
    missing_count = pd.read_csv(missing_path).shape[0] if missing_path.exists() else 0
    base_check = pd.read_csv(check_path) if check_path.exists() else pd.DataFrame()

    def trend_sentence(trend: pd.DataFrame, name: str) -> str:
        first = trend.iloc[0]
        last = trend.iloc[-1]
        change = last["mean"] - first["mean"]
        return (
            f"- {name}: {int(first['year'])} 年均值 {first['mean']:.3f}, "
            f"{int(last['year'])} 年均值 {last['mean']:.3f}, "
            f"变化 {change:.3f}。"
        )

    lines = [
        "# 复现结果摘要",
        "",
        "本摘要由 `code/run_exposure_reproduction.py` 自动生成，所有结果仅基于公开暴露指数数据。",
        "",
        "## 主要发现",
        "",
        "1. 基期暴露度最高的 SOC 职业组别是：",
    ]
    for _, row in minor.head(5).iterrows():
        lines.append(f"   - {row['title_chinese']} ({row['minor_soc_code']}): {row['exposure']:.3f}")

    lines.extend(
        [
            "",
            "2. 基期暴露度最低的 SOC 职业组别是：",
        ]
    )
    for _, row in minor.tail(5).sort_values("exposure").iterrows():
        lines.append(f"   - {row['title_chinese']} ({row['minor_soc_code']}): {row['exposure']:.3f}")

    lines.extend(
        [
            "",
            "3. 不同职业分类层级的年度平均暴露度均呈现下降或阶段性下降趋势。注意：这里是职业层面的非加权均值，不是原论文按新增职位结构加权后的市场暴露度。",
            trend_sentence(minor_trend, "SOC 职业组别"),
            trend_sentence(soc_trend, "SOC 详细职业"),
            trend_sentence(zl_trend, "智联二级职业"),
            "",
            "4. SOC 详细职业中，基期暴露度最高的前五类包括：",
        ]
    )
    for _, row in soc_detail.head(5).iterrows():
        lines.append(f"   - {row['onet_occupationtitle']} ({row['occu_soc_code']}): {row['exposure']:.3f}")

    lines.extend(
        [
            "",
            "5. 智联二级职业中，基期暴露度最高的前五类包括：",
        ]
    )
    for _, row in zl.head(5).iterrows():
        lines.append(f"   - {row['jd_class1']} / {row['jd_class2']}: {row['exposure']:.3f}")

    lines.extend(
        [
            "",
            "## 与原论文的关系",
            "",
            "这些结果能够复现和支持原论文关于“高暴露职业主要集中在白领、知识密集、文本处理和规则化任务较多的职业”的描述性结论。",
            "",
            "但由于当前文件夹没有智联招聘广告微观数据、招聘市场份额、Felten-LLM 工具变量和城市控制变量，本复现包不能复刻原论文表 3、表 4 的劳动需求和薪资回归。",
            "",
            "## 参考包增强检查",
            "",
            f"- 动态指数中 `exposure` 缺失明细已输出到 `results/tables/missing_exposure_rows.csv`，共 {missing_count} 行。",
            "- 已新增 `results/tables/base_vs_2018_2021_simple_average_check.csv`，用于比较基期指数与 2018-2021 年动态指数简单均值。基期指数和动态均值非常接近，但不是完全相同，说明公开基期指数很可能采用 pooled 样本或带权口径重新汇总。",
        ]
    )
    if not base_check.empty:
        for _, row in base_check.iterrows():
            lines.append(
                f"  - {row['level']}：匹配 {int(row['matched_units'])}/{int(row['units'])} 个单位，"
                f"平均差 {row['mean_diff']:.6f}，最大绝对差 {row['max_abs_diff']:.6f}。"
            )

    lines.extend(
        [
            "",
            "## 主要输出",
            "",
            "- `results/tables/data_inventory.csv`: 数据清单；",
            "- `results/tables/missing_exposure_rows.csv`: 动态指数缺失值明细；",
            "- `results/tables/exposure_summary_statistics.csv`: 暴露指数描述统计；",
            "- `results/tables/base_vs_2018_2021_simple_average_check.csv`: 基期指数与动态指数均值核验；",
            "- `results/tables/*top20*exposure.csv`: 高/低暴露职业排名；",
            "- `results/tables/annual_unweighted_exposure_trends.csv`: 年度非加权趋势；",
            "- `results/figures/fig_top_bottom20_base_minor_soc.png`: 高/低暴露职业图；",
            "- `results/figures/fig_top_bottom20_base_soc_detail.png`: SOC 详细职业高/低暴露图；",
            "- `results/figures/fig_top_bottom20_base_zl_occu.png`: 智联二级职业高/低暴露图；",
            "- `results/figures/fig_annual_unweighted_mean_exposure.png`: 年度平均暴露度趋势图。",
            "- `results/figures/fig_selected_minor_occupation_trends.png`: 部分 SOC 职业组别年度趋势图。",
            "",
        ]
    )

    with open(REPRO / "results" / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    data = read_index_files()
    save_data_inventory(data)
    save_missing_exposure_rows(data)
    save_summary_stats(data)
    save_top_bottom_tables(data)
    annual = save_annual_trends(data)
    save_occupation_changes(data)
    save_base_vs_dynamic_check(data)
    plot_base_distribution(data)
    plot_top_bottom_minor_soc(data)
    plot_top_bottom_base(
        data,
        "base_soc_detail",
        "fig_top_bottom20_base_soc_detail.png",
        n=20,
        title_note="职业名称保留 O*NET 英文原始标签",
    )
    plot_top_bottom_base(data, "base_zl_occu", "fig_top_bottom20_base_zl_occu.png", n=20)
    plot_annual_trends(annual)
    plot_quartile_trends(data)
    plot_selected_minor_occupation_trends(data)
    write_run_metadata(data)
    write_summary_report(data, annual)

    print("Reproduction finished.")
    print(f"Tables: {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
