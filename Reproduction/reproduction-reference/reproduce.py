# -*- coding: utf-8 -*-
"""Reproduce descriptive AI-LLM exposure-index results from released data.

This script uses the public exposure-index Excel files only. It can reproduce
descriptive rankings and annual exposure-index patterns, but not the paper's
labor-demand regressions because the released files do not include recruitment
counts, wages, education requirements, experience requirements, city identifiers,
or instrumental variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    file_name: str
    level_en: str
    level_cn: str
    id_cols: tuple[str, ...]
    label_col: str
    dynamic: bool


SPECS = [
    DatasetSpec(
        "base_minor",
        "exposure_base_minor_soc.xlsx",
        "SOC minor group",
        "SOC 职业组别",
        ("minor_soc_code", "occupation_title", "title_chinese"),
        "title_chinese",
        False,
    ),
    DatasetSpec(
        "base_detail",
        "exposure_base_soc_detail.xlsx",
        "SOC detailed occupation",
        "SOC 详细职业",
        ("occu_soc_code", "onet_occupationtitle"),
        "onet_occupationtitle",
        False,
    ),
    DatasetSpec(
        "base_zl",
        "exposure_base_zl_occu.xlsx",
        "Zhaopin level-2 occupation",
        "智联二级职业",
        ("jd_class1", "jd_class2"),
        "jd_class2",
        False,
    ),
    DatasetSpec(
        "year_minor",
        "exposure_by_year_minor_soc.xlsx",
        "SOC minor group by year",
        "SOC 职业组别 × 年份",
        ("minor_soc_code", "occupation_title", "title_chinese"),
        "title_chinese",
        True,
    ),
    DatasetSpec(
        "year_detail",
        "exposure_by_year_soc_detail.xlsx",
        "SOC detailed occupation by year",
        "SOC 详细职业 × 年份",
        ("occu_soc_code", "onet_occupationtitle"),
        "onet_occupationtitle",
        True,
    ),
    DatasetSpec(
        "year_zl",
        "exposure_by_year_zl_occu.xlsx",
        "Zhaopin level-2 occupation by year",
        "智联二级职业 × 年份",
        ("jd_class1", "jd_class2"),
        "jd_class2",
        True,
    ),
]


BASE_KEYS = ("base_minor", "base_detail", "base_zl")
YEAR_KEYS = ("year_minor", "year_detail", "year_zl")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_data_dir(root: Path) -> Path:
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and len(list(path.glob("exposure_*.xlsx"))) >= 6
    ]
    if not candidates:
        raise FileNotFoundError(
            "Could not find the exposure-index data folder containing exposure_*.xlsx files."
        )
    return candidates[0]


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def ensure_dirs(root: Path) -> dict[str, Path]:
    out = root / "reproduction" / "outputs"
    dirs = {
        "outputs": out,
        "tables": out / "tables",
        "figures": out / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for spec in SPECS:
        path = data_dir / spec.file_name
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_excel(path)
        required = set(spec.id_cols) | {"exposure"}
        if spec.dynamic:
            required.add("year")
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{spec.file_name} missing required columns: {missing}")
        data[spec.key] = df
    return data


def label_for(df: pd.DataFrame, spec: DatasetSpec) -> pd.Series:
    if spec.key.endswith("_zl"):
        return df["jd_class1"].astype(str) + " / " + df["jd_class2"].astype(str)
    if spec.key.endswith("_minor"):
        return df["title_chinese"].astype(str)
    return df["onet_occupationtitle"].astype(str)


def make_inventory(data: dict[str, pd.DataFrame], tables_dir: Path) -> pd.DataFrame:
    rows = []
    spec_by_key = {spec.key: spec for spec in SPECS}
    for key, df in data.items():
        spec = spec_by_key[key]
        exposure = df["exposure"]
        row = {
            "dataset": key,
            "file_name": spec.file_name,
            "level_cn": spec.level_cn,
            "rows": len(df),
            "unique_units": df[list(spec.id_cols)].drop_duplicates().shape[0],
            "missing_exposure": int(exposure.isna().sum()),
            "exposure_mean": exposure.mean(),
            "exposure_sd": exposure.std(),
            "exposure_min": exposure.min(),
            "exposure_p25": exposure.quantile(0.25),
            "exposure_median": exposure.median(),
            "exposure_p75": exposure.quantile(0.75),
            "exposure_max": exposure.max(),
        }
        if spec.dynamic:
            row["years"] = ",".join(map(str, sorted(df["year"].dropna().unique())))
        else:
            row["years"] = "base:2018-2021"
        rows.append(row)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(tables_dir / "dataset_inventory.csv", index=False, encoding="utf-8-sig")
    return inventory


def make_missing_table(data: dict[str, pd.DataFrame], tables_dir: Path) -> pd.DataFrame:
    rows = []
    spec_by_key = {spec.key: spec for spec in SPECS}
    for key, df in data.items():
        spec = spec_by_key[key]
        missing = df[df["exposure"].isna()].copy()
        if missing.empty:
            continue
        missing.insert(0, "dataset", key)
        keep = ["dataset", *spec.id_cols]
        if spec.dynamic:
            keep.append("year")
        rows.append(missing[keep])
    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["dataset", "year"])
    out.to_csv(tables_dir / "missing_exposure_rows.csv", index=False, encoding="utf-8-sig")
    return out


def make_rank_tables(data: dict[str, pd.DataFrame], tables_dir: Path) -> dict[str, pd.DataFrame]:
    spec_by_key = {spec.key: spec for spec in SPECS}
    outputs = {}
    for key in BASE_KEYS:
        df = data[key].copy()
        spec = spec_by_key[key]
        df["label"] = label_for(df, spec)
        top = df.sort_values("exposure", ascending=False).head(20).copy()
        bottom = df.sort_values("exposure", ascending=True).head(20).copy()
        top.insert(0, "rank_type", "top")
        top.insert(1, "rank", np.arange(1, len(top) + 1))
        bottom.insert(0, "rank_type", "bottom")
        bottom.insert(1, "rank", np.arange(1, len(bottom) + 1))
        out = pd.concat([top, bottom], ignore_index=True)
        out.to_csv(tables_dir / f"{key}_top_bottom_20.csv", index=False, encoding="utf-8-sig")
        outputs[key] = out
    return outputs


def make_yearly_tables(data: dict[str, pd.DataFrame], tables_dir: Path) -> pd.DataFrame:
    rows = []
    spec_by_key = {spec.key: spec for spec in SPECS}
    for key in YEAR_KEYS:
        spec = spec_by_key[key]
        df = data[key]
        grouped = (
            df.groupby("year")["exposure"]
            .agg(nonmissing="count", rows="size", mean="mean", sd="std", min="min", max="max")
            .reset_index()
        )
        grouped.insert(0, "dataset", key)
        grouped.insert(1, "level_cn", spec.level_cn)
        rows.append(grouped)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(tables_dir / "yearly_unweighted_exposure_summary.csv", index=False, encoding="utf-8-sig")
    return out


def make_base_vs_dynamic_check(data: dict[str, pd.DataFrame], tables_dir: Path) -> pd.DataFrame:
    pairs = [
        ("minor", "base_minor", "year_minor", ("minor_soc_code",)),
        ("detail", "base_detail", "year_detail", ("occu_soc_code",)),
        ("zl", "base_zl", "year_zl", ("jd_class1", "jd_class2")),
    ]
    rows = []
    for level, base_key, year_key, keys in pairs:
        base = data[base_key]
        year = data[year_key]
        avg = (
            year[year["year"].between(2018, 2021)]
            .groupby(list(keys), dropna=False)["exposure"]
            .mean()
            .reset_index(name="avg_exposure_2018_2021")
        )
        merged = base.merge(avg, on=list(keys), how="left")
        merged["base_minus_simple_avg"] = merged["exposure"] - merged["avg_exposure_2018_2021"]
        rows.append(
            {
                "level": level,
                "matched_units": int(merged["avg_exposure_2018_2021"].notna().sum()),
                "units": len(merged),
                "mean_diff": merged["base_minus_simple_avg"].mean(),
                "median_diff": merged["base_minus_simple_avg"].median(),
                "max_abs_diff": merged["base_minus_simple_avg"].abs().max(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "base_vs_2018_2021_simple_average_check.csv", index=False, encoding="utf-8-sig")
    return out


def plot_base_distributions(data: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    labels = [
        ("base_minor", "SOC 职业组别"),
        ("base_detail", "SOC 详细职业"),
        ("base_zl", "智联二级职业"),
    ]
    for ax, (key, title) in zip(axes, labels):
        exposure = data[key]["exposure"].dropna()
        ax.hist(exposure, bins=14, color="#3b82f6", edgecolor="white", alpha=0.88)
        ax.axvline(exposure.mean(), color="#ef4444", linewidth=1.6, label=f"均值 {exposure.mean():.3f}")
        ax.set_title(title)
        ax.set_xlabel("AI-LLM 暴露指数")
        ax.set_ylabel("职业数")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("基期 AI-LLM 暴露指数分布（2018-2021）", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_01_base_exposure_distributions.png", bbox_inches="tight")
    plt.close(fig)


def plot_top_bottom(
    df: pd.DataFrame,
    spec: DatasetSpec,
    figures_dir: Path,
    file_name: str,
    n: int = 20,
) -> None:
    ranked = df.copy()
    ranked["label"] = label_for(ranked, spec)
    top = ranked.sort_values("exposure", ascending=False).head(n).sort_values("exposure")
    bottom = ranked.sort_values("exposure", ascending=True).head(n).sort_values("exposure", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, max(5.5, n * 0.28)), sharex=True)
    axes[0].barh(top["label"], top["exposure"], color="#2563eb")
    axes[0].set_title(f"最高 {n} 个")
    axes[1].barh(bottom["label"], bottom["exposure"], color="#64748b")
    axes[1].set_title(f"最低 {n} 个")
    for ax in axes:
        ax.set_xlabel("AI-LLM 暴露指数")
        ax.set_xlim(0, 1)
        ax.tick_params(axis="y", labelsize=7)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=2, fontsize=6)
    fig.suptitle(f"{spec.level_cn}基期暴露指数排序（2018-2021）", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / file_name, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_mean(yearly: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for level_cn, group in yearly.groupby("level_cn"):
        ax.plot(group["year"], group["mean"], marker="o", linewidth=2, label=level_cn)
    ax.set_title("年度平均 AI-LLM 暴露指数（非加权，公开指数口径）")
    ax.set_xlabel("年份")
    ax.set_ylabel("平均暴露指数")
    ax.set_xticks(sorted(yearly["year"].unique()))
    ax.legend(frameon=False)
    fig.text(
        0.01,
        -0.03,
        "注：这里是公开职业指数的非加权年度均值，不等同于论文图 3 的招聘岗位加权月度市场暴露趋势。",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_05_yearly_unweighted_mean_exposure.png", bbox_inches="tight")
    plt.close(fig)


def plot_minor_quartiles(data: dict[str, pd.DataFrame], figures_dir: Path) -> pd.DataFrame:
    base = data["base_minor"][["minor_soc_code", "title_chinese", "exposure"]].copy()
    base["base_quartile"] = pd.qcut(
        base["exposure"], 4, labels=["Q1 最低暴露", "Q2 次低暴露", "Q3 次高暴露", "Q4 最高暴露"]
    )
    year = data["year_minor"].merge(base[["minor_soc_code", "base_quartile"]], on="minor_soc_code", how="left")
    out = (
        year.groupby(["year", "base_quartile"], observed=False)["exposure"]
        .mean()
        .reset_index(name="mean_exposure")
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#10b981", "#84cc16", "#f59e0b", "#ef4444"]
    for color, (quartile, group) in zip(colors, out.groupby("base_quartile", observed=False)):
        ax.plot(group["year"], group["mean_exposure"], marker="o", linewidth=2, label=str(quartile), color=color)
    ax.set_title("基期四分位职业组的年度平均暴露指数")
    ax.set_xlabel("年份")
    ax.set_ylabel("平均暴露指数")
    ax.set_xticks(sorted(year["year"].unique()))
    ax.legend(frameon=False, ncol=2)
    fig.text(
        0.01,
        -0.04,
        "注：此图按基期暴露度分组后计算年度平均暴露指数；公开数据没有招聘市场份额，不能复现论文图 4 的市场占比变化。",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_06_minor_quartile_yearly_mean_exposure.png", bbox_inches="tight")
    plt.close(fig)
    return out


def plot_selected_minor_trends(data: dict[str, pd.DataFrame], figures_dir: Path) -> pd.DataFrame:
    selected = [
        "金融专家",
        "批发和制造业的销售代表",
        "计算机职业",
        "销售代表，服务",
        "信息和记录文员",
        "其他销售及相关工作人员",
        "广告、公共关系和销售经理",
        "业务运营专家",
        "艺术和设计工作者",
        "运营专业经理",
        "秘书和行政助理",
        "工程师",
        "其他管理职业",
        "机动车操作员",
        "其他生产职业",
        "装配工和制造工",
    ]
    df = data["year_minor"][data["year_minor"]["title_chinese"].isin(selected)].copy()
    order = (
        data["base_minor"]
        .set_index("title_chinese")
        .loc[[x for x in selected if x in set(data["base_minor"]["title_chinese"])], "exposure"]
        .sort_values(ascending=False)
        .index.tolist()
    )
    df["title_chinese"] = pd.Categorical(df["title_chinese"], categories=order, ordered=True)
    df = df.sort_values(["title_chinese", "year"])

    ncols = 4
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 8.8), sharex=True)
    axes_flat = axes.ravel()
    for ax, title in zip(axes_flat, order):
        group = df[df["title_chinese"] == title]
        ax.plot(group["year"], group["exposure"], marker="o", linewidth=1.6, color="#2563eb")
        ax.set_title(str(title), fontsize=9)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    for ax in axes_flat[len(order) :]:
        ax.axis("off")
    fig.suptitle("部分 SOC 职业组别年度 AI-LLM 暴露指数变化", y=1.02, fontsize=13)
    fig.text(
        0.01,
        -0.02,
        "注：公开数据为年度指数；论文图 7 使用测度样本构造月度/职位层面的职业内部暴露变化，两者口径不同。",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_07_selected_minor_occupation_trends.png", bbox_inches="tight")
    plt.close(fig)
    return df


def write_report(
    root: Path,
    data_dir: Path,
    inventory: pd.DataFrame,
    missing: pd.DataFrame,
    rank_tables: dict[str, pd.DataFrame],
    yearly: pd.DataFrame,
    base_check: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    top_minor = rank_tables["base_minor"].query("rank_type == 'top'").head(5)
    bottom_minor = rank_tables["base_minor"].query("rank_type == 'bottom'").head(5)
    top_detail = rank_tables["base_detail"].query("rank_type == 'top'").head(5)
    top_zl = rank_tables["base_zl"].query("rank_type == 'top'").head(5)

    def bullets(df: pd.DataFrame, name_col: str = "label") -> str:
        lines = []
        for _, row in df.iterrows():
            lines.append(f"- {row[name_col]}: {row['exposure']:.4f}")
        return "\n".join(lines)

    yearly_pivot = yearly.pivot(index="year", columns="level_cn", values="mean")
    report = f"""# 公开 AI-LLM 暴露指数复现报告

## 数据来源

本报告由 `reproduction/reproduce.py` 自动生成，读取的数据目录为：

`{data_dir.relative_to(root)}`

当前公开数据是职业层面的 AI-LLM 暴露指数。它不包含招聘数量、岗位薪资、学历要求、工作经验、城市标识、Felten 工具变量等字段，因此不能单独复现论文中的劳动需求回归表。

## 数据清单

{inventory[["dataset", "file_name", "level_cn", "rows", "unique_units", "missing_exposure", "years"]].to_markdown(index=False)}

## 可以复现的内容

1. 基期职业暴露指数的分布。
2. 基期高暴露和低暴露职业排序。
3. 2018-2024 年职业暴露指数的年度变化。
4. 按基期暴露四分位分组后的年度平均暴露变化。
5. 部分 SOC 职业组别的年度暴露趋势。

## 不能仅靠公开指数复现的论文内容

| 论文结果 | 是否能用当前公开数据直接复现 | 原因 |
|---|---|---|
| 图 3 新发布职位月度暴露趋势 | 不能精确复现 | 缺少岗位级招聘广告和月度招聘权重 |
| 图 4 暴露四分位的市场占比变化 | 不能 | 缺少职业年度/月度招聘市场份额 |
| 图 6 暴露度与新增职位占比变化 | 不能 | 缺少招聘需求变化变量 |
| 表 2 暴露度与学历、经验、薪资关系 | 不能 | 缺少岗位特征变量 |
| 表 3 职业层面 IV 回归 | 不能 | 缺少需求、薪资、学历、经验变化和 Felten 工具变量 |
| 表 4 城市层面回归 | 不能 | 缺少城市职业结构、城市控制变量和城市层面需求变化 |

## 基期 SOC 职业组别排名

最高暴露前 5：

{bullets(top_minor)}

最低暴露前 5：

{bullets(bottom_minor)}

## 基期 SOC 详细职业最高暴露前 5

{bullets(top_detail)}

## 基期智联二级职业最高暴露前 5

{bullets(top_zl)}

## 年度非加权平均暴露指数

注意：下表是公开职业指数的非加权均值，不等同于论文使用招聘广告市场份额加权得到的市场暴露趋势。

{yearly_pivot.round(4).to_markdown()}

## 基期指数与 2018-2021 年动态指数简单平均的关系

基期指数与 2018-2021 动态指数的简单均值非常接近，但不是完全相同，说明基期指数更可能是在基期 pooled 样本或带权口径下重新汇总得到。

{base_check.round(6).to_markdown(index=False)}

## 缺失值

公开动态指数中存在少量 `exposure` 缺失，明细已输出到：

`{(tables_dir / "missing_exposure_rows.csv").relative_to(root)}`

缺失行数：{len(missing)}

## 输出文件

主要表格输出目录：

`{tables_dir.relative_to(root)}`

主要图形输出目录：

`{figures_dir.relative_to(root)}`

建议优先查看以下图形：

- `fig_01_base_exposure_distributions.png`
- `fig_02_minor_top_bottom_10.png`
- `fig_03_soc_detail_top_bottom_20.png`
- `fig_04_zl_top_bottom_20.png`
- `fig_05_yearly_unweighted_mean_exposure.png`
- `fig_06_minor_quartile_yearly_mean_exposure.png`
- `fig_07_selected_minor_occupation_trends.png`
"""
    (root / "reproduction" / "outputs" / "reproduction_report.md").write_text(
        dedent(report).strip() + "\n", encoding="utf-8"
    )


def main() -> None:
    root = project_root()
    data_dir = find_data_dir(root)
    dirs = ensure_dirs(root)
    configure_plots()

    data = load_data(data_dir)
    inventory = make_inventory(data, dirs["tables"])
    missing = make_missing_table(data, dirs["tables"])
    rank_tables = make_rank_tables(data, dirs["tables"])
    yearly = make_yearly_tables(data, dirs["tables"])
    base_check = make_base_vs_dynamic_check(data, dirs["tables"])

    spec_by_key = {spec.key: spec for spec in SPECS}
    plot_base_distributions(data, dirs["figures"])
    plot_top_bottom(data["base_minor"], spec_by_key["base_minor"], dirs["figures"], "fig_02_minor_top_bottom_10.png", n=10)
    plot_top_bottom(data["base_detail"], spec_by_key["base_detail"], dirs["figures"], "fig_03_soc_detail_top_bottom_20.png", n=20)
    plot_top_bottom(data["base_zl"], spec_by_key["base_zl"], dirs["figures"], "fig_04_zl_top_bottom_20.png", n=20)
    plot_yearly_mean(yearly, dirs["figures"])
    quartile = plot_minor_quartiles(data, dirs["figures"])
    quartile.to_csv(dirs["tables"] / "minor_base_quartile_yearly_mean_exposure.csv", index=False, encoding="utf-8-sig")
    selected = plot_selected_minor_trends(data, dirs["figures"])
    selected.to_csv(dirs["tables"] / "selected_minor_occupation_yearly_trends.csv", index=False, encoding="utf-8-sig")

    write_report(
        root=root,
        data_dir=data_dir,
        inventory=inventory,
        missing=missing,
        rank_tables=rank_tables,
        yearly=yearly,
        base_check=base_check,
        tables_dir=dirs["tables"],
        figures_dir=dirs["figures"],
    )

    print("Reproduction complete.")
    print(f"Tables:  {dirs['tables']}")
    print(f"Figures: {dirs['figures']}")
    print(f"Report:  {dirs['outputs'] / 'reproduction_report.md'}")


if __name__ == "__main__":
    main()
