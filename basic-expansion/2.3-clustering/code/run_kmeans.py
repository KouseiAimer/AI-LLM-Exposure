"""2.3 职业动态类型识别：K-means 聚类。"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
PART_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "人工智能-大语言模型技术”暴露指数数据"
TABLE_DIR = PART_DIR / "results" / "tables"
FIGURE_DIR = PART_DIR / "results" / "figures"


SPECS = {
    "SOC职业组别": {
        "base_file": "exposure_base_minor_soc.xlsx",
        "year_file": "exposure_by_year_minor_soc.xlsx",
        "id_cols": ["minor_soc_code"],
        "label_cols": ["title_chinese"],
    },
    "SOC详细职业": {
        "base_file": "exposure_base_soc_detail.xlsx",
        "year_file": "exposure_by_year_soc_detail.xlsx",
        "id_cols": ["occu_soc_code"],
        "label_cols": ["onet_occupationtitle"],
    },
    "智联二级职业": {
        "base_file": "exposure_base_zl_occu.xlsx",
        "year_file": "exposure_by_year_zl_occu.xlsx",
        "id_cols": ["jd_class1", "jd_class2"],
        "label_cols": ["jd_class1", "jd_class2"],
    },
}

FEATURES = ["base_exposure", "change_2018_2024", "slope_per_year", "volatility", "mean_dynamic"]
K_MIN = 2
K_MAX = 8


def setup() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def join_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg(" | ".join, axis=1)


def label_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg(" / ".join, axis=1)


def build_features() -> pd.DataFrame:
    parts = []
    for level, spec in SPECS.items():
        base = pd.read_excel(DATA_DIR / spec["base_file"])
        year = pd.read_excel(DATA_DIR / spec["year_file"])
        base["unit_id"] = join_cols(base, spec["id_cols"])
        base["unit_label"] = label_cols(base, spec["label_cols"])
        year["unit_id"] = join_cols(year, spec["id_cols"])
        rows = []
        for unit_id, group in year.dropna(subset=["exposure"]).groupby("unit_id"):
            group = group.sort_values("year")
            exposure_2018 = group.loc[group["year"] == 2018, "exposure"]
            exposure_2024 = group.loc[group["year"] == 2024, "exposure"]
            if exposure_2018.empty or exposure_2024.empty or len(group) < 5:
                continue
            slope, _ = np.polyfit(group["year"] - 2018, group["exposure"], 1)
            rows.append(
                {
                    "unit_id": unit_id,
                    "exposure_2018": float(exposure_2018.iloc[0]),
                    "exposure_2024": float(exposure_2024.iloc[0]),
                    "change_2018_2024": float(exposure_2024.iloc[0] - exposure_2018.iloc[0]),
                    "slope_per_year": float(slope),
                    "volatility": float(group["exposure"].std()),
                    "mean_dynamic": float(group["exposure"].mean()),
                    "n_years": int(len(group)),
                }
            )
        dyn = pd.DataFrame(rows)
        merged = base[["unit_id", "unit_label", "exposure"]].rename(columns={"exposure": "base_exposure"}).merge(
            dyn, on="unit_id", how="inner"
        )
        merged.insert(0, "level", level)
        parts.append(merged)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(TABLE_DIR / "clustering_feature_table.csv", index=False, encoding="utf-8-sig")
    return out


def cluster_label(row: pd.Series) -> str:
    exposure = row["base_exposure"]
    change = row["change_2018_2024"]
    if exposure >= 0.70:
        level = "高暴露"
    elif exposure <= 0.45:
        level = "低暴露"
    else:
        level = "中等暴露"
    if change <= -0.15:
        direction = "急剧下降型"
    elif change <= -0.025:
        direction = "下降型"
    elif change >= 0.025:
        direction = "上升型"
    else:
        direction = "稳定型"
    return level + direction


def min_cluster_threshold(n_units: int) -> int:
    """Avoid choosing a K that creates nearly empty clusters."""
    return max(3, int(np.ceil(0.05 * n_units)))


def evaluate_k_grid(level: str, x_scaled: np.ndarray, n_units: int) -> pd.DataFrame:
    rows = []
    threshold = min_cluster_threshold(n_units)
    for kk in range(K_MIN, min(K_MAX, n_units - 1) + 1):
        km = KMeans(n_clusters=kk, random_state=42, n_init=50)
        labels = km.fit_predict(x_scaled)
        counts = np.bincount(labels)
        rows.append(
            {
                "level": level,
                "k": kk,
                "n_units": n_units,
                "silhouette": silhouette_score(x_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(x_scaled, labels),
                "davies_bouldin": davies_bouldin_score(x_scaled, labels),
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
                "min_cluster_threshold": threshold,
                "eligible": bool(counts.min() >= threshold),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["silhouette_rank"] = metrics.groupby("level")["silhouette"].rank(ascending=False, method="min")
    metrics["calinski_rank"] = metrics.groupby("level")["calinski_harabasz"].rank(ascending=False, method="min")
    metrics["davies_rank"] = metrics.groupby("level")["davies_bouldin"].rank(ascending=True, method="min")
    metrics["average_rank"] = metrics[["silhouette_rank", "calinski_rank", "davies_rank"]].mean(axis=1)
    return metrics


def choose_k(metrics: pd.DataFrame) -> pd.Series:
    eligible = metrics[metrics["eligible"]].copy()
    if eligible.empty:
        eligible = metrics.copy()
    # Silhouette is the primary criterion because it directly evaluates cluster separation;
    # CH and DB are retained for diagnosis rather than forced into a brittle vote.
    chosen = eligible.sort_values(
        ["silhouette", "average_rank", "min_cluster_size", "k"],
        ascending=[False, True, False, True],
    ).iloc[0].copy()
    chosen["selection_rule"] = "最大轮廓系数；剔除低于最小簇规模阈值的 K；CH/DB 用于复核"
    return chosen


def run_kmeans(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_assignments = []
    profiles = []
    diagnostics = []
    selected_rows = []
    for level, df in features.groupby("level"):
        x = df[FEATURES].dropna().copy()
        df_model = df.loc[x.index].copy()
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        metric = evaluate_k_grid(level, x_scaled, len(df_model))
        chosen = choose_k(metric)
        k = int(chosen["k"])
        diagnostics.append(metric)
        selected_rows.append(chosen)

        km = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = km.fit_predict(x_scaled)
        df_model["cluster"] = labels
        df_model["selected_k"] = k

        profile = df_model.groupby("cluster", as_index=False)[FEATURES].mean()
        profile.insert(0, "level", level)
        profile["selected_k"] = k
        profile["n_units"] = df_model.groupby("cluster")["unit_id"].count().values
        profile["cluster_type"] = profile.apply(cluster_label, axis=1)
        df_model = df_model.merge(profile[["level", "cluster", "cluster_type"]], on=["level", "cluster"], how="left")

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(x_scaled)
        df_model["pc1"] = coords[:, 0]
        df_model["pc2"] = coords[:, 1]
        df_model["pca_explained_var_1"] = pca.explained_variance_ratio_[0]
        df_model["pca_explained_var_2"] = pca.explained_variance_ratio_[1]
        all_assignments.append(df_model)
        profiles.append(profile)

    assignments = pd.concat(all_assignments, ignore_index=True)
    profiles_df = pd.concat(profiles, ignore_index=True)
    diagnostics_df = pd.concat(diagnostics, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    assignments.to_csv(TABLE_DIR / "occupation_clusters.csv", index=False, encoding="utf-8-sig")
    profiles_df.to_csv(TABLE_DIR / "cluster_profiles.csv", index=False, encoding="utf-8-sig")
    diagnostics_df.to_csv(TABLE_DIR / "silhouette_scores.csv", index=False, encoding="utf-8-sig")
    diagnostics_df.to_csv(TABLE_DIR / "kmeans_diagnostics.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(TABLE_DIR / "cluster_selection.csv", index=False, encoding="utf-8-sig")
    return assignments, profiles_df, diagnostics_df, selected_df


def plot_silhouette(silhouettes: pd.DataFrame, selected: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.lineplot(data=silhouettes, x="k", y="silhouette", hue="level", marker="o", ax=ax)
    sns.scatterplot(
        data=selected,
        x="k",
        y="silhouette",
        hue="level",
        marker="X",
        s=170,
        edgecolor="black",
        linewidth=0.8,
        legend=False,
        ax=ax,
    )
    ax.set_title("不同 K 值的轮廓系数")
    ax.set_xlabel("聚类数量 K")
    ax.set_ylabel("轮廓系数")
    ax.legend(title="职业层级")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_2_3_silhouette_scores.png", dpi=300)
    plt.close(fig)


def plot_pca(assignments: pd.DataFrame) -> None:
    levels = list(assignments["level"].unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(5.3 * len(levels), 5), sharex=False, sharey=False)
    if len(levels) == 1:
        axes = [axes]
    for ax, level in zip(axes, levels):
        sub = assignments[assignments["level"] == level]
        sns.scatterplot(data=sub, x="pc1", y="pc2", hue="cluster_type", style="cluster", s=65, alpha=0.82, ax=ax)
        v1 = sub["pca_explained_var_1"].iloc[0]
        v2 = sub["pca_explained_var_2"].iloc[0]
        ax.set_title(f"{level} 聚类二维投影")
        ax.set_xlabel(f"主成分 1（解释 {v1:.1%}）")
        ax.set_ylabel(f"主成分 2（解释 {v2:.1%}）")
        ax.legend(title="聚类类型", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_2_3_cluster_pca.png", dpi=300)
    plt.close(fig)


def plot_cluster_profiles(profiles: pd.DataFrame) -> None:
    profile_long = profiles.melt(
        id_vars=["level", "cluster", "cluster_type", "n_units"],
        value_vars=FEATURES,
        var_name="feature",
        value_name="value",
    )
    feature_names = {
        "base_exposure": "基期暴露度",
        "change_2018_2024": "2018-2024变化",
        "slope_per_year": "年度斜率",
        "volatility": "波动性",
        "mean_dynamic": "年度均值",
    }
    profile_long["feature_cn"] = profile_long["feature"].map(feature_names)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
    for ax, (level, sub) in zip(axes, profile_long.groupby("level")):
        sns.barplot(data=sub, x="feature_cn", y="value", hue="cluster_type", ax=ax)
        ax.set_title(f"{level} 聚类特征画像")
        ax.set_xlabel("")
        ax.set_ylabel("特征均值")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="聚类类型", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_2_3_cluster_profiles.png", dpi=300)
    plt.close(fig)


def write_report(assignments: pd.DataFrame, profiles: pd.DataFrame, silhouettes: pd.DataFrame, selected: pd.DataFrame) -> None:
    selected_show = selected[
        [
            "level",
            "k",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "min_cluster_size",
            "min_cluster_threshold",
            "selection_rule",
        ]
    ].copy()
    diagnostics_show = silhouettes[
        ["level", "k", "silhouette", "calinski_harabasz", "davies_bouldin", "min_cluster_size", "eligible"]
    ].copy()
    lines = [
        "# 2.3 职业动态类型识别：K-means 聚类",
        "",
        "## 研究问题",
        "",
        "前两节分别说明了 AI-LLM 暴露度具有职业结构分化，并且存在年度动态变化。本节进一步将“暴露度水平”和“暴露度变化”合并，识别不同职业的动态类型。这样可以避免只用高低暴露二分法，而是形成更细致的职业画像。",
        "",
        "## 模型设定与合理性",
        "",
        "本节对每个职业构造五个特征：",
        "",
        "- 基期暴露度；",
        "- 2018-2024 年暴露度变化；",
        "- 年度线性斜率；",
        "- 年度波动性；",
        "- 年度平均暴露度。",
        "",
        "随后对这些特征标准化，并使用 K-means 聚类。K-means 适合本节，是因为它能够把多维连续特征压缩成若干可解释的职业类型。与旧版统一预设 K=4 不同，本轮不再把三个职业层级强行放在相同簇数下，而是对 SOC 职业组别、SOC 详细职业和智联二级职业分别选择 K 值，再将三套聚类结果纵向合并。",
        "",
        "K 值选择遵循三步规则。第一，在每个职业层级内部考察 K=2 至 K=8。第二，剔除最小簇规模低于阈值的 K，阈值设为 `max(3, 5%×该层级职业数)`，避免出现只有一两个职业的“孤立簇”。第三，在剩余 K 中选择轮廓系数最高者；Calinski-Harabasz 指数和 Davies-Bouldin 指数作为复核指标，而不是直接替代轮廓系数。",
        "",
        "## 最佳 K 选择",
        "",
        selected_show.round(4).to_markdown(index=False),
        "",
        "## K 值诊断表",
        "",
        diagnostics_show.round(4).to_markdown(index=False),
        "",
        "## 聚类画像",
        "",
        profiles[["level", "selected_k", "cluster", "cluster_type", "n_units", *FEATURES]].round(4).to_markdown(index=False),
        "",
        "## 各层级代表性职业",
    ]
    for level in assignments["level"].unique():
        lines.append("")
        lines.append(f"### {level}")
        sub = assignments[assignments["level"] == level]
        for cluster_type, group in sub.groupby("cluster_type"):
            examples = group.sort_values("base_exposure", ascending=False).head(5)["unit_label"].tolist()
            lines.append(f"- **{cluster_type}**：{'；'.join(examples)}")
    lines.extend(
        [
            "",
            "## 结论解释",
            "",
            "K-means 结果表明，不同职业层级的数据结构并不相同，因此不应机械地统一使用 K=4。SOC 职业组别的最佳 K 为 3，能够区分低暴露型、中等暴露下降型和较高暴露稳定型；SOC 详细职业和智联二级职业的最佳 K 均为 2，说明在当前五维特征下，两个层级最稳定的结构首先体现为高暴露与低/中暴露两大类，而继续细分会明显降低轮廓系数或产生过小簇。",
            "",
            "这一结论把 2.1 的职业结构分化和 2.2 的动态趋势整合起来：AI-LLM 暴露度不仅有结构差异，也有不同的职业演化路径。后续接入 CGSS 时，可以优先使用按层级最佳 K 得到的聚类类型；若研究需要更细的职业画像，可把旧版 K=4 作为解释性敏感性方案，而不应把它称为统计意义上的最佳方案。",
            "",
            "## 输出文件",
            "",
            "- `results/tables/clustering_feature_table.csv`",
            "- `results/tables/occupation_clusters.csv`",
            "- `results/tables/cluster_profiles.csv`",
            "- `results/tables/silhouette_scores.csv`",
            "- `results/tables/kmeans_diagnostics.csv`",
            "- `results/tables/cluster_selection.csv`",
            "- `results/figures/fig_2_3_silhouette_scores.png`",
            "- `results/figures/fig_2_3_cluster_pca.png`",
            "- `results/figures/fig_2_3_cluster_profiles.png`",
        ]
    )
    (PART_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    features = build_features()
    assignments, profiles, silhouettes, selected = run_kmeans(features)
    plot_silhouette(silhouettes, selected)
    plot_pca(assignments)
    plot_cluster_profiles(profiles)
    write_report(assignments, profiles, silhouettes, selected)
    print("2.3 K-means finished.")
    print(f"Tables: {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
