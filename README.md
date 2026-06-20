# AI-LLM Exposure 复现与扩展代码


本项目为复旦大学《社会科学数据挖掘》课程的项目报告主要为对文章《中国人工智能技术暴露度的测算  及其对劳动需求的影响》复现与扩展。
本仓库保存课程项目中用于复现与扩展“中国 AI-LLM 职业暴露度”研究的代码。仓库仅提交代码和必要说明文件；

## 仓库结构

```text
Reproduction/
  code/
    run_exposure_reproduction.py        # 公开职业暴露指数的复现分析
  reproduction-reference/               # 参考复现代码

basic-expansion/
  code/
    run_all_experiments.py              # 基础扩展实验的一键运行入口
  2.1-structure-anova/code/
    run_anova.py                        # 职业暴露度结构分化与 ANOVA
  2.2-dynamic-panel/code/
    run_dynamic_panel.py                # 年度暴露度趋势、固定效应与收敛检验
  2.3-clustering/code/
    run_kmeans.py                       # 暴露水平与动态特征的 K-means 聚类

CGSS/
  code/
    read_cgss_overview.py               # CGSS 数据初步读取与变量审查
    audit_crosswalk_cleaning.py         # 职业 crosswalk 与清洗审查

Data cleaning/
  clean_cgss_ai_exposure.py             # CGSS、ISCO-SOC 映射与 AI-LLM 暴露度合并

CGSS-expansion/
  code/
    run_cgss_soc_analysis.py            # CGSS 中 AI-LLM 暴露度、就业份额与收入分析
```

## 数据说明

代码依赖以下数据，但数据文件不随仓库提交：

- 中国人工智能-大语言模型技术暴露指数数据；
- CGSS 2018、2021、2023 年调查数据；
- ISCO-08 与 SOC 职业编码 crosswalk；
- 项目中生成的中间清洗数据、结果表和图表。

如需复现实验，请在本地按照代码中的相对路径放置数据文件，并先运行数据清洗脚本，再运行复现、基础扩展和 CGSS 扩展分析脚本。

## 运行顺序

建议按以下顺序运行：

1. `Reproduction/code/run_exposure_reproduction.py`
2. `basic-expansion/code/run_all_experiments.py`
3. `Data cleaning/clean_cgss_ai_exposure.py`
4. `CGSS-expansion/code/run_cgss_soc_analysis.py`

CGSS 相关分析依赖职业映射和清洗结果，因此应在数据清洗完成后再运行。  

## 说明

本仓库中的模型包括 ANOVA、职业固定效应趋势模型、Beta 收敛检验、K-means 聚类、分数 Logit、PPML、Double/Debiased Machine Learning 和 RIF 分位数回归。相关报告正文、与附录同样附在仓库中。
