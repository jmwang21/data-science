# 01 — Data Collection / pandas

![Chapter 01](assets/01-pandas.svg)

[← 返回總目錄](README.md)

本章以八個獨立 Lesson 進行複習。每個 Lesson 先介紹核心觀念，再使用小型 Dataset 示範語法及輸出，最後提供少量英文 Concept Check 與可展開的中文解析。可直接閱讀答案，不需限時作答。

## Learning path

| Lesson | Main topic | Review priority |
|---|---|---|
| [01.1 Filtering & Ranking](pandas/01-filtering-ranking.md) | 選取、條件、排序、群組排名 | 核心 |
| [01.2 GroupBy & Aggregation](pandas/02-groupby.md) | agg、transform、count、conditional aggregation | 核心 |
| [01.3 Merge, Join & Concat](pandas/03-merge.md) | join types、grain、cardinality、concat | 核心 |
| [01.4 Missing Values & Cleaning](pandas/04-cleaning.md) | missing、type conversion、duplicates、strings | 核心 |
| [01.5 Feature Engineering](pandas/05-features.md) | growth、ratio、flags、map/apply | 核心 |
| [01.6 Reshaping & Window](pandas/06-reshape-window.md) | pivot_table、melt、shift、rolling、cumsum | Window 核心；reshape 常用模板 |
| [01.7 DateTime](pandas/07-datetime.md) | parsing、date range、month、tenure | 核心 |
| [01.8 Integrated pandas](pandas/08-integrated.md) | 完整商家分析及 feature table | 核心 |

## What to focus on

掌握常用 API 的輸入、輸出粒度、使用時機以及最重要的參數，比背下所有 API 更重要。stack/unstack、複雜 MultiIndex、進階 regex 只需辨認用途。若已熟悉基本 Python，可直接從 Lesson 01.1 開始；多檔案整合與 NumPy 另見 [08](08-numpy-data.md)。

本教材為通用資料科學複習與原創示例，不代表 CodeSignal 官方題庫。程式片段未逐一執行驗證。

[開始 Lesson 01.1 →](pandas/01-filtering-ranking.md) · [下一章：Preprocessing →](02-preprocessing.md)
