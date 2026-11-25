# Data Science Comprehensive Review Handbook

![Study handbook](assets/overview.svg)

繁體中文觀念教學 · English assessment terminology · Python / pandas / NumPy / scikit-learn

## How to study

每個 Lesson 按「核心觀念 → 範例及推理 → 改變條件的 variation → 英文 concept check 與中文解析」安排。目標是理解、回想與遷移，不是把大量 API 或考題塞在同一頁。可直接展開答案；限時練習另列為選用內容。

| Chapter | Lessons | Focus |
|---|---:|---|
| 🔵 [01 pandas](01-pandas.md) | 8 | Filtering, GroupBy, Merge, Cleaning, Features, Reshaping/Window, DateTime, Integration |
| 🟢 [02 Preprocessing](02-preprocessing.md) | 4 | Missing/outliers, scaling/encoding, feature selection/imbalance, Pipeline |
| 🟣 [03 Model Development & Evaluation](03-models.md) | 5 | Validation, classification metrics, regression, tuning, clustering/calibration |
| 🟠 [04 ML Fundamentals](04-ml.md) | 15 | Learning paradigms, linear/logistic regression, optimization, trees/ensembles, KNN/SVM/NB, clustering/PCA, neural nets, generalization and drift |
| 🔴 [05 Probability & Statistics](05-statistics.md) | 6 | Probability, Bayes, distributions, expectation, inference, testing/A/B |
| 🔵 [06 Integrated Worked Cases](06-cases.md) | 3 | Merchant analysis, churn classifier, house price regression |
| ⚪ [07 Cheat Sheet](07-review.md) | reference | Quick syntax, metrics and formulas |
| 🟢 [08 Python / NumPy & Multi-file Data](08-numpy-data.md) | 2 | Array operations, schema normalization, multiple logical tables |
| 🟣 [09 Timed Mock Assessment (optional)](09-mock.md) | optional | Original synthetic mock, 90 minutes |

## Reading depth

- Core: 常用方法、重要公式、運作原理、常用參數及可讀範例。
- Variations: 改變假設、分母、資料粒度、資料分布、評估指標或資料切分方式時，如何修正推理。
- Extensions: 重要但非每題必用的知識；冷門 API、底層實作及非必要證明簡述即可。

已熟悉基礎者可直接從 01–05 逐章閱讀；機率統計可按順序從第一節開始。08 是 Python/NumPy 及多檔資料處理的補充；06 提供完整可閱讀案例。

## Assessment-focused ML review

[ML Assessment Reasoning](ml/assessment-reasoning.md) 整合 generalization、ensemble、AUC、log loss、binary classification 與 code interpretation。原創英文多選題逐項解析；不是正式題庫或對考題頻率的保證。

## Assessment-style workflow

- [Multi-file data integration](python/02-multi-file.md)：多份 CSV → 正規化 → 各邏輯表清理 → 彙總及合併。
- [House price regression](cases/03-house-price.md)：既定 70/15/15 split、無標籤 test、自由選模型、validation MAE、輸出預測。提供 [已執行的合成資料範例](solutions/review_house_price.py)。
- [Classification and threshold constraints](cases/02-churn.md)：原創分類變化案例，非房價題的同一任務。

> [!NOTE]
> 本手冊使用原創或合成資料，僅將常見 assessment-style 技能融入教材，不代表官方考試題庫、配分或錄取標準。實際提交欄位與環境依題意。房價合成腳本已執行檢查；其餘獨立程式片段未逐一執行驗證。
