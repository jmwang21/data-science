# 04.1 — Learning Paradigms & Problem Formulation

[← ML index](../04-ml.md) · [Next →](linear-regression.md)

## 1. What is the learning task?

先確定預測目標、可用資料、輸出、實際部署情境與評估方式，再選 model。Supervised learning 學習 X → y 的關係；unsupervised learning 不提供 target，找資料結構。

| Paradigm | Training signal | Example |
|---|---|---|
| Supervised | Labeled examples | Regression: house price; classification: churn |
| Unsupervised | No target labels | Clustering, PCA |
| Semi-supervised | Few labeled + many unlabeled rows | Product classification with scarce labels |
| Self-supervised | Targets derived from the input itself | Predict masked tokens |
| Reinforcement learning | Reward from interaction | Sequential decision making |

Self-supervised 不等於人工標籤；semi-supervised 可利用 pseudo-labels，但錯誤 pseudo-labels 可能強化錯誤。RL 的 action 影響 future state/reward，不只是用固定 labels 學習。

## 2. Regression, classification and ranking

Regression predicts a continuous value. Classification predicts a category or its probability. Ranking orders candidates by relevance or expected value; a ranking score need not be calibrated.

例如將有流失風險的商家排序，若只能聯繫 100 人，precision@100 可能比 accuracy 更貼近營運目標；若要估計預期收益，機率 calibration 及效益估計也重要。

## 3. Data and target definition

Define an observation unit, prediction cutoff, feature window, label window and deployment population. 如果只有已採用廣告商家有 outcome，學到的關係未必適用所有商家。這是 selection bias，不是多收同樣類型樣本就會解決。

## Concept variations

<details><summary>1. You have one million unlabeled customer records and need to discover behavioral segments. Which learning paradigm?</summary>

Unsupervised learning. 不應先假設 cluster label 等於 churn probability；仍要確認分群穩定性及商業意義。

</details>

<details><summary>2. Only 2% of product listings have verified categories. Can semi-supervised learning help?</summary>

Potentially. It uses labeled and unlabeled examples, but distribution mismatch and noisy pseudo-labels need validation.

</details>

<details><summary>3. A historical dataset has prices for all houses. Should the task always be formulated as regression?</summary>

No. 若需求是預測實際價格，用 regression；若只預測是否超過指定價格，可定義 binary classification。Objective 由題意決定。

</details>
