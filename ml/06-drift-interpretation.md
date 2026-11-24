# 04.6 — Domain Shift, Model Interpretation & Selection

[← Chapter index](../04-ml.md) · [Previous](05-generalization.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## Domain shift & monitoring 

- Covariate shift：P(X) 改變；在嚴格定義下 P(Y|X) 不變。
- Label shift：P(Y) 改變且 P(X|Y) 不變。
- Concept shift：P(Y|X) 改變。
- Selection bias：樣本取得方式造成樣本與目標母體不同。

無 ground truth 時可監控 missingness、schema、feature/prediction distribution、out-of-range、prediction confidence、人工抽樣；無法直接知道真實 AUC、precision 或 recall。KS test 比較一維 empirical CDF 最大差；p-value 受樣本量影響。PSI = Σ(actual-expected) ln(actual/expected)，先固定 bins、處理零比例；0.1/0.25 只是常見經驗界線，不是普遍標準。Drift 不必然表示 performance degradation，需補 labels 或實驗驗證。

## Practical algorithm selection (P0)

| Situation | First candidates | Main tradeoff |
|---|---|---|
| Interpretable continuous target | OLS, Ridge, Lasso | linear structure vs nonlinear bias |
| Sparse text classification | LogisticRegression, MultinomialNB | scalable sparse representations |
| Mixed tabular nonlinear data | RandomForest, GradientBoosting | stronger interactions vs complexity |
| Small scaled nonlinear dataset | SVM, KNN | kernel/distance sensitivity |
| Customer segmentation | KMeans, hierarchical, DBSCAN | shape, density, scaling |
| Reduce correlated features | PCA | variance preservation vs interpretability |

## Interpreting changes in deployment

| Scenario | Likely shift to investigate |
|---|---|
| Customer income distribution changes, stable buying relation | Covariate shift |
| Fraud prevalence changes, stable feature distributions conditional on fraud status | Label shift |
| Same observed transaction features now imply higher fraud risk | Concept drift |

These categories need assumptions; observing P(X) changes alone does not identify the shift mechanism. Once labels arrive, evaluate performance by cohort and time.

Feature importance: tree impurity reduction may favor high-cardinality features; permutation importance measures performance drop when shuffling, can understate correlated features; SHAP attributes a model prediction relative to a baseline, not causal proof.

<details><summary>Variation: In a new market fraud prevalence doubles, but TPR and FPR stay fixed. Which metric must be reconsidered?</summary>

Precision changes because its denominator depends on prevalence; ROC point may stay fixed. Calibration and business threshold may need reevaluation.

</details>

<details><summary>Variation: A feature has high SHAP values. Can removing that business factor be claimed to improve outcomes?</summary>

No. Model attribution is not intervention effect. Need causal design or experiment.

</details>

[← Chapter index](../04-ml.md)
