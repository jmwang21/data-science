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

Bias–variance decomposition under squared loss: expected prediction error includes squared bias, variance and irreducible noise. More capacity may reduce bias and increase variance; additional representative data can reduce variance. Learning curves help distinguish high bias (both errors high) from variance (large gap), but distribution shift can also create a gap.

For drift, distinguish covariate shift P(X), label shift P(Y), and concept drift P(Y|X). Unlabeled data can reveal feature/score distribution changes but cannot directly establish actual accuracy. Monitoring should compare relevant cohorts and obtain delayed labels when possible.

<details><summary>Concept check: why does adding trees differ from increasing tree depth?</summary>

More trees reduce Monte Carlo variability of the ensemble estimate; greater depth increases individual tree capacity. They affect complexity and generalization differently.

</details>



## Interpreting changes in deployment

Covariate shift: P(X) changes with P(Y|X) stable; label shift: P(Y) changes with P(X|Y) stable; concept drift: P(Y|X) changes. Selection bias refers to nonrepresentative data acquisition.

Without labels, can monitor feature/score distribution, missingness, schema, cohort mix, PSI or KS statistics. Cannot directly measure accuracy or prove model degradation. Once labels arrive, evaluate performance by cohort and time. Drift metric thresholds are context dependent.

Feature importance: tree impurity reduction may favor high-cardinality features; permutation importance measures performance drop when shuffling, can understate correlated features; SHAP attributes a model prediction relative to a baseline, not causal proof.

<details><summary>Variation: In a new market fraud prevalence doubles, but TPR and FPR stay fixed. Which metric must be reconsidered?</summary>

Precision changes because its denominator depends on prevalence; ROC point may stay fixed. Calibration and business threshold may need reevaluation.

</details>

<details><summary>Variation: A feature has high SHAP values. Can removing that business factor be claimed to improve outcomes?</summary>

No. Model attribution is not intervention effect. Need causal design or experiment.

</details>

[← Chapter index](../04-ml.md)
