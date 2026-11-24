# 04.2 — Decision Trees, Random Forest & Boosting

[← Chapter index](../04-ml.md) · [Previous](01-learning-linear.md) · [Next →](03-distance-margin-bayes.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## Decision Tree 

Classification 使用 Gini $1-\sum p_k^2$ 或 entropy $-\sum p_k\log_2 p_k$ 衡量 impurity；尋找最大 weighted impurity reduction。Regression tree 常用 squared error。

例：節點 5 正 / 5 負，Gini=.5。切成兩節點各 5 筆，4P1N 與 1P4N，Gini 都 .32；gain=.18。

| Parameter | 增加的典型效果 |
|---|---|
| max_depth | 複雜度增加，可能過擬合 |
| min_samples_leaf | 限制小葉節點，通常降低 variance |
| min_samples_split | 限制分裂，通常降低複雜度 |

優點：非線性、交互作用、不需 scaling。缺點：單棵樹 high variance、外插差。Impurity importance 偏好高基數特徵，且相關特徵會分攤重要性。

## Ensemble Learning — comparison & multi-select 

| Method | Training | Main intuition | Pitfall |
|---|---|---|---|
| Bagging | independent models on bootstrap samples | average to reduce variance | correlated errors limit gain |
| Random Forest | bagged trees + random feature subsets | decorrelate trees | not sequential residual fitting |
| Boosting | sequential weak learners | improve current loss | can overfit; tune depth, rate, iterations |
| Hard voting | trained classifiers vote on labels | majority decision | ignores confidence |
| Soft voting | average class probabilities | combine probability estimates | calibration and class order matter |
| Stacking | meta-model combines out-of-fold base predictions | learn combination | in-sample meta-features cause leakage |

If train accuracy is 98% and validation accuracy is 65%, high variance is plausible, but distribution shift, invalid splits and noisy validation labels also deserve investigation. More trees in a random forest are not the same as deeper trees. A high-capacity single tree may overfit; restricting depth or increasing min_samples_leaf can help.

**Select all true:** A. Random forests use randomized feature selection. B. Boosting always trains independent models. C. Stacking should use out-of-fold training predictions for the meta-model. D. A train/validation gap proves overfitting in every case.

<details><summary>Answer & concept correction</summary>

A, C. B is false because boosting is sequential. D is too absolute: distribution changes and evaluation issues can also cause the gap.

</details>

## Random Forest — why randomness helps

Bootstrap n draws with replacement leaves about 36.8% of training observations out of each tree for large n. At each split, randomly select candidate features. Averaging decorrelated trees reduces variance: under equal variance v and correlation rho, average variance is v[rho+(1−rho)/M]. More trees cannot remove shared errors. OOB score estimates performance using trees that did not train on each observation, but repeated tuning on OOB can overfit the estimate.

| Parameter | Effect |
|---|---|
| n_estimators | stabilizes average, raises computation |
| max_depth | higher increases single-tree capacity |
| min_samples_leaf | higher smooths predictions |
| max_features | lower reduces correlation but may weaken trees |
| bootstrap | sampling with replacement, enables OOB |

max_features='sqrt' ≈ square root of count, 'log2' ≈ log2, float proportion, int count. Common sklearn RF classifier default sqrt, regressor 1.0. Regression averages numeric predictions; sklearn classification averages probabilities and takes argmax.

## Gradient Boosting — sequential loss reduction

Squared-error example: y=[10,20,30]; initial mean=20; residual=[−10,0,10]. New tree predicts [−8,0,8]. With learning_rate=.5, ensemble predictions become [16,20,24], residual [−6,0,6]. Other differentiable losses fit negative gradient rather than raw residuals.

Learning_rate and n_estimators interact. Small learning rate often requires more trees. Depth/leaf size controls base learners; subsample can regularize; early stopping uses validation. XGBoost, LightGBM, CatBoost are gradient boosting variants with implementation differences; focus on principles, not obscure configuration.

<details><summary>Variation: Which model can train trees largely in parallel, Random Forest or Gradient Boosting? Why?</summary>

Random Forest. Trees are trained independently given sampled data/randomness. Boosting's next learner depends on current ensemble predictions.

</details>

<details><summary>Variation: Is decreasing max_features guaranteed to improve validation performance?</summary>

No. It may decorrelate trees but remove informative split candidates and increase individual tree error. Validate tradeoff.

</details>

<details><summary>Variation: Why can stacking on in-sample base predictions produce misleading meta-model performance?</summary>

Base models may have memorized training labels. Use out-of-fold predictions for training the meta-model, then fit base models for inference.

</details>

[← Chapter index](../04-ml.md) · [Next →](03-distance-margin-bayes.md)
