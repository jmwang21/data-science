# 04.15 — Domain Shift, Interpretation & Model Selection

[← ML index](../04-ml.md) · [Previous](validation-leakage.md)

## 1. Distribution shift

| Type | Definition | Scenario |
|---|---|---|
| Covariate shift | P(X) changes; P(Y given X) stable | New market has different income mix |
| Label shift | P(Y) changes; P(X given Y) stable | Fraud prevalence changes |
| Concept drift | P(Y given X) changes | Same features now imply different risk |

Selection bias arises from nonrepresentative sampling, e.g. observing labels only for approved loan applicants. Observing feature drift does not identify which mechanism caused it.

## 2. What can be monitored without labels?

Schema, missingness, feature distributions, score distributions, cohort composition and extreme values can be monitored. KS test compares empirical CDFs; p-value depends on sample size. PSI compares proportions in fixed bins:

**PSI = Σᵢ (actualᵢ − expectedᵢ) × ln(actualᵢ / expectedᵢ)**

Handle zero proportions explicitly; thresholds are heuristics, not universal standards. Without ground truth cannot directly calculate actual recall, precision, accuracy, MAE or AUC. When delayed labels arrive, evaluate cohorts over time, assess calibration and check business outcomes. Retraining is an option only after investigating the cause and validating on representative data.

## 3. Interpretation

| Method | Meaning | Limitation |
|---|---|---|
| Linear coefficient | conditional association per unit | correlated features, scaling, noncausal |
| Impurity importance | total tree split improvement | may favor high-cardinality features |
| Permutation importance | validation performance drop when shuffling | correlated predictors may mask importance |
| Partial dependence | average model prediction as feature varies | unrealistic combinations with correlated features |
| SHAP (P1) | contribution to a prediction relative to baseline | depends on background/assumptions; not causal |

Example validation AUC=.82, after shuffling feature A=.75. Permutation importance=.07 for that shuffle. If B carries similar information, shuffling A alone may make little difference. Repeat shuffles and evaluate on suitable held-out data.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
X,y=make_classification(n_samples=300,n_features=8,random_state=42)
Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
m=RandomForestClassifier(n_estimators=100,random_state=42).fit(Xtr,ytr)
r=permutation_importance(m,Xv,yv,scoring='roc_auc',n_repeats=5,random_state=42)
print(r.importances_mean)
```

## 4. Compare algorithms by problem constraints

| Scenario | Reasonable candidates | Tradeoff |
|---|---|---|
| Interpretable continuous target | OLS, Ridge, Lasso | linear structure vs nonlinear bias |
| Sparse text | LogisticRegression, MultinomialNB, linear SVM | efficient sparse baselines |
| Mixed tabular nonlinear | RandomForest, boosting | interactions, complexity |
| Small scaled nonlinear dataset | KNN, kernel SVM | distance/margin, inference cost |
| Customer segmentation | K-means, hierarchical, DBSCAN | geometry, density and scale |
| Large unstructured data | neural networks | learned representation, data/compute requirements |

Select via appropriate validation and metric, not model reputation. Review [Classification Metrics](../modeling/02-classification-metrics.md), [Regression & MAE](../modeling/03-regression.md) and [Calibration](../modeling/05-clustering-calibration.md).

## Concept variations

<details><summary>1. Fraud prevalence rises but TPR and FPR remain fixed. Does precision stay constant?</summary>

No. Precision = prevalence × TPR / [prevalence × TPR + (1−prevalence) × FPR]. Base rate changes precision; the ROC operating point can remain unchanged.

</details>

<details><summary>2. A KS test indicates feature drift. Can you conclude model accuracy decreased?</summary>

No. Without labels you can detect distribution changes but not establish performance degradation. Investigate cohort and instrumentation changes and obtain labels.

</details>

<details><summary>3. Feature A has a high SHAP contribution. Does changing A necessarily improve business outcome?</summary>

No. Attribution describes model prediction, not causal intervention effect.

</details>
