# 04.14 — Validation, Data Leakage & Generalization Design

[← ML index](../04-ml.md) · [Next →](06-drift-interpretation.md)

## 1. Match evaluation to deployment

| Scenario | Split strategy |
|---|---|
| IID binary samples, rare positives | Stratified train/test or StratifiedKFold |
| Multiple rows per patient, predict unseen patients | Group split |
| Future transactions | Chronological split / TimeSeriesSplit |
| Future new merchants | Time-based split plus entity isolation if task requires |
| Tune hyperparameters and estimate performance without holdout | Nested CV |

Stratification preserves class ratio, not group independence. Group splits prevent entity overlap but do not automatically guarantee chronological ordering. Random split can be suitable for IID data, not all business problems.

## 2. Leakage examples

| Risk | Example | Remedy |
|---|---|---|
| Target leakage | final cancellation status used to predict cancellation | remove unavailable features |
| Temporal leakage | next-month GMV used to predict current churn | as-of cutoff |
| Group leakage | same patient in both folds for unseen-patient task | GroupKFold |
| Preprocessing leakage | scale all data before CV | fit within folds |
| Feature selection leakage | use all y to select predictors, then CV | selector inside Pipeline |
| Test selection bias | report best of 12 test AUCs | validation selection, untouched test |

An existing customer's history can be valid for future prediction if available at cutoff. Entity overlap is problematic when deployment assumes unseen entities, not automatically in every situation.

## 3. CV and nested CV

Ordinary K-fold: each fold trains on K−1 parts, validates on remaining part. For tuning, repeat across hyperparameter candidates, compare mean validation metric. GridSearchCV(refit=True) refits on all supplied development rows after choosing parameters.

Nested CV: outer fold estimates the entire selection process; inner fold tunes only using outer training data. Evaluate outer validation once per fold, aggregate scores. Do not select the outer fold model with highest score as final model; choose/refit using full development data if needed.

```python
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score
X,y=make_classification(n_samples=300,n_features=6,random_state=42)
pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
inner=StratifiedKFold(3,shuffle=True,random_state=42)
outer=StratifiedKFold(5,shuffle=True,random_state=43)
search=GridSearchCV(pipe,{'logisticregression__C':[.1,1,10]},
                    cv=inner,scoring='roc_auc')
print(cross_val_score(search,X,y,cv=outer,scoring='roc_auc'))
```

Separate final test remains untouched if one is supplied. With an unlabeled test, output predictions instead of claiming to calculate test AUC/MAE. See [Model Workflow](../modeling/01-workflow-validation.md).

## Concept variations

<details><summary>1. A researcher selects features using all 10,000 labels, then performs 5-fold CV. What's wrong?</summary>

Validation labels influenced selected features; the estimate can be optimistic. Fit selector inside each training fold.

</details>

<details><summary>2. Does stratification prevent merchant ID overlap?</summary>

No. It preserves label proportions; use group-based splits to isolate merchants when necessary.

</details>

<details><summary>3. Why not report the highest score among outer folds in nested CV?</summary>

Outer folds estimate the selection procedure across data subsets. Report aggregate and variability, not a cherry-picked fold.

</details>
