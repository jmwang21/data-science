# 04.13 — Bias, Variance & Generalization

[← ML index](../04-ml.md) · [Next →](validation-leakage.md)

## 1. Bias-variance tradeoff

For squared loss under standard assumptions, expected prediction error = bias² + variance + irreducible noise. Bias is systematic error of average predictor; variance is sensitivity to training samples; noise is uncertainty not removable given available features.

| Model behavior | Likely issue | Next action |
|---|---|---|
| Train error 40, validation 42 | high bias or noise | better features, capacity, inspect target |
| Train error 5, validation 40 | high variance or shift | simpler model, regularization, check split |
| Train 10, validation 12 | small gap | compare baseline and business target |
| Train 2, random validation 3, future validation 60 | temporal shift/leakage possible | check time cutoff, representative validation |

These are diagnostic clues, not proofs. A large gap can reflect distribution differences; both errors high may be irreducible noise.

## 2. Learning curve vs validation curve

Learning curve varies training sample size. If validation improves while train/validation gap shrinks with more data, collecting representative data may help. If both curves plateau at poor scores, more data alone may be less effective.

Validation curve varies one hyperparameter (e.g., tree depth). Choose a region with good validation score, not maximal train score. Validation curve is not a learning curve.

```python
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve,StratifiedKFold
import numpy as np
X,y=make_classification(n_samples=400,n_features=10,random_state=42)
sizes,tr,va=learning_curve(DecisionTreeClassifier(max_depth=4,random_state=42),
    X,y,train_sizes=[.3,.6,1.],cv=StratifiedKFold(5,shuffle=True,random_state=42),
    scoring='roc_auc')
print(sizes,tr.mean(axis=1),va.mean(axis=1))
```

## 3. Regularization and capacity

L1 promotes sparsity; L2 shrinks coefficients; ElasticNet mixes both. Larger alpha for Ridge/Lasso and smaller C for LogisticRegression/SVM generally strengthen regularization. For trees, depth restriction and minimum leaf size limit capacity. For neural networks, dropout, weight decay and early stopping can help. For boosting, jointly tune iterations, learning rate and tree capacity.

Data augmentation may reduce variance if transformations preserve labels; duplicated observations alone do not create independent information. Over-regularization can underfit. Bagging reduces variance when component errors are not perfectly correlated.

## Concept variations

<details><summary>1. Both training and validation error are high and close. Is more data always the best solution?</summary>

No. Inspect capacity, features and label quality; high bias or noise may dominate. Learning curves help decide.

</details>

<details><summary>2. If training error decreases after increasing regularization, is that impossible?</summary>

Not universally impossible: optimization, finite samples and metric choice can differ. In ideal nested objective cases stronger penalty often worsens unpenalized training fit; validation may improve.

</details>

<details><summary>3. Can a model overfit validation even if test has not been accessed?</summary>

Yes. Repeated adaptive model selection can exploit validation noise. Keep final test independent or use nested CV.

</details>
