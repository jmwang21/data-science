# 04.8 — Support Vector Machine

[← ML index](../04-ml.md) · [Next →](naive-bayes.md)

## 1. Margin and support vectors

**English:** An SVM seeks a decision boundary with a large margin while controlling classification violations.

Linear boundary: wᵀx+b=0；margin boundaries: wᵀx+b=±1；margin width=2/||w||. Support vectors are points that determine the boundary (including relevant margin violations under soft margin).

Hard-margin requires separability; soft-margin allows violations. Hinge loss = max(0,1−y×score), with y∈{−1,+1}. A correctly classified point with score insufficiently far from the boundary can still have positive loss.

Example y=+1: score=2 → loss=0; score=.2 → .8; score=−1 → 2.

## 2. Kernels and parameters

Kernel computes similarity corresponding to an implicit feature representation. Linear kernel fits linear boundary, RBF kernel uses exp(−γ||x−z||²) and can produce nonlinear boundaries.

| Parameter | Increasing it typically means |
|---|---|
| C | stronger penalty for violations; weaker regularization |
| gamma (RBF) | narrower influence of each point; more local boundary |
| epsilon (SVR) | larger error-insensitive tube |

High C and high gamma may overfit. Low C/gamma may underfit. Gamma has no analogous role in a linear kernel. Scale input for distance and optimization. Kernel SVM can be expensive on large datasets; linear SVM often useful for high-dimensional sparse data.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
X,y=make_classification(n_samples=250,n_features=5,random_state=42)
Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
m=make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',gamma='scale'))
m.fit(Xtr,ytr)
print(roc_auc_score(yv,m.decision_function(Xv)))
```

decision_function 是 score，AUC 可直接使用；若需 calibrated probability，可在開發資料做 calibration。SVR 則預測連續值，epsilon-insensitive loss 不懲罰 tube 內的 residual。

## Concept variations

<details><summary>1. Does a correctly classified point always have zero hinge loss?</summary>

No. y×score must be at least 1; a correct but low-margin point still has loss.

</details>

<details><summary>2. RBF SVM has almost zero training error but high validation error. What could help?</summary>

Try lower C and/or gamma; check scaling and split before concluding overfitting.

</details>

<details><summary>3. Is decision_function output necessarily a probability?</summary>

No. It is a decision score. Ranking metrics can use it; probability metrics require appropriate probability estimates.

</details>
