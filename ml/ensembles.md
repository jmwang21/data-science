# 04.6 — Bagging, Random Forest, Boosting & Stacking

[← ML index](../04-ml.md) · [Next →](knn.md)

## 1. Why ensembles work

Bagging trains models independently on bootstrap samples; boosting learns sequentially; voting combines predictions; stacking learns a combination. Ensemble 效果取決於 individual model quality 與 error correlation。

假設 M 個模型各自 variance=v、pairwise correlation=ρ，平均的 variance = v[ρ+(1−ρ)/M]。v=100、M=10、ρ=.2 時為 28；ρ=.9 則為 91。更多模型無法消除共享誤差。

## 2. Random Forest

Bootstrap 從 n rows 有放回抽 n 次，某列沒被抽到的機率=(1−1/n)ⁿ，n 大時約 36.8%。每棵樹看到約 63.2% 不同資料列。每個 split 隨機選候選 feature，減少樹間相關性。

Regression 平均各樹數值；sklearn classification 平均各樹 class probabilities，再選最大者。

| Parameter | Effect |
|---|---|
| n_estimators | stabilizes averaging; increases compute |
| max_features | candidate features per split; diversity vs strength |
| max_depth | individual tree capacity |
| min_samples_leaf | smooths leaf predictions |
| bootstrap / max_samples | controls row sampling |
| class_weight | changes class contribution to learning |

max_features: sqrt、log2、整數或比例。OOB prediction 只用未抽到該 observation 的樹，提供 development performance；反覆用 OOB 選模型仍可能選擇偏誤。

## 3. Gradient Boosting

**English:** Gradient boosting builds an additive model by sequentially fitting weak learners to improve the current loss.

Update: Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)

Squared error example: y=[10,20,30]，初始 mean=20；residual=[−10,0,10]。下一棵樹預測 [−8,0,8]，learning_rate=.5，更新 prediction=[16,20,24]，residual=[−6,0,6]。一般 differentiable loss 使用 negative gradient，不一定是 raw residual。

| Parameter | Effect |
|---|---|
| learning_rate | contribution per learner |
| n_estimators | number of sequential stages |
| max_depth / max_leaf_nodes | base learner complexity |
| subsample | row fraction; stochastic regularization |
| early stopping | stop based on development score |

小 learning rate 通常需要較多 trees；固定 trees 時未必越小越好。AdaBoost 增加錯分樣本權重。XGBoost 使用 regularized tree objectives，LightGBM 常用 histogram 與 leaf-wise growth，CatBoost 提供 categorical handling / ordered boosting。理解差異即可，不要求冷門參數。sklearn GradientBoosting 與 HistGradientBoosting 是可用的替代。

## 4. Voting and stacking

| Method | Combination |
|---|---|
| Hard voting | majority class labels |
| Soft voting | average class probabilities |
| Stacking | meta-model trained on out-of-fold base predictions |

Example p=[.51,.51,.01]：hard vote=1；soft average=.343 → class 0 at threshold .5。Stacking 使用未被該 fold base model 訓練的 observations 建立 meta features；再以完整 development data fit base models 供 inference。In-sample base predictions 可能讓 meta-model 學到過度樂觀的訊號。

## 5. Implementation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
X,y=make_classification(n_samples=300,n_features=8,random_state=42)
Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
models={
    'rf':RandomForestClassifier(n_estimators=100,min_samples_leaf=3,
                               oob_score=True,random_state=42),
    'gb':GradientBoostingClassifier(n_estimators=100,learning_rate=.05,
                                    max_depth=2,random_state=42)
}
for name,m in models.items():
    m.fit(Xtr,ytr)
    print(name,roc_auc_score(yv,m.predict_proba(Xv)[:,1]))
```

## Concept variations

<details><summary>1. Does increasing tree depth have the same effect as increasing n_estimators?</summary>

No. Depth controls individual capacity; more trees stabilizes the ensemble average.

</details>

<details><summary>2. Does lowering max_features always improve Random Forest?</summary>

No. It may decorrelate trees but weaken each tree. Validate the tradeoff.

</details>

<details><summary>3. Select all true: A. Bagging can reduce variance. B. Boosting always fits raw residuals. C. Stacking uses out-of-fold predictions. D. Random Forest trains independent trees.</summary>

A,C,D. B is false for general losses; squared-error boosting is the familiar residual case.

</details>

<details><summary>4. A smaller learning rate yields poor training and validation scores with the same number of trees. What could help?</summary>

Increase boosting iterations or reconsider base learner capacity; lowering learning rate further may underfit. Tune jointly using validation.

</details>
