# 04.7 — K-Nearest Neighbors

[← ML index](../04-ml.md) · [Next →](svm.md)

## 1. Mechanism

**English:** KNN predicts using the labels or target values of the k nearest training observations.

KNN 是 instance-based / lazy learner。Training 主要儲存資料；inference 搜尋鄰居。分類 vote、回歸平均。Euclidean distance = sqrt(Σ(xⱼ−zⱼ)²)，Manhattan distance = Σabs(xⱼ−zⱼ)。

| Neighbor distance | Label | Regression target |
|---:|---:|---:|
| .1 | 0 | 10 |
| 1 | 1 | 20 |
| 2 | 1 | 40 |

k=1 predicts 0；k=3 uniform predicts 1。Distance weights=[10,1,.5]，class 0 wins。Regression uniform mean=23.33，weighted mean=140/11.5≈12.17。

## 2. Key parameters and limitations

| Parameter | Effect |
|---|---|
| n_neighbors | small k: flexible/high variance; large k: smooth/high bias |
| weights | uniform or distance |
| metric | determines similarity |

Scaling 很重要，否則金額可能主宰距離、年齡失去影響。高維資料的距離辨別力可能降低（curse of dimensionality）；irrelevant features、outliers、class imbalance 也可能影響近鄰。KNN inference 成本隨資料量及搜尋方法改變，通常不像 linear model 一樣只需 dot product。

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
X,y=make_classification(n_samples=200,n_features=6,random_state=42)
Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
for k in [1,5,25]:
    m=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=k))
    m.fit(Xtr,ytr)
    print(k,m.score(Xtr,ytr),m.score(Xv,yv))
```

## Concept variations

<details><summary>1. What generally happens when k increases?</summary>

Predictions become smoother; variance tends to fall while bias may rise. Validation performance is not guaranteed to improve monotonically.

</details>

<details><summary>2. Is adding more features always beneficial for KNN?</summary>

No. Irrelevant dimensions can distort distance and reduce the distinction between neighbors.

</details>

<details><summary>3. Why could KNN perform worse than trees on unscaled numeric inputs?</summary>

KNN uses distances; large-unit features dominate. Trees use split order and are usually insensitive to monotonic scaling.

</details>
