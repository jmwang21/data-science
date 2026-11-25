# 04.5 — Decision Tree

[← ML index](../04-ml.md) · [Next →](ensembles.md)

## 1. How a tree learns

**English:** A decision tree recursively partitions the feature space to minimize impurity or prediction error.

每個 node 找 feature 與 threshold，將資料分成左右子群；greedy algorithm 每次選當下最佳 split，不保證全域最佳樹。Leaf 回傳類別分布或連續數值預測。

| Criterion | Formula / meaning |
|---|---|
| Gini impurity | 1 − Σ pₖ² |
| Entropy | −Σ pₖ log₂(pₖ), with 0 log 0 = 0 |
| Information gain | Parent entropy − weighted child entropy |
| Regression squared error | Weighted reduction of within-node squared error |

Gini 與 Entropy 都在純節點為 0；二分類均衡節點 Gini=.5、Entropy=1。分類樹用 leaf class proportions 產生機率；若有 sample weights，使用加權分布。

## 2. Worked example: compare splits

Parent: 6 positives and 4 negatives. Gini=1−.6²−.4²=.48。

| Split | Left P/N | Right P/N | Weighted Gini | Gain |
|---|---|---|---:|---:|
| A | 3/0 | 3/4 | .3429 | .1371 |
| B | 4/1 | 2/3 | .40 | .08 |

選 A。不能只看哪個 child 最純，必須考慮兩邊 sample size。Entropy 也使用 weighted average，並非固定與 Gini 產生相同選擇。

Regression example: y=[10,20,80,90]。Parent mean=50，SSE=5000；split [10,20] / [80,90] 的 leaf means=15,85，總 SSE=100，因此 reduction=4900。若 criterion 為 absolute_error，leaf 以 median 為代表。

## 3. Complexity and pruning

| Parameter | What increasing it does |
|---|---|
| max_depth | allows deeper, more complex trees |
| min_samples_split | requires more rows before a split |
| min_samples_leaf | requires more rows per leaf, smooths estimates |
| max_leaf_nodes | allows more leaves |
| ccp_alpha | stronger cost-complexity pruning |

Pre-pruning 在建樹時限制分裂，post-pruning 在長成後刪除貢獻較小的 subtree。Cost-complexity pruning balances leaf impurity and number of leaves。高容量可能降低 training error 但造成 variance。樹通常不需 scaling，但仍須處理 categorical representation 與 estimator 不支援的缺失值。樹的葉預測是局部常數，通常不善於 extrapolation。

## 4. Implementation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
X,y=make_classification(n_samples=300,n_features=6,random_state=42)
Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
for depth in [2,5,None]:
    m=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=3,random_state=42)
    m.fit(Xtr,ytr)
    print(depth,roc_auc_score(ytr,m.predict_proba(Xtr)[:,1]),
          roc_auc_score(yv,m.predict_proba(Xv)[:,1]))
```

## Concept variations

<details><summary>1. A split makes one child perfectly pure. Is it necessarily optimal?</summary>

No. 比較 weighted impurity reduction，另一側樣本數及 impurity 同樣重要。

</details>

<details><summary>2. A tree has near-perfect training accuracy and weak validation performance. Which parameters can restrict capacity?</summary>

Reduce max_depth or increase min_samples_leaf/ccp_alpha. 同時確認 validation 是否有 distribution shift。

</details>

<details><summary>3. Does standardizing numeric features necessarily improve a decision tree?</summary>

No. 樹主要根據 feature ordering 找 threshold，單調線性 scaling 通常不改變可選的 partitions。

</details>
