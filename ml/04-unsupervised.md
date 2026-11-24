# 04.4 — Clustering, PCA & Anomaly Detection

[← Chapter index](../04-ml.md) · [Previous](03-distance-margin-bayes.md) · [Next →](05-generalization.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## K-Means & Clustering 

目標：最小化 within-cluster SSE $\sum_i\|x_i-\mu_{c_i}\|^2$。初始化中心 → assign 最近中心 → 更新均值 → 重複直到收斂。對 scaling、初始化、outliers 敏感；偏好近似球形、大小相近的 cluster。

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
Z = StandardScaler().fit_transform(X)
for k in range(2,6):
    m = KMeans(n_clusters=k,n_init=10,random_state=42).fit(Z)
    print(k,m.inertia_,silhouette_score(Z,m.labels_),
          davies_bouldin_score(Z,m.labels_))
```

假設至少 6 筆有效樣本，且各 k 產生足夠不同 clusters。

| Metric | 越好方向 | 解讀 |
|---|---|---|
| Inertia | 越低（k 增大通常下降） | Elbow 看邊際改善，不是直接取最低 |
| Silhouette | 越高，範圍 [-1,1] | $(b-a)/\max(a,b)$；a 組內平均距離，b 最近其他組平均距離 |
| Davies-Bouldin | 越低 | 組內分散 / 組間距離 |

最佳 k 還需穩定性、解釋性、營運可執行性。PCA 可降維/去相關，但最大 variance 不保證對 target 最有用；supervised task 中 PCA 必須在 train fold fit。

## Beyond K-means

| Method | Main idea | Limitation |
|---|---|---|
| K-means | minimize squared distance to centroids | favors compact clusters, sensitive to scale/outliers |
| Hierarchical | merge clusters using linkage | cost, linkage choice |
| DBSCAN | density-connected points, eps/min_samples | varying density can be difficult |
| PCA | maximize projected variance in orthogonal directions | linear; high variance not necessarily predictive |
| Isolation Forest (P1) | anomalies isolate in fewer random splits | anomaly is not automatically fraud |

PCA example: two standardized features are strongly positively correlated. First principal component roughly (x1+x2)/sqrt(2); second captures their difference. If explained variance ratios are [.8,.15,.05], first two retain 95% variance. Scale before PCA when units differ; fit on train when part of predictive pipeline.

<details><summary>Variation: The elbow curve decreases monotonically with K. Why not choose the largest K?</summary>

Inertia always nonincreasing as capacity increases. At K=n distinct points, it can be zero but produces no useful segmentation. Compare elbow, silhouette, stability and business utility.

</details>

<details><summary>Variation: PCA retains 99% of variance. Does that imply 99% of predictive information is preserved?</summary>

No. Low-variance directions may be strongly related to target. Evaluate downstream performance.

</details>

[← Chapter index](../04-ml.md) · [Next →](05-generalization.md)
