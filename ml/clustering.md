# 04.10 — Clustering

[← ML index](../04-ml.md) · [Next →](pca.md)

## 1. K-means

**English:** K-means minimizes the sum of squared distances between observations and their assigned cluster centroids (within-cluster SSE, or inertia).

Algorithm: initialize K centroids → assign each row to its nearest centroid → recompute each centroid as the cluster mean → repeat until convergence. Local optima depend on initialization; k-means++ and multiple initializations help.

Worked example: points [1,2,3,8,9,10], K=2, initial centroids 1 and 8. Assign [1,2,3] and [8,9,10]; update centroids to 2 and 9. Inertia=(1+0+1)+(1+0+1)=4. If K=1 centroid=5.5, inertia=77.5. Higher K cannot increase the theoretical minimum inertia, so choosing only lowest inertia is not sensible.

Scale variables when units differ. Extreme values can pull centroids; K-means prefers compact, roughly spherical clusters. It does not require the target labels.

## 2. How to choose K

| Metric | Direction | Meaning |
|---|---|---|
| Inertia | lower | squared within-cluster distances; use elbow |
| Silhouette | higher, range −1 to 1 | cohesion versus separation |
| Davies-Bouldin | lower | within-cluster scatter relative to inter-centroid distance |

Silhouette for a point: s = (b−a)/max(a,b), where a=average distance to own cluster and b=smallest average distance to another cluster. a=2,b=6 → .667；a=6,b=2 → −.667。Negative 值表示可能更接近其他群。Singleton 通常定義為 0。

DBI compares each cluster's worst relative overlap with another cluster and averages across clusters. It is an internal score; lower does not guarantee commercial value.

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,davies_bouldin_score
X,_=make_blobs(n_samples=200,centers=3,random_state=42)
Z=StandardScaler().fit_transform(X)
for k in range(2,7):
    m=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Z)
    print(k,m.inertia_,silhouette_score(Z,m.labels_),
          davies_bouldin_score(Z,m.labels_))
```

This is exploratory clustering. For future prediction, fit scaler and clustering on development data and use the fitted models for new rows. Evaluate stability across seeds/resamples and examine size, profile, actionable differences.

## 3. Alternative methods

| Method | Mechanism | When useful / limitation |
|---|---|---|
| Hierarchical | iteratively merge clusters using linkage | dendrogram and flexible cuts; can be expensive |
| DBSCAN | density-connected neighborhoods | arbitrary shapes, noise; difficult varying density |
| Gaussian Mixture (P1) | probabilistic mixture with soft assignments | overlapping elliptical clusters; distribution assumptions |

Single linkage uses minimum pair distance, complete linkage maximum, average linkage average, Ward minimizes increase in within-cluster variance (Euclidean). DBSCAN eps defines neighborhood, min_samples controls minimum density; core points connect clusters, border points attach, noise may get label −1. No fixed K, but parameters matter.

GMM uses posterior component probabilities; unlike K-means, membership can be uncertain and covariance structures can differ.

## Concept variations

<details><summary>1. K=5 has lower inertia but worse silhouette than K=3. Which must be chosen?</summary>

Neither automatically. Inertia naturally falls with K; silhouette measures separation. Compare elbow, stability, group sizes and business objective.

</details>

<details><summary>2. Data contains two curved clusters with isolated points. Which algorithm may be preferable to K-means?</summary>

DBSCAN if density conditions are suitable. It can identify nonconvex clusters and noise.

</details>

<details><summary>3. Cluster IDs change from [0,1] to [1,0] across runs. Does that prove clustering changed?</summary>

No. Labels are arbitrary; compare memberships (e.g., adjusted Rand index) and cluster profiles rather than IDs directly.

</details>
