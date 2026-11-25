# 04.11 — PCA, Dimensionality Reduction & Anomaly Detection

[← ML index](../04-ml.md) · [Next →](neural-networks.md)

## 1. Principal Component Analysis

**English:** PCA projects centered data onto orthogonal directions that successively maximize explained variance.

PCA creates new features, rather than selecting original ones. First PC maximizes projected variance; second PC maximizes remaining variance subject to orthogonality. Eigenvalues of the covariance matrix correspond to variance along PCs (or equivalent SVD formulation).

Two standardized features with correlation .8 have covariance matrix [[1,.8],[.8,1]]. PCs are approximately (x₁+x₂)/sqrt(2) and (x₁−x₂)/sqrt(2), eigenvalues 1.8 and .2. First explains 90% of total variance. PCA signs are arbitrary.

Explained variance ratio = component variance / total variance. For ratios [.80,.15,.05], two PCs retain 95%. Standardize if original feature units are not comparable; PCA centers but does not automatically standardize. Fit only on training folds in supervised workflow.

```python
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
X,y=make_classification(n_samples=200,n_features=10,random_state=42)
pipe=make_pipeline(StandardScaler(),PCA(n_components=.95),
                   LogisticRegression(max_iter=1000))
print(cross_val_score(pipe,X,y,cv=5,scoring='roc_auc'))
```

## 2. Compare dimensionality methods

| Method | Purpose |
|---|---|
| Feature selection | retain original feature subset |
| PCA | unsupervised linear projection; preserve variance |
| LDA (P1) | supervised projection emphasizing class separation |
| t-SNE / UMAP (P1) | nonlinear exploratory visualization; not default predictive features |

High variance need not mean predictive value. A low-variance feature can contain most target signal. 2D visualization can distort distances and cluster sizes; avoid treating visual separation as proof.

## 3. Anomaly detection

IsolationForest randomly partitions data; anomalies tend to isolate with fewer splits. Local Outlier Factor compares local density. One-Class SVM estimates a boundary around typical observations. Anomaly is not necessarily error or fraud; validate with domain knowledge and labels when available.

<details><summary>1. Does retaining 99% of variance guarantee preserving 99% of predictive information?</summary>

No. PCA ignores y. Validate downstream performance.

</details>

<details><summary>2. Should PCA be fitted before cross-validation because it uses no labels?</summary>

No for ordinary held-out generalization. Validation features affect learned components; fit inside Pipeline.

</details>

<details><summary>3. Is an anomalous high-value transaction necessarily fraudulent?</summary>

No. Anomaly score reflects unusualness under the model; ground truth and business investigation determine fraud.

</details>
