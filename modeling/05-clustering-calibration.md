# 03.5 — Clustering, Multiclass & Calibration

[← Chapter index](../03-models.md) · [Previous](04-tuning.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## Additional model evaluation and implementation (P0)

| Task | sklearn implementation | Key input |
|---|---|---|
| Multiclass classification | classification_report(y_true,y_pred) | labels |
| Probability quality | log_loss(y_true,probs), brier_score_loss(y_true,p) | probabilities |
| Clustering | silhouette_score(X,labels), davies_bouldin_score(X,labels) | features + cluster assignments |
| Ranking | precision/recall at K | sorted scores |
| Calibration | calibration_curve(y_true,p,n_bins=10) | binary probabilities |

Multiclass macro gives each class equal weight; weighted weights by support; micro pools TP/FP/FN. In single-label multiclass classification, micro F1 equals accuracy when all classes included. Brier score is mean squared probability error for binary outcomes; lower is better. A perfectly ranked model can be poorly calibrated.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
# X_numeric: complete finite numeric matrix; > max(k) observations
Z=StandardScaler().fit_transform(X_numeric)
scores={}
for k in range(2,7):
    labels=KMeans(n_clusters=k,n_init=10,random_state=42).fit_predict(Z)
    if 2<=len(set(labels))<len(Z):
        scores[k]=silhouette_score(Z,labels)
best_k=max(scores,key=scores.get)
```

For clustering without labels, silhouette and Davies-Bouldin are internal criteria, not evidence of business usefulness. DB index lower is better. Evaluate stability, cluster size and interpretability; if true labels exist, adjusted Rand index can compare assignments. For prediction on new entities fit preprocessing and cluster model on development data; use KMeans.predict for new points.

### Randomized search and early stopping (P1)
RandomizedSearchCV samples parameter candidates, unlike exhaustive grid. Choose distributions/ranges appropriate to parameter scale; set random_state. For early stopping, monitor held-out loss and retain best iteration; never monitor test. For forecasting, chronological validation matters more than random split; compare naive last-value/seasonal baselines and prevent future exogenous-feature leakage.

<details><summary>Concept check: does the model with best ROC-AUC necessarily have lowest Log Loss?</summary>

No. ROC-AUC measures ordering, while Log Loss penalizes inaccurate or overconfident probabilities. The same ordering can have different probability calibration.

</details>



## Interpreting model outputs

Multiclass probabilities: row sum ≈1; columns follow estimator.classes_. Macro F1 對每類等權；weighted F1 依 support 加權；micro F1 對所有決策聚合。單標籤多分類納入所有 classes 時 micro F1=accuracy。

Clustering silhouette s=(b−a)/max(a,b)，a 是同群平均距離，b 是最近其他群平均距離。若 a=2、b=6，s=2/3；若 a=6、b=2，s=−2/3，可能分錯群。DBI lower is better。Inertia 隨 K 不增，不能單靠最小 inertia 選 K。

<details><summary>Variation: K=4 has higher silhouette than K=3, but one group contains only two customers. Must K=4 be chosen?</summary>

No. Metric 是幾何分群品質；需看穩定性、商業用途、群大小與可解釋性。

</details>

<details><summary>Variation: Two classifiers rank transactions identically, but one predicts 0.99 for almost all positive-looking cases. Are their calibrated probabilities equivalent?</summary>

No. 排序相同可有相同 AUC，但 probability calibration 和 log loss 可能不同。

</details>

[← Chapter index](../03-models.md)
