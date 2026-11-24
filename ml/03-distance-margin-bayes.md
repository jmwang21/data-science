# 04.3 — KNN, SVM & Naive Bayes

[← Chapter index](../04-ml.md) · [Previous](02-trees-ensembles.md) · [Next →](04-unsupervised.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## SVM 

最大化 margin；soft margin 容許分類錯誤。C 大通常較少容忍錯分，可能提高 variance；C 小較強正則化。RBF kernel 的 gamma 大，單筆樣本影響範圍較小，決策邊界可能更複雜。

```python
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
svm = make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',gamma='scale'))
# AUC 可直接使用 decision_function；若需 predict_proba 須設 probability=True
```

SVM 對尺度敏感，大資料的 kernel SVM 成本可能高。Linear SVM 適合高維 sparse features。

## KNN — a distance-based learner

KNN predicts by vote (classification) or average (regression) among nearest training observations. k small means flexible decision boundary and sensitivity to noise; k large smooths and can underfit. Standardize when feature units differ. High-dimensional distances can become less informative.

Example: three nearest neighbors labels [1,1,0], uniform vote gives class 1; with k=1 nearest label=0 would give class 0 (if nearest is the 0). Distance weights can change the vote.

## SVM — margin and kernel

Soft margin allows violations with penalty C. Large C discourages violations; high gamma RBF makes points influence a more local region. Together they can increase complexity. SVC decision_function returns scores, not necessarily calibrated probabilities.

## Naive Bayes

Bayes classifier models P(class|features) proportional to P(class)×P(features|class). Naive assumption: features conditionally independent given class. GaussianNB for continuous Gaussian class-conditional features, MultinomialNB for nonnegative counts. Not global independence; correlated features can double-count evidence.

<details><summary>Variation: An unscaled feature is measured in dollars (0–1,000,000) and another in years (0–20). What happens to KNN?</summary>

Dollar differences may dominate Euclidean distance; scale appropriately. A tree usually does not have this issue.

</details>

<details><summary>Variation: If SVM training error is almost zero but validation error is high, what parameter change could help?</summary>

Try reducing C and/or RBF gamma, validated on appropriate split. This may smooth decision boundaries; not guaranteed.

</details>

<details><summary>Variation: Does Naive Bayes require that two features be marginally independent?</summary>

No. The assumption is conditional independence within each class.

</details>

[← Chapter index](../04-ml.md) · [Next →](04-unsupervised.md)
