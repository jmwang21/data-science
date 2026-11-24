# 02.2 — Scaling, Encoding & Transformations

[← Chapter index](../02-preprocessing.md) · [Previous](01-missing-outliers.md) · [Next →](03-feature-engineering.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## Scaling & encoding 

StandardScaler: $z=(x-\mu)/\sigma$；MinMaxScaler: $(x-min)/(max-min)$。LR、SVM、KNN、K-means 常受尺度影響；tree-based 模型通常不需要 scaling。OneHotEncoder 適用 nominal category；OrdinalEncoder 適用有序類別或明確編碼需求，不應任意假設順序。未知類別可用 handle_unknown='ignore'。

## Worked examples — scaling, encoding and missingness 

Training values [10,20,30] have mean 20 and population SD ≈8.165. StandardScaler produces approximately [-1.225,0,1.225]. A test value 40 becomes 2.449 using the same parameters, not a newly fit test scaler. MinMaxScaler fit on these values maps [10,20,30] to [0,.5,1]; 40 becomes 1.5 by default (not clipped).

| Tier | basic | pro |
|---|---:|---:|
| basic | 1 | 0 |
| pro | 0 | 1 |
| unseen enterprise | 0 | 0 |

The example assumes OHE fit on basic and pro, `handle_unknown='ignore'`, no dropped column. Unknown levels are not automatically missing values. High-cardinality merchant ID generally needs a justified encoding or exclusion; target encoding must be learned without leaking validation labels.

### Leakage-safe imputation

```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median')
train_num = imp.fit_transform(X_train[['monthly_gmv']])
valid_num = imp.transform(X_valid[['monthly_gmv']])
```

Never fit on combined train/validation. Median is robust to extreme observations but not universally optimal; missingness indicators can help if absence carries signal. Avoid imputing target labels for supervised evaluation.

### Feature selection & imbalance

Supervised selection (e.g. SelectKBest) must be inside the Pipeline. For imbalanced classification, `class_weight='balanced'` may change the loss emphasis but does not guarantee better recall or precision at a chosen threshold. Resampling must occur within training folds, never on validation/test. Evaluate on the distribution relevant to deployment.

### Data quality checklist

| Check | Question |
|---|---|
| Schema | Are numeric fields numeric and dates parsed? |
| Grain | Is a row an order, merchant, or merchant-month? |
| Completeness | Is missing value absence of data or a true zero? |
| Freshness | Was the feature available at prediction time? |
| Validity | Is negative GMV a refund, error, or legitimate value? |
| Duplication | Is the business key unique at the defined grain? |

## Worked variation: choice of scale and encoding

StandardScaler 是 feature-wise；Normalizer 是 row-wise。若某筆向量 [3,4] 做 L2 normalize 得 [0.6,0.8]，不是標準化為 mean=0。

| Transform | Appropriate case | Key limitation |
|---|---|---|
| StandardScaler | regularized LR, SVM, KNN | outlier 敏感 |
| MinMaxScaler | fixed training range | unseen extremes 可超出 [0,1] |
| RobustScaler | heavy-tailed features | 不是刪除 outlier |
| log1p | nonnegative skew | 改變尺度；不保證常態 |
| OrdinalEncoder | 有真實次序的 tier | 任意編碼會引入假順序 |
| OneHotEncoder | nominal category | high cardinality 維度增加 |

```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
train = pd.DataFrame({'tier':['basic','pro','basic']})
enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
enc.fit(train)
print(enc.transform(pd.DataFrame({'tier':['new','pro']})))
# [[0.,0.],[0.,1.]]
```

sklearn 1.2+ 使用 sparse_output；舊版為 sparse。避免記所有冷門參數。Target encoding 需 out-of-fold，否則訓練列的 target 會直接進入自身 feature。

<details><summary>Variation: A zipcode is stored as an integer. Should it be standardized as a continuous feature?</summary>

不一定。郵遞區號的數值距離通常沒有連續量意義，可視為類別或建立合理的地理特徵。dtype 不等於 feature semantics。

</details>

<details><summary>Variation: Does a Random Forest normally require standardized numerical inputs?</summary>

No. 樹比較 feature ordering 與切分點，單調 scaling 通常不影響切分。KNN/SVM 則會受距離尺度影響。

</details>

[← Chapter index](../02-preprocessing.md) · [Next →](03-feature-engineering.md)
