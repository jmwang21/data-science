# 02.2 — Scaling, Encoding & Transformations

[← Chapter index](../02-preprocessing.md) · [Previous](01-missing-outliers.md) · [Next →](03-feature-engineering.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## Scaling & encoding 

StandardScaler: $z=(x-\mu)/\sigma$；MinMaxScaler: $(x-min)/(max-min)$。LR、SVM、KNN、K-means 常受尺度影響；tree-based 模型通常不需要 scaling。OneHotEncoder 適用 nominal category；OrdinalEncoder 適用有序類別或明確編碼需求，不應任意假設順序。OneHotEncoder 未知類別可用 handle_unknown='ignore'；OrdinalEncoder 可用 handle_unknown='use_encoded_value', unknown_value=-1。

## Worked examples — scaling, encoding and missingness 

Training values [10,20,30] have mean 20 and population SD ≈8.165. StandardScaler produces approximately [-1.225,0,1.225]. A test value 40 becomes 2.449 using the same parameters, not a newly fit test scaler. MinMaxScaler fit on these values maps [10,20,30] to [0,.5,1]; 40 becomes 1.5 by default (not clipped).

| Tier | basic | pro |
|---|---:|---:|
| basic | 1 | 0 |
| pro | 0 | 1 |
| unseen enterprise | 0 | 0 |

The example assumes OHE fit on basic and pro, `handle_unknown='ignore'`, no dropped column. Unknown levels are not automatically missing values. High-cardinality merchant ID generally needs a justified encoding or exclusion; target encoding must be learned without leaking validation labels.

## How to choose a transformation

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

sklearn 1.2+ 使用 sparse_output；舊版為 sparse。高 cardinality 時保留稀疏矩陣或比較頻率編碼。Target encoding 必須防止自身 label 進入 feature，training 可使用 out-of-fold encoding。

### Variation: nonlinear transformations

對 [1,10,1000]，log1p 轉換約 [0.693,2.398,6.909]，壓縮大值間差距。這不是 scaling：它改變分布形狀。Box-Cox 要求 strictly positive；Yeo-Johnson 可處理 0/負值（P1）。若模型目標是原尺度的 MAE，對 target 做 log transform 後必須反轉換再評分。

<details><summary>Variation: A zipcode is stored as an integer. Should it be standardized as a continuous feature?</summary>

不一定。郵遞區號的數值距離通常沒有連續量意義，可視為類別或建立合理的地理特徵。dtype 不等於 feature semantics。

</details>

<details><summary>Variation: Does a Random Forest normally require standardized numerical inputs?</summary>

No. 樹比較 feature ordering 與切分點，單調 scaling 通常不影響切分。KNN/SVM 則會受距離尺度影響。

</details>

[← Chapter index](../02-preprocessing.md) · [Next →](03-feature-engineering.md)
