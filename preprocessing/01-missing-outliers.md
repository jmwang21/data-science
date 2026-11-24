# 02.1 — Missing Values & Outliers

[← Chapter index](../02-preprocessing.md) · [Next →](02-scaling-encoding.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## Missing values & cleaning 

```python
import pandas as pd
import numpy as np
# df 為輸入資料
df.isna().sum()       # 各欄缺失數
df.isna().mean()      # 各欄缺失比例
df.dropna(subset=['target'])
df.drop_duplicates(subset=['order_id'])
pd.to_numeric(df['amount'],errors='coerce')
```

`isna().count()` 計算非缺失布林值數量，不是缺失數。缺失不等於零；沒有交易可定義 GMV=0，但未知金額不一定是零。去重前確認 business key，避免刪除合法重複交易。

| API | 用途 |
|---|---|
| `s.fillna(0)` | 填缺失 |
| `s.replace([-999,-1],np.nan)` | 替換指定值 |
| `s.replace({'A':1,'B':2})` | mapping，未匹配保留 |
| `s.map({'A':1,'B':2})` | mapping，未匹配通常 NaN |
| `s.where(s>=0,0)` | 條件成立保留 |

## Infinity & outliers 

```python
df['growth'] = (df.feb_gmv-df.jan_gmv)/df.jan_gmv.replace(0,np.nan)
df = df.replace([np.inf,-np.inf],np.nan)
np.isinf(df['growth']).sum()
np.isfinite(df['growth'])
```

浮點向量除以零可得 inf/-inf，0/0 為 NaN；Python scalar 除以零則可能拋出 ZeroDivisionError。`pd.to_numeric(errors='coerce')` 不會消除 inf。`fillna` 也不會。成長率分母為零通常應保留未定義，而非假設 0%。

IQR = Q3-Q1，常用界線 Q1-1.5IQR、Q3+1.5IQR；不是必然應刪除的異常。Winsorization / clipping 可降低極端值影響，界線須從 training fit。

## Imputation decisions — 不只是 fillna

**English:** Impute missing values using statistics estimated from the training set.

| Training values | Mean | Median |
|---|---:|---:|
| 10, 20, 30, 1000 | 265 | 25 |

若下一筆缺失，用 mean=265 可能產生不合理的典型值；median=25 較穩健，但若缺失與高收入族群相關，median 也可能造成偏差。

```python
from sklearn.impute import SimpleImputer
import numpy as np
X_train = np.array([[10.],[20.],[30.],[1000.],[np.nan]])
X_val = np.array([[np.nan],[40.]])
imp = SimpleImputer(strategy='median',add_indicator=True)
print(imp.fit_transform(X_train))
print(imp.transform(X_val)) # [[25.,1.],[40.,0.]]
```

MCAR：缺失不依賴資料；MAR：給定已觀察變數後缺失不依賴未知值；MNAR：即使給定觀察資料仍依賴未觀察值。不要假定補值能消除所有偏差。

Outlier 是罕見觀測，不必然是錯誤。IQR=Q3−Q1；常見界線 Q1−1.5IQR、Q3+1.5IQR。Clip 閾值應由 train 決定。真實高價房屋若是預測目標的重要族群，直接刪除可能讓模型更差。

### Concept variations

<details><summary>1. A validation outlier is beyond the training 99th percentile. Should you recompute the percentile on validation?</summary>

No. 若採用 clipping，沿用 training 學到的閾值，否則 validation 參與 preprocessing fit。也先確認 clipping 是否合理。

</details>

<details><summary>2. Missing target values occur in the training set. Should you impute them with the median house price?</summary>

通常不應。補 target 會人為製造標籤與扭曲 loss。一般排除無標籤列或採有明確設計的半監督方法；feature 缺失與 target 缺失是不同問題。

</details>

[← Chapter index](../02-preprocessing.md) · [Next →](02-scaling-encoding.md)
