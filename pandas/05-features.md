# 01.5 — Feature Engineering & Conditional Transformation

[← pandas 目錄](../01-pandas.md) · [Previous](04-cleaning.md) · [Next →](06-reshape-window.md)

## 1. Derived numeric features

| merchant | jan_gmv | feb_gmv | orders |
|---|---:|---:|---:|
| A | 100 | 150 | 3 |
| B | 200 | 100 | 2 |
| C | 0 | 50 | 1 |
| D | 0 | 0 | 0 |

```python
import pandas as pd
import numpy as np
df = pd.DataFrame({'merchant':['A','B','C','D'],
                   'jan_gmv':[100,200,0,0],
                   'feb_gmv':[150,100,50,0],
                   'orders':[3,2,1,0]})
df['growth'] = (df.feb_gmv-df.jan_gmv)/df.jan_gmv.replace(0,np.nan)
df['aov'] = df.feb_gmv / df.orders.replace(0,np.nan)
```

| merchant | growth | aov |
|---|---:|---:|
| A | 0.5 | 50 |
| B | -0.5 | 50 |
| C | NaN | 50 |
| D | NaN | NaN |

Growth=(current−previous)/previous。從零開始沒有可定義的相對成長率。分母先 replace(0,np.nan)，避免 inf；0/0 為 NaN。題目若明確要求無訂單 AOV=0，才另外 fillna(0)。

## 2. Conditional features

**English prompt:** Flag merchants whose growth is at least 20% and whose February GMV is at least 100.

```python
df['flag'] = ((df.growth>=.2)&(df.feb_gmv>=100)).astype(int)
df['segment'] = np.where(df.feb_gmv>=100,'high','low')
```

flag 為 [1,0,0,0]；segment 為 [high,high,low,low]。np.where(condition, true_value, false_value) 回傳 array。Series.where 則在 condition 為 True 時保留原值，False 時替換。

```python
df['high_gmv_only'] = df.feb_gmv.where(df.feb_gmv>=100,0)
# [150,100,0,0]
```

多條件可用 np.select，但第一個 True 的條件勝出。常用比例、布林 flag、差值與交互特徵比冷門 API 更重要。

## 3. map vs apply vs vectorized operations

```python
df['tier'] = df.merchant.map({'A':'pro','B':'basic','C':'basic'})
df['tier'] = df.tier.fillna('unknown')
df['total'] = df.jan_gmv + df.feb_gmv
df['max_month'] = df[['jan_gmv','feb_gmv']].max(axis=1)
```

map 適合 value-to-value mapping，未匹配的 D 會變 NaN。apply 用於自訂函數，但可直接向量化的運算優先用向量化，通常更快且易讀。

```python
# Row-wise apply is possible but unnecessary here
df.apply(lambda row: row.jan_gmv+row.feb_gmv,axis=1)
```

assign 可一次新增多個欄位，回傳新 DataFrame；lambda 可引用前面已建立的欄位。cut/qcut 只需理解固定區間 vs 分位數分箱，遇到有明確要求再查參數。

## 4. Feature availability

建模前問：在 prediction cutoff 時，這個 feature 是否已知？如果要預測 2 月 GMV，2 月實際 GMV 不能作為 feature。過去交易、商家註冊年資、截至 1 月底的統計可以使用。

## Concept check

**A merchant has zero January revenue and positive February revenue. Is its month-over-month growth rate equal to 100%?**

<details><summary>Answer</summary>

No. 分母為零，relative growth 未定義。可另設 new_activity flag，但不能把它任意當作 100%。

</details>
