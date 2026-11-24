# 01.6 — Reshaping & Window Functions

[← pandas 目錄](../01-pandas.md) · [Previous](05-features.md) · [Next →](07-datetime.md)

## 1. Long vs wide format (P1)

Long format 每列代表商家月份：

| merchant | month | amount |
|---|---|---:|
| A | Jan | 100 |
| A | Jan | 50 |
| A | Feb | 300 |
| B | Jan | 200 |

```python
import pandas as pd
import numpy as np
df = pd.DataFrame({'merchant':['A','A','A','B'],
                   'month':['Jan','Jan','Feb','Jan'],
                   'amount':[100,50,300,200]})
wide = df.pivot_table(index='merchant',columns='month',
                      values='amount',aggfunc='sum',fill_value=0)
wide = wide.reindex(columns=['Jan','Feb'],fill_value=0)
```

| merchant | Jan | Feb |
|---|---:|---:|
| A | 150 | 300 |
| B | 200 | 0 |

pivot 要求 merchant/month 組合唯一；此處 A/Jan 重複，因此使用 pivot_table 聚合。aggfunc 預設 mean，GMV 必須指定 sum。fill_value=0 只在無紀錄可視為零時合理。

```python
long = wide.reset_index().melt(id_vars='merchant',
                               var_name='month',value_name='amount')
```

melt 把 Jan、Feb 變回列。會得到四筆商家月份，不會恢復 A/Jan 原始兩筆交易（聚合不可逆）。

**到此為止已足夠掌握主要 reshape 用法。** stack/unstack 只需知道它們移動 index levels，不必深入背 MultiIndex 細節。

## 2. Window functions (P0)

**English prompt:** For each merchant, compute previous-month revenue, growth and a two-month rolling average.

| merchant | month | gmv |
|---|---|---:|
| A | 2025-01 | 100 |
| A | 2025-02 | 200 |
| A | 2025-03 | 300 |
| B | 2025-01 | 50 |
| B | 2025-02 | 100 |

```python
monthly = pd.DataFrame({'merchant':['A','A','A','B','B'],
                        'month':pd.PeriodIndex(['2025-01','2025-02','2025-03',
                                               '2025-01','2025-02'],freq='M'),
                        'gmv':[100,200,300,50,100]})
monthly = monthly.sort_values(['merchant','month'])
g = monthly.groupby('merchant')['gmv']
monthly['prev'] = g.shift(1)
monthly['growth'] = (monthly.gmv-monthly.prev)/monthly.prev.replace(0,np.nan)
monthly['rolling_2'] = g.transform(lambda s:s.rolling(2,min_periods=1).mean())
monthly['cumulative'] = g.cumsum()
```

| merchant | month | prev | growth | rolling_2 | cumulative |
|---|---|---:|---:|---:|---:|
| A | Jan | NaN | NaN | 100 | 100 |
| A | Feb | 100 | 1.0 | 150 | 300 |
| A | Mar | 200 | 0.5 | 250 | 600 |
| B | Jan | NaN | NaN | 50 | 50 |
| B | Feb | 50 | 1.0 | 75 | 150 |

shift(1) 取得群組上一列；diff() 計算差；cumsum() 累計；rolling(2) 使用當前與上一列。min_periods=1 允許不足兩筆時仍計算。

## 3. Past-only features

若預測當月結果，當月實際 GMV 不能進入 rolling feature：

```python
monthly['past_2'] = g.transform(
    lambda s:s.shift(1).rolling(2,min_periods=1).mean()
)
```

A 的 past_2 是 [NaN,100,150]。另注意 shift 是上一列而不是必然上一個月，缺少二月時三月可能連到一月。要計算 calendar-month growth，先建完整時間格或明確處理缺期。

## Concept check

**Why should the data be sorted before applying a grouped shift?**

<details><summary>Answer</summary>

shift 根據目前列順序找上一筆，不會自動按日期排序。若排序不正確，lag 與 growth 就可能指向錯誤時間，甚至使用未來資訊。

</details>
