# 01.4 — Missing Values & Data Cleaning

[← pandas 目錄](../01-pandas.md) · [Previous](03-merge.md) · [Next →](05-features.md)

## 1. Inspect data before cleaning

| order_id | amount | status |
|---|---|---|
| O1 | 100 | completed |
| O2 | bad | completed |
| O3 | None | cancelled |
| O4 | 0 | completed |
| O4 | 0 | completed |

```python
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'order_id':['O1','O2','O3','O4','O4'],
    'amount':['100','bad',None,'0','0'],
    'status':['completed','completed','cancelled','completed','completed']
})
df.isna().sum()   # amount: 1
df.isna().mean()  # amount: 0.2
df.duplicated().sum() # 1
```

isna() 回傳 Boolean DataFrame，sum 計算 True 數，mean 計算缺失比例。count 計算非缺失值，因此 isna().count() 不是 missing count。

## 2. Data type conversion

**English prompt:** Convert amount to a numeric column, coercing invalid values to missing.

```python
df['amount'] = pd.to_numeric(df.amount, errors='coerce')
```

amount 變為 [100, NaN, NaN, 0, 0]，缺失比例成為 40%。原本 bad 是字串，不是缺失，轉型後才變 NaN。

astype 適合資料已符合目標型別；to_numeric/to_datetime 的 errors='coerce' 可將非法值轉成 NaN/NaT。不要把 ID 當數值轉換。

## 3. Missing values, duplicates and invalid values

```python
clean = df.drop_duplicates(subset=['order_id'], keep='first')
valid = clean.dropna(subset=['amount'])
```

valid 保留 O1、O4。去重需確認 business key：若一張訂單有多個 line items，不應只依 order_id 去重。keep='last' 是保留最後出現的列，不代表最新時間；需要最新紀錄時先排序。

| Method | What it does |
|---|---|
| dropna(subset=['amount']) | Remove rows missing amount |
| fillna(0) | Replace missing values with zero |
| replace(['bad','unknown'], np.nan) | Replace specified values |
| map({'basic':0,'pro':1}) | Map values; unmatched values usually become missing |

0 是實際觀察值，NaN 是未知。填零必須由業務定義決定。若是 ML imputation，中位數、平均值等應從 training data 學習，詳見 [Preprocessing](../02-preprocessing.md)。

## 4. String cleaning

```python
s = pd.Series(['  BASIC ', 'Pro', None],dtype='string')
s.str.strip().str.lower() # basic, pro, <NA>
s.str.contains('pro',case=False,regex=False,na=False)
# False, True, False
```

常用 strip、lower、contains、replace 足夠應付大部分清理；regex 只需熟悉簡單擷取及匹配，不必大量背特殊語法。

## Concept check

**Which operation counts missing values in each column: df.isna().sum() or df.isna().count()?**

<details><summary>Answer</summary>

df.isna().sum()。isna() 的結果是 True/False，count 會把兩者都算作有效值。

</details>
