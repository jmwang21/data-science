# 01.2 — GroupBy & Aggregation

[← pandas 目錄](../01-pandas.md) · [Previous](01-filtering-ranking.md) · [Next →](03-merge.md)

## 1. Why GroupBy?

SQL 的 GROUP BY 概念可直接對應 pandas。先確定 output grain（輸出粒度），再決定是彙總成新表還是保留每筆交易。

| merchant | order | amount | status |
|---|---|---:|---|
| A | O1 | 100 | completed |
| A | O2 | 300 | cancelled |
| B | O3 | 200 | completed |
| B | O4 | NaN | completed |
| C | O5 | 50 | completed |

```python
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'merchant':['A','A','B','B','C'],
    'order':['O1','O2','O3','O4','O5'],
    'amount':[100,300,200,np.nan,50],
    'status':['completed','cancelled','completed','completed','completed']
})
```

## 2. agg — one row per group

**English prompt:** Calculate total revenue, transaction count and average valid amount for each merchant.

```python
summary = df.groupby('merchant', as_index=False).agg(
    revenue=('amount','sum'),
    transactions=('order','size'),
    avg_amount=('amount','mean'),
    valid_amounts=('amount','count')
)
```

| merchant | revenue | transactions | avg_amount | valid_amounts |
|---|---:|---:|---:|---:|
| A | 400 | 2 | 200 | 2 |
| B | 200 | 2 | 200 | 1 |
| C | 50 | 1 | 50 | 1 |

Named aggregation 的語法是 output_name=(input_column, aggregation)。as_index=False 讓 group key 保留為欄位，方便後續 merge。

`size` 計算列數；`count` 排除所選欄位 NaN；`nunique` 計算 distinct values（預設忽略缺失）。例如 B 有兩筆交易，但只有一筆有效金額，所以 mean 為 200 而非 100。

**注意：** sum 預設略過 NaN，整組皆 NaN 時預設得到 0；需要區分未知與零時可用 sum(min_count=1)。

## 3. transform — preserve the original rows

若題目要求每筆訂單占自己商家總額的比例，不能只輸出商家層級。

```python
df['merchant_total'] = df.groupby('merchant')['amount'].transform('sum')
df['share'] = df['amount'] / df['merchant_total'].replace(0,np.nan)
```

| merchant | amount | merchant_total | share |
|---|---:|---:|---:|
| A | 100 | 400 | 0.25 |
| A | 300 | 400 | 0.75 |
| B | 200 | 200 | 1.00 |
| B | NaN | 200 | NaN |
| C | 50 | 50 | 1.00 |

transform 把每組結果 broadcast 回原來的 rows；index 對齊原資料。直接把 agg 結果指定回原 df，會有 index/shape 不符問題。

## 4. Conditional aggregation

**English prompt:** Compute total transaction amount, completed GMV, and completion rate for each merchant.

```python
temp = df.assign(
    completed_amount=df.amount.where(df.status.eq('completed'),0),
    completed=df.status.eq('completed').astype(int)
)
res = temp.groupby('merchant',as_index=False).agg(
    total_amount=('amount','sum'),
    completed_gmv=('completed_amount','sum'),
    completion_rate=('completed','mean')
)
```

| merchant | total_amount | completed_gmv | completion_rate |
|---|---:|---:|---:|
| A | 400 | 100 | 0.5 |
| B | 200 | 200 | 1.0 |
| C | 50 | 50 | 1.0 |

布林值轉 0/1 後平均就是符合條件的比例。此處 completion rate 分母是全部交易；completed GMV 忽略未知金額。若題目要求先過濾無效金額，分母可能不同，務必確認。

不能一開始就把 cancelled 排除，否則 completion rate 恆為 100%。若只有 completed GMV 一個指標，先 filter 則很合理。

## 5. Other useful patterns

```python
# Retain entire groups with at least two transactions
large = df.groupby('merchant').filter(lambda g: len(g) >= 2)
# Number each row within its group
df['row_number'] = df.groupby('merchant').cumcount() + 1
```

large 保留 A、B 四筆；row_number 為 [1,2,1,2,1]。需要時間順序就先排序。groupby.apply 可處理複雜自訂運算，但常用的 agg/transform 已足夠，無需背進階參數。

加權比例：若 A 有 1/2 完成、B 有 90/100 完成，整體為 91/102，不是兩商家 completion rate 的平均。

## Concept check

**Which operation calculates the average amount per merchant while preserving the original number of rows?**

A. agg('mean') · B. transform('mean') · C. filter · D. size

<details><summary>Answer</summary>

B. transform 回傳與原資料對齊的群組平均；agg 是每組一列，filter 用於保留群組，size 計算筆數。

</details>
