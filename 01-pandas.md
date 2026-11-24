# 01 — Data Collection / pandas

![Chapter 01](assets/01-pandas.svg)

[返回目錄](README.md)

> 閱讀導向：先掌握核心觀念、語法與完整範例；Concept checks 可直接展開答案，非必須獨立考試。P0 為核心，P1 為延伸。

## 1. Comprehensive review — ingestion, indexing, transformation and text (P0) 

| Operation | Example | Key distinction |
|---|---|---|
| Inspect | `df.shape; df.info(); df.describe(include='all')` | Check types and grain before arithmetic |
| Select | `df.loc[:, ['amount','status']]` | Single column returns Series; list returns DataFrame |
| Unique | `s.unique(); s.nunique(); s.value_counts(dropna=False)` | Counts vs distinct values |
| Rename | `df.rename(columns={'amount':'gmv'})` | Returns new object by default |
| Sort | `df.sort_values(['amount','order_id'],ascending=[False,True])` | Explicit tie-break |
| Index | `df.set_index('order_id').reset_index()` | Index labels need not be unique |
| Filter | `df.loc[df.status.isin(['completed','paid'])]` | Membership test |
| Assign | `df.assign(net=lambda d:d.amount*0.9)` | Returns new frame |
| Concatenate | `pd.concat([a,b],ignore_index=True)` | Column names aligned |
| Cross-tab | `pd.crosstab(df.category,df.status,normalize='index')` | Row proportions |

### Text, mapping and conditional features

```python
s = pd.Series(['  ABC-123 ', 'def-456', None], dtype='string')
clean = s.str.strip().str.lower()
code = clean.str.extract(r'([a-z]+)-(\d+)', expand=True)
flag = clean.str.contains('abc', regex=False, na=False)
# clean: abc-123, def-456, <NA>; flag: True, False, False
df['bucket'] = pd.cut(df.amount, bins=[-np.inf,100,500,np.inf],
                      labels=['small','medium','large'])
df['segment'] = np.select([df.amount>=500, df.amount>=100],
                          ['large','medium'], default='small')
```

`map` maps individual Series values; `apply` calls a function on a Series or along a DataFrame axis; prefer vectorization where possible. `axis=1` applies to rows. `np.select` picks the first true condition, so order matters. `cut` uses fixed boundaries; `qcut` uses empirical quantiles and can fail on duplicate edges. Avoid chained assignment: use `df.loc[mask,'col']=value`. Series assignment aligns index labels, not just row positions.

### Additional aggregation and dates (P0)

```python
orders['order_date'] = pd.to_datetime(orders.order_date)
daily = orders.set_index('order_date').amount.resample('D').sum()
counts = orders.groupby('merchant_id',dropna=False).size()
large_groups = orders.groupby('merchant_id').filter(lambda g: len(g)>=2)
orders['pct_change'] = orders.sort_values('order_date').groupby('merchant_id').amount.pct_change(fill_method=None)
```

For a stable time series sort by entity and timestamp, decide how to treat gaps and ties, and distinguish event count from calendar duration. `groupby(dropna=False)` retains missing group keys. `explode` expands list-valued cells into rows; `stack/unstack` reshape index levels (P1). Do not confuse weighted averages with averages of group averages.

Reference: [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html).

## 2. Filtering & Ranking 

`loc` 依標籤選取，`iloc` 依位置選取。布林條件用 `&`、`|`、`~`，每個比較式加括號，不用 Python 的 `and` / `or`。

```python
import pandas as pd
import numpy as np
orders = pd.DataFrame({
    'merchant_id': ['A','A','B','B'],
    'amount': [100,200,500,100],
    'status': ['completed','cancelled','completed','completed'],
    'order_date': ['2025-01-10','2025-02-10','2025-01-20','2025-02-20']
})
orders.loc[(orders.amount >= 100) & orders.status.eq('completed'), ['merchant_id','amount']]
orders.sort_values(['amount','merchant_id'], ascending=[False,True])
orders.nlargest(2,'amount')
```

`rank(method='dense', ascending=False)`：100,100,80 → 1,1,2；`min` → 1,1,3；`first` → 1,2,3。每個 category 的排名必須先 groupby。

```python
result['rank'] = result.groupby('category')['gmv'].rank(method='dense', ascending=False)
```

## 3. GroupBy & Aggregation 

```python
res = orders.groupby('merchant_id', as_index=False).agg(
    total_gmv=('amount','sum'),
    total_orders=('amount','size'),
    avg_amount=('amount','mean')
)
```

| merchant_id | total_gmv | total_orders | avg_amount |
|---|---:|---:|---:|
| A | 300 | 2 | 150 |
| B | 600 | 2 | 300 |

`size` 計算所有列；`count` 排除該欄缺失；`nunique` 預設排除缺失。`agg` 縮減每組為一列；`transform` 回傳與原資料相同長度。

```python
orders['share'] = orders.amount / orders.groupby('merchant_id').amount.transform('sum')
```

### Conditional Aggregation（Completed GMV）

同時計算全部金額與完成金額時，不可先過濾 cancelled。

```python
res = orders.assign(
    completed_amount=lambda x: x.amount.where(x.status.eq('completed'),0),
    is_completed=lambda x: x.status.eq('completed').astype(int)
).groupby('merchant_id',as_index=False).agg(
    total_gmv=('amount','sum'),
    completed_gmv=('completed_amount','sum'),
    completion_rate=('is_completed','mean')
)
```

A: 300,100,0.5；B: 600,600,1.0。`where(condition, other)` 成立保留原值，否則替換。若只需要 completed GMV，可先 filter 再 groupby，但無完成訂單的商家需另行保留。

## 4. Merge & Join 

| Join | 保留範圍 |
|---|---|
| inner | 兩邊匹配 |
| left | 左表全部 |
| right | 右表全部 |
| outer | 兩邊全部 |

```python
merchants = pd.DataFrame({'merchant_id':['A','B','C'],'category':['Fashion','Fashion','Tech']})
result = merchants.merge(res,on='merchant_id',how='left',validate='one_to_one',indicator=True)
```

`validate` 可檢查 one_to_one / one_to_many / many_to_one。多對多會產生笛卡兒式配對，可能膨脹 GMV。先檢查 key 重複、資料粒度與列數。pandas merge 的 null keys 可能彼此匹配，與 SQL 不同。

**重要錯題：** left join 後再 filter `amount.notna()` 會刪掉沒有訂單的商家。應先清理 orders、聚合，再 left join 完整 merchants。只對業務定義為零的欄位 fillna(0)。

## 5. Pivot / Reshape 

```python
orders['month'] = pd.to_datetime(orders.order_date).dt.to_period('M')
wide = orders.pivot_table(index='merchant_id',columns='month',values='amount',aggfunc='sum',fill_value=0)
long = wide.reset_index().melt(id_vars='merchant_id',var_name='month',value_name='gmv')
```

`pivot` 要求 index/column 組合唯一；`pivot_table` 可聚合重複組合，預設 mean，GMV 應指定 sum。`melt` 將寬表轉長表。缺少整個月份時用 `reindex(columns=..., fill_value=0)`。

## 6. DateTime 

```python
d = pd.to_datetime(orders.order_date,errors='coerce')
orders.loc[d.between('2025-01-01','2025-03-01',inclusive='left')]
d.dt.year; d.dt.month; d.dt.day; d.dt.to_period('M')
(pd.Timestamp('2025-03-01')-d).dt.days
```

`between` 預設 both；left 為包含起點、不含終點，適合有時分秒的月份篩選。日期相減是 Timedelta，`.dt.days` 才是整數天數。商家 tenure 應在 merchant 粒度計算，不需在訂單上取平均。

## 7. Window Functions 

```python
monthly = orders.groupby(['merchant_id','month'],as_index=False).amount.sum()
monthly = monthly.sort_values(['merchant_id','month'])
g = monthly.groupby('merchant_id').amount
monthly['prev'] = g.shift(1)
monthly['growth'] = (monthly.amount-monthly.prev)/monthly.prev.replace(0,np.nan)
monthly['rolling_2'] = g.transform(lambda s:s.rolling(2,min_periods=1).mean())
monthly['past_2'] = g.transform(lambda s:s.shift(1).rolling(2,min_periods=1).mean())
```

`shift(1)` 前一筆；`diff()` 差值；`cumsum()` 累計；rolling 是依列數而非日曆月份。若月份缺漏，先補齊月份。預測當期時，rolling feature 不應含當期尚未知的 target。

## 8. Additional patterns & edge cases 

### Top N per group vs dense rank

```python
# Exactly two rows per category (if available), deterministic tie-break
out = result.sort_values(['category','gmv','merchant_id'],ascending=[True,False,True]).groupby('category').head(2)
# Top two distinct GMV ranks, potentially more than two rows
out = result.loc[result.groupby('category').gmv.rank(method='dense',ascending=False)<=2]
```

### GroupBy transform vs aggregate

For GMV [100,200] in A, [500,100] in B: `.agg('sum')` returns A=300, B=600; `.transform('sum')` returns [300,300,600,600] aligned to original rows. Use transform for contribution, group average comparison, and row-level features.

### Join cardinality example

Left A,A and right A,A produce four rows for key A. If each left row represents an order, its amount is counted twice. Use a unique merchant dimension or aggregate at the required grain before join; do not blindly call drop_duplicates to hide the issue.

### Month-over-month growth

| Merchant | Jan | Feb | Growth |
|---|---:|---:|---:|
| A | 100 | 200 | 100% |
| B | 500 | 100 | -80% |
| C | 0 | 600 | NaN (undefined) |

The calculation requires a complete merchant-month grid if missing months represent zero sales. Otherwise shift may compare March with January instead of February. A missing observation and a confirmed zero are different business states.

### Practice — rank and monthly growth

For each category, select merchants in the top two distinct GMV ranks. Then explain why ranking before the merchant population is restored might change the answer.

<details><summary>Answer</summary>

Group by category, rank descending with dense method, then filter rank<=2. Restore merchant population first because zero-GMV merchants may qualify when only one or no positive GMV ranks exist. Whether they should qualify is a separate business constraint.
</details>

## 9. Concept check 

For each merchant, calculate completed GMV, order count, and share of total completed GMV. Retain merchants without orders.

<details><summary>Reference solution</summary>

```python
a = orders.loc[orders.status.eq('completed')].groupby('merchant_id',as_index=False).agg(gmv=('amount','sum'),n=('amount','size'))
r = merchants.merge(a,how='left',on='merchant_id')
r[['gmv','n']] = r[['gmv','n']].fillna(0)
total = r.gmv.sum()
r['share'] = r.gmv/total if total else np.nan
```
</details>

## 10. Additional core patterns — joins and time alignment (P0)

`merge(left_on='customer_id',right_on='id')` handles differing key names; `suffixes=('_left','_right')` resolves overlapping non-key columns. Anti-join pattern: left merge with indicator=True then keep _merge=='left_only'. `pd.concat([a,b],axis=1)` aligns by index, not row position. For SQL-like row_number use deterministic sort then groupby.cumcount()+1. Use `pd.to_datetime(...,utc=True)` for timestamps with offsets, convert to business timezone before deriving dates.

<details><summary>Concept check: why can a left merge increase row count?</summary>

A left join guarantees retention, not uniqueness. Each matching right row produces a result. Validate many_to_one when joining a dimension to event records.

</details>

[下一章：Preprocessing →](02-preprocessing.md)
