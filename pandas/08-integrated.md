# 01.8 — Integrated pandas: from transactions to a feature table

[← pandas 目錄](../01-pandas.md) · [Previous](07-datetime.md) · [Next: Preprocessing →](../02-preprocessing.md)

此章是閱讀範例，不是限時測驗。把前七章的核心操作串起來。

## 1. Problem statement (English)

Given a merchant dimension and an order table, compute each merchant's completed GMV and order count for January and February. Retain all merchants, including those with no completed orders. Calculate month-over-month GMV growth, treating undefined growth as missing. Return one row per merchant.

**解讀：** merchant universe 由主表決定；先處理交易，再彙總；分母為零的 growth 保留 NaN；最後檢查 grain。

## 2. Input

```python
import pandas as pd
import numpy as np
merchants = pd.DataFrame({'merchant':['A','B','C','D'],
                          'category':['Fashion','Fashion','Tech','Tech']})
orders = pd.DataFrame({
    'merchant':['A','A','A','B','B','C'],
    'date':['2025-01-05','2025-02-01','2025-02-05',
            '2025-01-10','2025-02-15','2025-02-20'],
    'amount':[100,150,80,200,100,50],
    'status':['completed','completed','cancelled','completed','completed','completed']
})
```

## 3. Clean and filter

```python
orders['date'] = pd.to_datetime(orders.date,errors='coerce')
orders['amount'] = pd.to_numeric(orders.amount,errors='coerce')
valid = orders.loc[
    orders.status.eq('completed') &
    orders.amount.notna() &
    orders.date.between('2025-01-01','2025-03-01',inclusive='left')
].copy()
```

A 的 cancelled 80 被排除。其他五筆保留。

## 4. Build merchant-level features

```python
valid['jan'] = valid.amount.where(valid.date.dt.month.eq(1),0)
valid['feb'] = valid.amount.where(valid.date.dt.month.eq(2),0)
agg = valid.groupby('merchant',as_index=False).agg(
    jan_gmv=('jan','sum'),
    feb_gmv=('feb','sum'),
    order_count=('amount','size')
)
```

| merchant | jan_gmv | feb_gmv | order_count |
|---|---:|---:|---:|
| A | 100 | 150 | 2 |
| B | 200 | 100 | 2 |
| C | 0 | 50 | 1 |

用 conditional aggregation 直接建立月份欄位，不一定需要 pivot。當題目只要求幾個固定月份，這個方式通常很清楚。

## 5. Restore the merchant population

```python
out = merchants.merge(agg,on='merchant',how='left',validate='one_to_one')
cols = ['jan_gmv','feb_gmv','order_count']
out[cols] = out[cols].fillna(0)
out['order_count'] = out.order_count.astype(int)
out['growth'] = (out.feb_gmv-out.jan_gmv)/out.jan_gmv.replace(0,np.nan)
out['total_gmv'] = out.jan_gmv+out.feb_gmv
out['category_rank'] = out.groupby('category').total_gmv.rank(
    method='dense',ascending=False).astype(int)
assert len(out)==len(merchants)
assert out.merchant.is_unique
```

| merchant | category | jan_gmv | feb_gmv | order_count | growth | total_gmv | category_rank |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Fashion | 100 | 150 | 2 | 0.5 | 250 | 2 |
| B | Fashion | 200 | 100 | 2 | -0.5 | 300 | 1 |
| C | Tech | 0 | 50 | 1 | NaN | 50 | 1 |
| D | Tech | 0 | 0 | 0 | NaN | 0 | 2 |

D 必須存在，因為題意要求保留全體商家。若接下來要用此資料預測未來，還需將 label period 與 feature period 分離，避免 leakage。

## 6. Self-review

<details><summary>Why not fill all missing values with zero after the merge?</summary>

這裡只補 jan/feb GMV 與 order count，因為題目定義無有效交易時為零。growth 的分母為零時則是未定義，不能任意改為 0。

</details>

<details><summary>What if orders and support tickets are provided in separate CSV files?</summary>

先依邏輯表讀取、清理與 concat 各自分區，再分別 aggregate 到 merchant 粒度，最後 merge。不要將兩張事件表直接按 merchant key join。參考 [多檔案整合](../08-numpy-data.md)。

</details>

更完整的商業案例見 [Integrated Cases](../06-cases.md)。需要模型實作時再進入 [Preprocessing](../02-preprocessing.md)。
