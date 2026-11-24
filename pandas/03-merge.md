# 01.3 — Merge, Join & Concat

[← pandas 目錄](../01-pandas.md) · [Previous](02-groupby.md) · [Next →](04-cleaning.md)

## 1. Dataset and grain

Merchant dimension（每商家一列）：

| merchant_id | category |
|---|---|
| A | Fashion |
| B | Tech |
| C | Beauty |

Order table（每訂單一列）：

| order_id | merchant_id | amount |
|---|---|---:|
| O1 | A | 100 |
| O2 | A | 300 |
| O3 | B | 200 |
| O4 | D | 80 |

```python
import pandas as pd
merchants = pd.DataFrame({'merchant_id':['A','B','C'],
                          'category':['Fashion','Tech','Beauty']})
orders = pd.DataFrame({'order_id':['O1','O2','O3','O4'],
                       'merchant_id':['A','A','B','D'],
                       'amount':[100,300,200,80]})
```

## 2. Join types

```python
left = merchants.merge(orders,on='merchant_id',how='left')
```

| merchant_id | category | order_id | amount |
|---|---|---|---:|
| A | Fashion | O1 | 100 |
| A | Fashion | O2 | 300 |
| B | Tech | O3 | 200 |
| C | Beauty | NaN | NaN |

| how | Result |
|---|---|
| inner | Only matched keys: A (2 rows), B |
| left | All merchant rows, including C |
| right | All order rows, including D |
| outer | Matched rows plus C and D |

常用參數：on 為共同 key；left_on/right_on 用於不同欄名；how 控制保留哪邊；suffixes 避免非 key 欄同名；validate 檢查預期關係。

## 3. Aggregate before merging

**English prompt:** Return one row per merchant with total GMV and order count. Retain merchants without transactions.

若直接 merge，A 會有兩筆，不符合一商家一列。先將 orders 彙總成 merchant grain，再接 merchant dimension。

```python
summary = orders.groupby('merchant_id',as_index=False).agg(
    gmv=('amount','sum'), n_orders=('order_id','size'))
out = merchants.merge(summary,on='merchant_id',how='left',
                      validate='one_to_one')
out[['gmv','n_orders']] = out[['gmv','n_orders']].fillna(0)
out['n_orders'] = out.n_orders.astype(int)
```

| merchant_id | category | gmv | n_orders |
|---|---|---:|---:|
| A | Fashion | 400 | 2 |
| B | Tech | 200 | 1 |
| C | Beauty | 0 | 0 |

D 不在目標母體，故不輸出。此例將無交易定義為 0；真實資料若無法區分缺資料與零，不能盲目補值。

## 4. Cardinality & fanout

A 有 2 筆 orders、3 筆 support tickets，直接 on merchant_id merge 產生 6 列。每筆 order 金額被重複三次。這不是 drop_duplicates 可以可靠修復的問題。

| validate | Meaning |
|---|---|
| one_to_one | Both keys unique |
| one_to_many | Left key unique |
| many_to_one | Right key unique |
| many_to_many | Allows both sides nonunique; no uniqueness check |

```python
# Every order should match at most one merchant dimension row
checked = orders.merge(merchants,on='merchant_id',how='left',
                       validate='many_to_one',indicator=True)
unknown = checked.loc[checked['_merge'].eq('left_only')]
```

unknown 是 O4。indicator 用於對帳（both、left_only、right_only）。pandas 的 null join keys 可能彼此匹配，先檢查 key 缺失。左接後再用右表欄位篩選，可能把原本應保留的未匹配列刪掉。

## 5. Concat — same logical table, different files

```python
jan = orders.iloc[:2]
feb = orders.iloc[2:]
all_orders = pd.concat([jan,feb],ignore_index=True)
```

concat axis=0 縱向疊資料，依欄名對齊，ignore_index=True 重建 index。axis=1 橫向依 index 對齊，而不是依 merchant_id。merge 才是依 business key 配對。join 通常用 index 做對應，可先認識即可。

多檔讀取、schema normalization 另見 [Python/NumPy & Multi-file Data Integration](../08-numpy-data.md)。

## Variation — filtering before and after joins

**English:** Retain all merchants but count only orders with amount above 200.

```python
eligible = orders.loc[orders.amount>200]
counts = eligible.groupby('merchant_id').size().rename('n').reset_index()
res = merchants.merge(counts,on='merchant_id',how='left',validate='one_to_one')
res['n'] = res.n.fillna(0).astype(int)
# A:1, B:0, C:0
```

If you instead left join and then filter amount>200, B and C disappear. If the output should include all merchants, preserve the target population.

<details><summary>Variation: An orders table has 100 rows; after left joining a merchant dimension it has 105. What should you inspect first?</summary>

Right-side duplicate keys and join cardinality. For one merchant record per key, validate='many_to_one' should fail when right keys duplicate. Audit null keys and key types too.

</details>

<details><summary>Variation: January and February have the same columns but different row indexes. Should they be merged on the index?</summary>

No. Same logical table partitions normally concatenate row-wise using ignore_index=True; join by an actual business key only when the two tables represent related entities.

</details>

## Concept check

**A merchant has three orders and two support tickets. How many rows can a direct inner merge on merchant_id produce?**

<details><summary>Answer</summary>

6 rows. 兩事件表都是 one-to-many，直接 merge 形成 many-to-many；先分別彙總成商家粒度再 merge。

</details>
