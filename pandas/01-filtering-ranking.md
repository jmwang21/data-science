# 01.1 — Filtering & Ranking

[← pandas 目錄](../01-pandas.md) · [Next: GroupBy & Aggregation →](02-groupby.md)

> **Core review** — 先掌握讀取、選取、條件篩選、排序和排名。不必背所有 pandas API。

## 1. Dataset and inspection

以下是訂單層級（order-level）資料，一列代表一張訂單。後續所有例子都用同一份資料。

| order_id | merchant_id | category | amount | status |
|---|---|---|---:|---|
| O01 | A | Fashion | 100 | completed |
| O02 | A | Fashion | 300 | cancelled |
| O03 | B | Fashion | 300 | completed |
| O04 | C | Tech | 500 | completed |
| O05 | D | Tech | 200 | completed |

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'order_id': ['O01','O02','O03','O04','O05'],
    'merchant_id': ['A','A','B','C','D'],
    'category': ['Fashion','Fashion','Fashion','Tech','Tech'],
    'amount': [100,300,300,500,200],
    'status': ['completed','cancelled','completed','completed','completed']
})
```

若題目提供 CSV，常見起手式：

```python
# df = pd.read_csv('orders.csv', dtype={'order_id': 'string'})
df.head()
df.shape       # (5, 5)
df.dtypes
df.info()
df['status'].value_counts()
```

先確認 column names、data types、missing values、business key 是否唯一。ID 不應因看起來像數字就轉成 integer（可能失去前導零）。`shape` 是屬性，不是 method。

## 2. Selecting rows and columns

### loc vs iloc

`loc` 使用 label；`iloc` 使用整數位置。

```python
df.loc[1:3, ['order_id','amount']]
df.iloc[1:3, [0,3]]
```

第一個回傳 O02、O03、O04；第二個回傳 O02、O03。原因是 label slicing 包含終點（在這個唯一且連續的 index 下），position slicing 不包含終點。

`df['amount']` 是 Series，`df[['amount']]` 是一欄 DataFrame。模型的 X 通常要 2D，y 通常是 1D。

## 3. Boolean filtering

**English prompt:** Select completed orders with an amount of at least 200. Return only the order ID and amount.

```python
mask = (df['status'] == 'completed') & (df['amount'] >= 200)
out = df.loc[mask, ['order_id','amount']]
```

| order_id | amount |
|---|---:|
| O03 | 300 |
| O04 | 500 |
| O05 | 200 |

- `&`: AND；`|`: OR；`~`: NOT。每個比較式括號包好。
- `isin([...])` 適合多個可能值；`between(lo,hi)` 預設含兩端。
- `df.loc[mask,'amount'] = 0` 是更新符合條件的列；不要使用 chained assignment。

**at least 200** = ≥200；**greater than 200** = >200；**at most 200** = ≤200。

## 4. Sorting and Top N

```python
out = df.sort_values(['amount','order_id'], ascending=[False,True])
top = out.head(3)
```

結果排序 O04、O02、O03、O05、O01；top 為前三筆。第二個排序 key 用來處理 ties（同分）。若只需單欄前 N 筆，`nlargest(3,'amount')` 也可。

## 5. Ranking within a group

**English prompt:** Rank orders by amount within each category, using dense ranking in descending order.

```python
df['rank'] = df.groupby('category')['amount'].rank(
    method='dense', ascending=False
)
```

| order_id | category | amount | rank |
|---|---|---:|---:|
| O01 | Fashion | 100 | 2 |
| O02 | Fashion | 300 | 1 |
| O03 | Fashion | 300 | 1 |
| O04 | Tech | 500 | 1 |
| O05 | Tech | 200 | 2 |

對 [300,300,100]，dense = [1,1,2]，min = [1,1,3]，first = [1,2,3]。dense top 1 可能回傳兩列；若題意是 exactly one row per category，就先 sort 再 groupby.head(1)。

```python
one = df.sort_values(['category','amount','order_id'],
                     ascending=[True,False,True]).groupby('category').head(1)
```

## Variation — different meaning of Top N

**English:** Select the two highest-ranked orders per category, breaking ties by ascending order ID.

This means exactly two rows per category if at least two exist, not the top two distinct amount values.

```python
top2 = (df.sort_values(['category','amount','order_id'],
                       ascending=[True,False,True])
          .groupby('category').head(2))
# Fashion: O02, O03; Tech: O04, O05
```

If wording is "include ties at the second distinct rank", use dense rank <= 2 instead: Fashion O01 is also retained. If the requirement first excludes cancelled orders, apply filtering before ranking, otherwise cancelled O02 may wrongly affect ranks.

<details><summary>Variation: How would you return only merchants A or B whose amount is not between 100 and 200 inclusive?</summary>

`df.loc[df.merchant_id.isin(['A','B']) & ~df.amount.between(100,200)]`. O02, O03. Parentheses or clear masks prevent precedence mistakes.

</details>

## Concept check

1. Which operator combines two element-wise Boolean conditions in pandas: and or &?
2. If two orders tie for first place, does dense rank <= 1 always return one order?

<details><summary>Answers & explanation</summary>

1. &；and 嘗試把整個 Series 當成一個 Boolean。
2. No. Dense rank gives both rank 1. 若需要固定筆數，使用 deterministic tie-break + head。

</details>
