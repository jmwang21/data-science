# 06.1 — Merchant Performance & Growth Opportunity

[← Cases index](../06-cases.md)

> Worked case · 閱讀解題思路及完整解法，觀念變化可直接展開。所有資料為合成。

You are a data scientist at an e-commerce platform. Identify high-potential merchants using completed orders in January and February 2025. Retain all merchants, including those with no valid orders.

### Input

```python
import pandas as pd
import numpy as np
merchants = pd.DataFrame([
 ('M001','Fashion','2024-01-10'),('M002','Fashion','2024-03-15'),
 ('M003','Electronics','2024-02-01'),('M004','Electronics','2024-06-20'),
 ('M005','Fashion','2024-08-01'),('M006','Electronics','2024-09-10')
],columns=['merchant_id','category','join_date'])
orders = pd.DataFrame([
 ('O001','M001','C01','2025-01-05',100,'completed'),
 ('O002','M001','C02','2025-01-20',200,'completed'),
 ('O003','M001','C01','2025-02-10',400,'completed'),
 ('O004','M001','C03','2025-02-15',300,'completed'),
 ('O005','M002','C04','2025-01-08',200,'completed'),
 ('O006','M002','C04','2025-02-12',250,'completed'),
 ('O007','M002','C05','2025-02-20',150,'cancelled'),
 ('O008','M003','C06','2025-01-10',500,'completed'),
 ('O009','M003','C07','2025-02-05',800,'completed'),
 ('O010','M003','C08','2025-02-18',200,'completed'),
 ('O011','M004','C09','2025-02-01',600,'completed'),
 ('O012','M004','C10','2025-02-15',np.nan,'completed'),
 ('O013','M005','C11','2025-01-15',100,'completed'),
 ('O014','M005','C12','2025-02-25',50,'completed')
],columns=['order_id','merchant_id','customer_id','order_date','amount','status'])
```

| Part | Requirements | Points |
|---|---|---:|
| A | Parse dates; completed, nonmissing amount, Jan–Feb 2025 | 5 |
| B | total_gmv, total_orders, unique_customers, avg_order_value | 5 |
| C | jan_gmv, feb_gmv, growth; undefined if Jan=0 | 6 |
| D | Retain merchants; zero-fill counts and GMV/AOV; tenure as of March 1 | 6 |
| E | Dense GMV rank descending within category; rank≤2 AND growth≥.2 AND customers≥2 → 1 | 5 |
| F | Return specified columns, sorted category/rank/merchant_id ascending | 3 |

### Expected output (split for reading)

| merchant_id | category | tenure_days | total_gmv | total_orders | unique_customers | avg_order_value |
|---|---|---:|---:|---:|---:|---:|
| M003 | Electronics | 394 | 1500 | 3 | 3 | 500 |
| M004 | Electronics | 254 | 600 | 1 | 1 | 600 |
| M006 | Electronics | 172 | 0 | 0 | 0 | 0 |
| M001 | Fashion | 416 | 1000 | 4 | 3 | 250 |
| M002 | Fashion | 351 | 450 | 2 | 1 | 225 |
| M005 | Fashion | 212 | 150 | 2 | 2 | 75 |

| merchant_id | jan_gmv | feb_gmv | gmv_growth | category_rank | high_potential |
|---|---:|---:|---:|---:|---:|
| M003 | 500 | 1000 | 1 | 1 | 1 |
| M004 | 0 | 600 | NaN | 2 | 0 |
| M006 | 0 | 0 | NaN | 3 | 0 |
| M001 | 300 | 700 | 1.3333 | 1 | 1 |
| M002 | 200 | 250 | .25 | 2 | 0 |
| M005 | 100 | 50 | -.5 | 3 | 0 |

<details><summary>Reference solution</summary>

```python
def analyze_merchant_performance(merchants, orders):
    valid = orders.assign(order_date=lambda x: pd.to_datetime(x.order_date)).loc[
        lambda x: x.status.eq('completed') & x.amount.notna() &
        x.order_date.between('2025-01-01','2025-03-01',inclusive='left')]
    agg = valid.assign(
        jan=lambda x: x.amount.where(x.order_date.dt.month.eq(1),0),
        feb=lambda x: x.amount.where(x.order_date.dt.month.eq(2),0)
    ).groupby('merchant_id',as_index=False).agg(
        total_gmv=('amount','sum'),total_orders=('order_id','size'),
        unique_customers=('customer_id','nunique'),jan_gmv=('jan','sum'),feb_gmv=('feb','sum'))
    r = merchants.assign(join_date=lambda x: pd.to_datetime(x.join_date)).merge(
        agg,on='merchant_id',how='left',validate='one_to_one')
    cols=['total_gmv','total_orders','unique_customers','jan_gmv','feb_gmv']
    r[cols]=r[cols].fillna(0)
    r=r.assign(
        tenure_days=lambda x: (pd.Timestamp('2025-03-01')-x.join_date).dt.days,
        avg_order_value=lambda x: (x.total_gmv/x.total_orders.replace(0,np.nan)).fillna(0),
        gmv_growth=lambda x: (x.feb_gmv-x.jan_gmv)/x.jan_gmv.replace(0,np.nan),
        category_rank=lambda x: x.groupby('category').total_gmv.rank(method='dense',ascending=False).astype(int),
        high_potential=lambda x: ((x.category_rank<=2)&(x.gmv_growth>=.2)&(x.unique_customers>=2)).astype(int))
    cols=['merchant_id','category','tenure_days','total_gmv','total_orders','unique_customers',
          'avg_order_value','jan_gmv','feb_gmv','gmv_growth','category_rank','high_potential']
    return r[cols].sort_values(['category','category_rank','merchant_id']).reset_index(drop=True)
```

常見錯誤：先 join 再 filter 導致無訂單商家遺失；未處理 infinity；日期基準錯誤；全域排名；混淆欄位改名與選取。
</details>

[← Cases index](../06-cases.md)
