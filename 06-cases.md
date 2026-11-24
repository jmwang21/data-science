# 06 — Integrated Business Cases

![Chapter 06](assets/06-cases.svg)

[返回目錄](README.md)

兩題分別對應 pandas 與 sklearn。先獨立解題，再展開答案。均使用合成資料；兩個案例均為原創合成練習，非正式考題。

## Case A — Merchant Performance & Growth Opportunity

**30 points · 25–30 min**

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

## Case B — Merchant Churn Prediction

**30 points · 35–45 min · 新練習**

Predict merchant churn using numeric and categorical features. Use only features available at the prediction date. The synthetic dataset below is for workflow practice, not a real churn dataset.

```python
import numpy as np
import pandas as pd
rng=np.random.default_rng(42)
n=600
X=pd.DataFrame({'tenure_days':rng.integers(10,1000,n),
                'monthly_gmv':rng.lognormal(7,1,n),
                'order_count':rng.poisson(15,n),
                'merchant_tier':rng.choice(['basic','pro','enterprise'],n)})
logit=1.0-.002*X.tenure_days-.0003*X.monthly_gmv-.06*X.order_count+.5*X.merchant_tier.eq('basic')
y=pd.Series(rng.binomial(1,1/(1+np.exp(-logit))),name='churn')
X.loc[rng.choice(n,30,replace=False),'monthly_gmv']=np.nan
```

Requirements:

- A (5): stratified train/validation/test 60/20/20, random_state=42.
- B (6): median impute numerical + scale; most frequent categorical impute + OHE ignore unknown; ColumnTransformer/Pipeline.
- C (7): compare LogisticRegression C=[.1,1,10] and RandomForest max_depth=[3,5,None], using 5-fold stratified ROC-AUC GridSearchCV on train only. Select best by validation ROC-AUC.
- D (6): maximize recall on validation with precision≥.60, tie highest threshold among [.3,.4,.5,.6,.7]. Handle no feasible threshold explicitly.
- E (6): return model name, params, validation scores, selected threshold, test AUC and precision/recall/F1 when a feasible threshold exists. Never tune with test.

<details><summary>Reference solution</summary>

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def churn_case(X,y):
    Xd,Xt,yd,yt=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    Xtr,Xv,ytr,yv=train_test_split(Xd,yd,test_size=.25,stratify=yd,random_state=42)
    pre=ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),
         ['tenure_days','monthly_gmv','order_count']),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                         ('ohe',OneHotEncoder(handle_unknown='ignore'))]),['merchant_tier'])])
    specs={'logistic':(LogisticRegression(max_iter=1000,random_state=42),{'model__C':[.1,1,10]}),
           'forest':(RandomForestClassifier(n_estimators=100,random_state=42),{'model__max_depth':[3,5,None]})}
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    fitted={}; scores={}
    for name,(est,params) in specs.items():
        grid=GridSearchCV(Pipeline([('pre',pre),('model',est)]),params,cv=cv,scoring='roc_auc',refit=True)
        grid.fit(Xtr,ytr)
        fitted[name]=grid
        scores[name]=roc_auc_score(yv,grid.predict_proba(Xv)[:,1])
    name=max(scores,key=scores.get)
    best=fitted[name]
    pv=best.predict_proba(Xv)[:,1]
    threshold=None; best_recall=-1
    for t in [.3,.4,.5,.6,.7]:
        pred=(pv>=t).astype(int)
        r=recall_score(yv,pred,zero_division=0)
        if precision_score(yv,pred,zero_division=0)>=.6 and r>=best_recall:
            threshold=t; best_recall=r
    pt=best.predict_proba(Xt)[:,1]
    metrics={'roc_auc':roc_auc_score(yt,pt)}
    if threshold is not None:
        pred=(pt>=threshold).astype(int)
        metrics.update(precision=precision_score(yt,pred,zero_division=0),
                       recall=recall_score(yt,pred,zero_division=0),f1=f1_score(yt,pred,zero_division=0))
    return {'best_model':name,'best_params':best.best_params_,'validation_auc':scores,
            'best_threshold':threshold,'test_metrics':metrics}
```

No feasible threshold means the operating constraint was not met on validation; do not silently lower the requirement or choose on test. Report AUC and threshold status separately. The dataset is small and noisy; the exact score is not a target or a guarantee of generalization.
</details>

[下一章：Cheat Sheet & Common Pitfalls →](07-review.md)
