# 06.2 — Merchant Churn Prediction

[← Cases index](../06-cases.md)

> Worked case · 閱讀解題思路及完整解法，觀念變化可直接展開。所有資料為合成。

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

[← Cases index](../06-cases.md)
