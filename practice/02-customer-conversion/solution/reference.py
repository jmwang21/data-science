from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,log_loss,confusion_matrix

DATA = Path(__file__).resolve().parents[1] / 'data'
COLS = ['customer_id','channel','plan','tenure_days','session_count','active_days',
        'avg_duration','last_seen_days','purchase_count','total_spend','jan_spend',
        'feb_spend','refund_rate','converted']

def read(path):
    df = pd.read_csv(path,dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    return df

def build_features(data_dir=DATA):
    d=Path(data_dir)
    customers, labels=read(d/'customers.csv'),read(d/'labels.csv')
    labels['converted']=labels.converted.astype(int)
    s=pd.concat([read(p) for p in sorted(d.glob('sessions_*.csv'))],ignore_index=True).drop_duplicates('session_id')
    p=pd.concat([read(p) for p in sorted(d.glob('purchases_*.csv'))],ignore_index=True)
    start,end=pd.Timestamp('2025-01-01'),pd.Timestamp('2025-03-01')
    p['updated_at']=pd.to_datetime(p.updated_at,errors='coerce')
    p=p.loc[p.updated_at.lt(end)].sort_values('updated_at').drop_duplicates('purchase_id',keep='last').copy()
    s['session_date']=pd.to_datetime(s.session_date,errors='coerce')
    s['duration_sec']=pd.to_numeric(s.duration_sec,errors='coerce').replace([np.inf,-np.inf],np.nan)
    p['purchase_date']=pd.to_datetime(p.purchase_date,errors='coerce')
    p['amount']=pd.to_numeric(p.amount,errors='coerce').replace([np.inf,-np.inf],np.nan)
    s=s.loc[s.session_date.between(start,end,inclusive='left') & s.duration_sec.ge(0) & s.customer_id.isin(customers.customer_id)].copy()
    p=p.loc[p.purchase_date.between(start,end,inclusive='left') & p.amount.ge(0) & p.status.isin(['completed','refunded']) & p.customer_id.isin(customers.customer_id)].copy()
    s['day']=s.session_date.dt.normalize()
    sa=s.groupby('customer_id',as_index=False).agg(session_count=('session_id','size'),active_days=('day','nunique'),avg_duration=('duration_sec','mean'),last_seen=('day','max'))
    sa['last_seen_days']=(end-sa.last_seen).dt.days
    pa=p.assign(refund=p.status.eq('refunded').astype(int)).groupby('customer_id',as_index=False).agg(refund_rate=('refund','mean'))
    done=p.loc[p.status.eq('completed')].copy()
    done['jan']=done.amount.where(done.purchase_date.dt.month.eq(1),0)
    done['feb']=done.amount.where(done.purchase_date.dt.month.eq(2),0)
    ca=done.groupby('customer_id',as_index=False).agg(purchase_count=('purchase_id','size'),total_spend=('amount','sum'),jan_spend=('jan','sum'),feb_spend=('feb','sum'))
    out=customers.merge(sa,on='customer_id',how='left',validate='one_to_one').merge(pa,on='customer_id',how='left',validate='one_to_one').merge(ca,on='customer_id',how='left',validate='one_to_one').merge(labels,on='customer_id',how='left',validate='one_to_one')
    zero=['session_count','active_days','purchase_count','total_spend','jan_spend','feb_spend']
    out[zero]=out[zero].fillna(0)
    out['tenure_days']=(end-pd.to_datetime(out.signup_date,errors='coerce')).dt.days
    out=out[COLS].sort_values('customer_id').reset_index(drop=True)
    assert out.shape==(300,14) and out.customer_id.is_unique
    assert out.session_count.sum()==1655
    assert len(p)==684
    assert out.purchase_count.sum()==574
    assert out.total_spend.sum()==77431
    assert out.converted.sum()==116
    return out

def train_model(features):
    X,y=features.drop(columns=['customer_id','converted']),features.converted
    Xtr,Xrest,ytr,yrest=train_test_split(X,y,test_size=.30,stratify=y,random_state=42)
    Xv,Xt,yv,yt=train_test_split(Xrest,yrest,test_size=.50,stratify=yrest,random_state=42)
    cat=['channel','plan']
    num=X.columns.difference(cat).tolist()
    pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    models={'logistic':LogisticRegression(max_iter=1000,random_state=42),'forest':RandomForestClassifier(n_estimators=100,min_samples_leaf=4,random_state=42)}
    fitted,scores={},{}
    for name,m in models.items():
        pipe=Pipeline([('pre',clone(pre)),('model',m)]).fit(Xtr,ytr)
        fitted[name]=pipe
        scores[name]=roc_auc_score(yv,pipe.predict_proba(Xv)[:,list(pipe.classes_).index(1)])
    best=max(scores,key=scores.get)
    return fitted[best],best,scores,Xt,yt

def evaluate(model,X_test,y_test):
    p=model.predict_proba(X_test)[:,list(model.classes_).index(1)]
    pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y_test,pred),precision=precision_score(y_test,pred,zero_division=0),recall=recall_score(y_test,pred,zero_division=0),f1=f1_score(y_test,pred,zero_division=0),specificity=tn/(tn+fp),roc_auc=roc_auc_score(y_test,p),average_precision=average_precision_score(y_test,p),log_loss=log_loss(y_test,p,labels=[0,1]))

if __name__=='__main__':
    features=build_features()
    model,name,scores,Xt,yt=train_model(features)
    p=model.predict_proba(Xt)[:,list(model.classes_).index(1)]
    predictions=pd.DataFrame({'customer_id':features.loc[Xt.index,'customer_id'].to_numpy(),'probability':p,'prediction':(p>=.5).astype(int)})
    assert len(predictions)==45 and predictions.customer_id.is_unique and np.isfinite(p).all()
    print(name,scores,evaluate(model,Xt,yt))
    print(predictions.head())
