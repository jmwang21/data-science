from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score

DATA = Path(__file__).resolve().parents[1] / 'data'
FEATURES = ['category','tier','tenure_days','total_gmv','total_orders','unique_customers','jan_gmv','feb_gmv','gmv_growth','ticket_count','open_tickets']

def load_csv(path):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    return df

def load_parts(pattern, key):
    paths = sorted(DATA.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    df = pd.concat([load_csv(path) for path in paths], ignore_index=True)
    if df[key].isna().any():
        raise ValueError(f'Missing {key}')
    return df.drop_duplicates(subset=key, keep='first')

def build_features():
    m = load_csv(DATA/'merchants.csv')
    y = load_csv(DATA/'labels.csv')
    o = load_parts('orders_*.csv','order_id')
    t = load_parts('tickets_*.csv','ticket_id')
    o['order_date'] = pd.to_datetime(o['order_date'], errors='coerce')
    o['amount'] = pd.to_numeric(o['amount'], errors='coerce')
    o = o.loc[o.order_date.between('2025-01-01','2025-03-01',inclusive='left') & o.status.eq('completed') & np.isfinite(o.amount)]
    a = o.assign(jan=lambda x:x.amount.where(x.order_date.dt.month.eq(1),0),feb=lambda x:x.amount.where(x.order_date.dt.month.eq(2),0)).groupby('merchant_id',as_index=False).agg(total_gmv=('amount','sum'),total_orders=('order_id','size'),unique_customers=('customer_id','nunique'),jan_gmv=('jan','sum'),feb_gmv=('feb','sum'))
    t['ticket_date'] = pd.to_datetime(t.ticket_date,errors='coerce')
    t = t.loc[t.ticket_date.between('2025-01-01','2025-03-01',inclusive='left')]
    b = t.assign(is_open=lambda x:x.status.eq('open').astype(int)).groupby('merchant_id',as_index=False).agg(ticket_count=('ticket_id','size'),open_tickets=('is_open','sum'))
    result = m.merge(a,on='merchant_id',how='left',validate='one_to_one').merge(b,on='merchant_id',how='left',validate='one_to_one').merge(y,on='merchant_id',how='left',validate='one_to_one')
    fill=['total_gmv','total_orders','unique_customers','jan_gmv','feb_gmv','ticket_count','open_tickets']
    result[fill]=result[fill].fillna(0)
    result['tenure_days']=(pd.Timestamp('2025-03-01')-pd.to_datetime(result.join_date)).dt.days
    result['gmv_growth']=(result.feb_gmv-result.jan_gmv)/result.jan_gmv.replace(0,np.nan)
    result['churn']=pd.to_numeric(result.churn,errors='raise').astype(int)
    assert len(result)==len(m) and result.merchant_id.is_unique
    assert set(result.churn.unique()) <= {0,1}
    return result[['merchant_id']+FEATURES+['churn']].sort_values('merchant_id').reset_index(drop=True)

def train_classifier(df):
    X,y=df[FEATURES],df.churn
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    cat=['category','tier']; num=[c for c in FEATURES if c not in cat]
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    model=Pipeline([('pre',pre),('model',LogisticRegression(max_iter=1000,random_state=42))])
    model.fit(X_train,y_train)
    return model,X_test,y_test

def evaluate(model,X_test,y_test):
    prob=model.predict_proba(X_test)[:,list(model.classes_).index(1)]
    pred=(prob>=.5).astype(int)
    return dict(accuracy=accuracy_score(y_test,pred),precision=precision_score(y_test,pred,zero_division=0),recall=recall_score(y_test,pred,zero_division=0),f1=f1_score(y_test,pred,zero_division=0),roc_auc=roc_auc_score(y_test,prob),log_loss=log_loss(y_test,prob,labels=[0,1]))

if __name__=='__main__':
    df=build_features()
    model,X_test,y_test=train_classifier(df)
    print('Feature table:',df.shape)
    print(df.head().to_string(index=False))
    print('Test rows:',len(X_test))
    print(evaluate(model,X_test,y_test))
