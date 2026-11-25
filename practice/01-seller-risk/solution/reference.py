from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

DATA = Path(__file__).resolve().parents[1] / 'data'
COLS = ['seller_id','category','tier','tenure_days','completed_orders','total_gmv',
        'unique_buyers','jan_gmv','feb_gmv','gmv_growth','ticket_count','open_tickets','high_risk']

def read(path):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    return df

def build_features(data_dir=DATA):
    d = Path(data_dir)
    sellers = read(d/'sellers.csv')
    labels = read(d/'labels.csv')
    labels['high_risk'] = labels.high_risk.astype(int)
    orders = pd.concat([read(p) for p in sorted(d.glob('orders_*.csv'))], ignore_index=True).drop_duplicates('order_id')
    tickets = pd.concat([read(p) for p in sorted(d.glob('tickets_*.csv'))], ignore_index=True).drop_duplicates('ticket_id')
    orders['order_date'] = pd.to_datetime(orders.order_date, errors='coerce')
    orders['amount'] = pd.to_numeric(orders.amount, errors='coerce').replace([np.inf,-np.inf],np.nan)
    tickets['created_at'] = pd.to_datetime(tickets.created_at, errors='coerce')
    start, end = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-03-01')
    orders = orders.loc[orders.order_date.between(start,end,inclusive='left') & orders.status.eq('completed') & orders.amount.notna() & orders.seller_id.isin(sellers.seller_id)].copy()
    tickets = tickets.loc[tickets.created_at.between(start,end,inclusive='left') & tickets.seller_id.isin(sellers.seller_id)].copy()
    orders['jan'] = orders.amount.where(orders.order_date.dt.month.eq(1),0)
    orders['feb'] = orders.amount.where(orders.order_date.dt.month.eq(2),0)
    agg = orders.groupby('seller_id',as_index=False).agg(completed_orders=('order_id','size'),total_gmv=('amount','sum'),unique_buyers=('buyer_id','nunique'),jan_gmv=('jan','sum'),feb_gmv=('feb','sum'))
    tickets['is_open'] = tickets.status.eq('open').astype(int)
    ta = tickets.groupby('seller_id',as_index=False).agg(ticket_count=('ticket_id','size'),open_tickets=('is_open','sum'))
    out = sellers.merge(agg,on='seller_id',how='left',validate='one_to_one').merge(ta,on='seller_id',how='left',validate='one_to_one').merge(labels,on='seller_id',how='left',validate='one_to_one')
    zero = ['completed_orders','total_gmv','unique_buyers','jan_gmv','feb_gmv','ticket_count','open_tickets']
    out[zero] = out[zero].fillna(0)
    out['tenure_days'] = (end-pd.to_datetime(out.join_date,errors='coerce')).dt.days
    out['gmv_growth'] = (out.feb_gmv-out.jan_gmv)/out.jan_gmv.replace(0,np.nan)
    out = out[COLS].sort_values('seller_id').reset_index(drop=True)
    assert out.shape == (240,13) and out.seller_id.is_unique
    assert out.completed_orders.sum() == 1518
    assert out.ticket_count.sum() == 429
    assert out.total_gmv.sum() == 397459
    assert out.high_risk.sum() == 82
    return out

def train_model(features):
    X, y = features.drop(columns=['seller_id','high_risk']), features.high_risk
    Xtr,Xt,ytr,yt = train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    cat = ['category','tier']
    num = X.columns.difference(cat).tolist()
    pre = ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('enc',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    model = Pipeline([('pre',pre),('model',LogisticRegression(max_iter=1000,random_state=42))]).fit(Xtr,ytr)
    return model,Xt,yt

def evaluate(model,X_test,y_test):
    p = model.predict_proba(X_test)[:,list(model.classes_).index(1)]
    pred = (p>=.5).astype(int)
    return dict(accuracy=accuracy_score(y_test,pred),precision=precision_score(y_test,pred,zero_division=0),recall=recall_score(y_test,pred,zero_division=0),f1=f1_score(y_test,pred,zero_division=0),roc_auc=roc_auc_score(y_test,p),log_loss=log_loss(y_test,p,labels=[0,1]))

if __name__ == '__main__':
    features = build_features()
    model,X_test,y_test = train_model(features)
    print(features.shape, len(X_test), evaluate(model,X_test,y_test))
