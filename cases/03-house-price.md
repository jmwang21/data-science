# 06.3 — House Price Regression

[← Cases index](../06-cases.md)

> Worked case · 閱讀解題思路及完整解法，觀念變化可直接展開。所有資料為合成。

這是原創合成案例，參考已確認的 free-form regression 任務形式，不複製正式測驗內容或資料。

**Assessment-style requirements:** Use the supplied training (70%), validation (15%) and unlabeled test (15%) datasets. Compare regression models using validation mean absolute error and return predictions for unseen test rows.

| File | Labels | Role |
|---|---|---|
| train.csv | price | Fit models and optional inner CV |
| val.csv | price | Compare candidate models and tune decisions |
| test.csv | no price | Predict after model selection |

此例自行假設 id、price 欄位名稱與輸出 schema；實際題目未完整顯示的提交格式不可推定。前題清理後的資料不代表完全無缺失，先驗證 schema。既有 split 不需要再次拆成 70/15/15。

### 1. 確定資料、baseline、評估與輸出契約
MAE 越低越好；DummyRegressor(strategy='median') 作為最低基準。比較 regularized linear model、RandomForest、GradientBoosting。不同模型使用同樣資料與 metric。先檢查 schema、target 缺失、ID、數值及類別欄位。

### 2. 完整可閱讀程式

```python
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

def house_price_workflow(train,val,test):
    features=train.columns.drop(['id','price']).tolist()
    X,y=train[features],train.price
    Xv,yv=val[features],val.price
    Xt=test[features]
    num=X.select_dtypes(include='number').columns.tolist()
    cat=[c for c in features if c not in num]
    # Dense OHE is suitable for moderate cardinality in this example.
    pre=ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median')),
                         ('scale',StandardScaler())]),num),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                         ('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)
    ])
    specs={
        'baseline':(DummyRegressor(strategy='median'),{}),
        'ridge':(Ridge(),{'model__alpha':[.1,1,10]}),
        'rf':(RandomForestRegressor(n_estimators=100,random_state=42),
              {'model__min_samples_leaf':[1,5],'model__max_depth':[8,None]}),
        'gb':(GradientBoostingRegressor(loss='absolute_error',random_state=42),
              {'model__n_estimators':[100,200],'model__learning_rate':[.05,.1]})
    }
    cv=KFold(5,shuffle=True,random_state=42)
    fitted,scores={},{}
    for name,(est,params) in specs.items():
        search=GridSearchCV(Pipeline([('pre',clone(pre)),('model',est)]),
                           params,cv=cv,scoring='neg_mean_absolute_error',n_jobs=-1)
        search.fit(X,y)
        fitted[name]=search
        scores[name]=mean_absolute_error(yv,search.predict(Xv))
    best=min(scores,key=scores.get)
    # Refit only after selection; validation no longer independent afterwards.
    final=clone(fitted[best].best_estimator_).fit(
        pd.concat([X,Xv],ignore_index=True),
        pd.concat([y,yv],ignore_index=True))
    submission=pd.DataFrame({'id':test.id,'price':final.predict(Xt)})
    assert len(submission)==len(test)
    assert submission.price.notna().all()
    return submission,scores,best,fitted[best].best_params_

# submission,scores,best,params=house_price_workflow(
#     pd.read_csv('train.csv'),pd.read_csv('val.csv'),pd.read_csv('test.csv'))
# submission.to_csv('submission.csv',index=False)
```

此範例假設每份資料已具有合理 schema、數值欄位型別一致、target 非缺失、至少五筆 training、類別為字串。高 cardinality 需避免大型 dense OHE；sklearn 舊版 OneHotEncoder 使用 sparse=False。若資料有 group/time dependency，替換 KFold 為對應切分。樹不需要 scaling；為簡潔使用共同 pipeline。實際輸出欄位與格式必須依題意修改。

### 3. 如何改善 MAE
先分析 residual=y-prediction：正值為低估。依 price band、地區及房屋大小檢查 MAE；觀察訓練與 validation 差距。可能改善：房齡、面積與區域交互特徵；調整 max_depth/min_samples_leaf；比較 loss；對右偏 target 做 log1p 並在 expm1 後的原始尺度驗證。不得因 validation 表現反覆嘗試太多組合而把 validation 當成獨立 test。

<details><summary>Concept check：為何 CV 的最佳 negative MAE 要取最大，但 validation MAE 要取最小？</summary>

scikit-learn scorer 統一使用 higher-is-better，所以 negative MAE 是 MAE 乘以 −1；直接使用 mean_absolute_error 則越低越好。若 test 沒有 labels，不能自行計算 test MAE，只能產生 predictions。

</details>

[← Cases index](../06-cases.md)
