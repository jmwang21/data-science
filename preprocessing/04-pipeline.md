# 02.4 — ColumnTransformer, Pipeline & Leakage

[← Chapter index](../02-preprocessing.md) · [Previous](03-feature-engineering.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## ColumnTransformer & Pipeline 

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
num = ['tenure_days','monthly_gmv','order_count']
cat = ['merchant_tier']
pre = ColumnTransformer([
    ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
    ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cat)
])
model = Pipeline([('pre',pre),('model',LogisticRegression(max_iter=1000))])
# X_train, y_train, X_test 由切分取得
model.fit(X_train,y_train)
prob = model.predict_proba(X_test)[:,1]
```

`fit` 學參數；`transform` 使用既有參數；`fit_transform` 兩者合併。先 split，再 fit preprocessing；CV 時每個 fold 獨立 fit。Pipeline 可確保 GridSearchCV 不會先用全部 training 資料 fit scaler 再切 folds。

## Complete worked example

**English:** Fit numerical and categorical preprocessing on training data and use the same fitted transformation on validation.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
train = pd.DataFrame({'income':[10.,20.,30.,np.nan],
                      'tier':['basic','pro','basic','pro']})
val = pd.DataFrame({'income':[100.,np.nan],
                    'tier':['new','basic']})
pre = ColumnTransformer([
    ('num',Pipeline([('imp',SimpleImputer(strategy='median')),
                     ('scale',StandardScaler())]),['income']),
    ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['tier'])
])
print(pre.fit_transform(train))
print(pre.transform(val))
# [[11.3137, 0, 0], [0, 1, 0]] approximately
```

median=20；補值後 mean=20, SD=sqrt(50)。val=100 得 80/sqrt(50)=11.3137。unknown tier 變全零。ColumnTransformer 對不同 columns 套不同處理，再拼接。Pipeline 將 preprocessing 與 estimator 綁在一起；GridSearchCV 每 fold clone 並 fit。

### Variation: same preprocessing, different validation designs

<details><summary>1. Can a scaler be fitted before splitting if it never sees target labels?</summary>

No. 此時 scaler 看到 validation feature distribution。對 holdout/CV 要以對應 training subset fit。

</details>

<details><summary>2. Should SMOTE be placed before ordinary GridSearchCV?</summary>

No. 先 SMOTE 再 CV 可能讓相似或合成 samples 跨 folds；若使用 imbalanced-learn pipeline，將 resampling 放在每個 training fold。

</details>

<details><summary>3. What does model__C mean in GridSearchCV?</summary>

指定名為 model 的 Pipeline step 中的參數 C。若 step 名是 clf，則用 clf__C。參數可用 pipe.get_params() 查。

</details>

[← Chapter index](../02-preprocessing.md)
