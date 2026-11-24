# 02 — Data Processing / Preprocessing

![Chapter 02](assets/02-preprocessing.svg)

[返回目錄](README.md)

> 閱讀導向：先掌握核心觀念、語法與完整範例；Concept checks 可直接展開答案，非必須獨立考試。P0 為核心，P1 為延伸。

## 1. Missing values & cleaning 

```python
import pandas as pd
import numpy as np
# df 為輸入資料
df.isna().sum()       # 各欄缺失數
df.isna().mean()      # 各欄缺失比例
df.dropna(subset=['target'])
df.drop_duplicates(subset=['order_id'])
pd.to_numeric(df['amount'],errors='coerce')
```

`isna().count()` 計算非缺失布林值數量，不是缺失數。缺失不等於零；沒有交易可定義 GMV=0，但未知金額不一定是零。去重前確認 business key，避免刪除合法重複交易。

| API | 用途 |
|---|---|
| `s.fillna(0)` | 填缺失 |
| `s.replace([-999,-1],np.nan)` | 替換指定值 |
| `s.replace({'A':1,'B':2})` | mapping，未匹配保留 |
| `s.map({'A':1,'B':2})` | mapping，未匹配通常 NaN |
| `s.where(s>=0,0)` | 條件成立保留 |

## 2. Infinity & outliers 

```python
df['growth'] = (df.feb_gmv-df.jan_gmv)/df.jan_gmv.replace(0,np.nan)
df = df.replace([np.inf,-np.inf],np.nan)
np.isinf(df['growth']).sum()
np.isfinite(df['growth'])
```

浮點向量除以零可得 inf/-inf，0/0 為 NaN；Python scalar 除以零則可能拋出 ZeroDivisionError。`pd.to_numeric(errors='coerce')` 不會消除 inf。`fillna` 也不會。成長率分母為零通常應保留未定義，而非假設 0%。

IQR = Q3-Q1，常用界線 Q1-1.5IQR、Q3+1.5IQR；不是必然應刪除的異常。Winsorization / clipping 可降低極端值影響，界線須從 training fit。

## 3. Comprehensive review — preprocessing decisions (P0) 

| Method | When useful | Important parameters / risks |
|---|---|---|
| SimpleImputer | Numeric or categorical missing values | strategy=median/mean/most_frequent/constant; add_indicator |
| KNNImputer (P1) | Similar observations carry information | n_neighbors; scale affects distance |
| RobustScaler | Features with extreme values | median and interquartile range |
| Normalizer | Text/vector similarity | norm=l2; scales each row, not each feature |
| PowerTransformer | Skewed continuous variables | Yeo-Johnson accepts zero/negative; Box-Cox strictly positive |
| OneHotEncoder | Nominal categories | handle_unknown, min_frequency |
| OrdinalEncoder | Ordered categories | explicit categories; unknown_value with handle_unknown |
| PolynomialFeatures | Linear model needs interactions | degree, include_bias=False; dimensionality grows rapidly |

```python
from sklearn.preprocessing import RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
# Numerical, imputed inputs; fit on training only.
pipe = make_pipeline(VarianceThreshold(), RobustScaler(),
                     SelectKBest(f_classif, k=3),
                     LogisticRegression(max_iter=1000))
# Requires at least 3 retained features.
pipe.fit(X_train,y_train)
```

### Missing mechanisms and outliers
MCAR: missingness unrelated to observed/unobserved values; MAR: explainable by observed variables; MNAR: depends on unobserved values even after conditioning. No universal imputation fixes MNAR. Investigate missingness by cohort. Mean preserves the observed mean but can reduce variance; median is robust; mode works for categories. Never fill unknown labels as negative without business justification.

Z-score outlier rules suit roughly symmetric data; IQR is less sensitive to extremes. Use training quantiles to clip validation/test: `lo,hi=X_train['gmv'].quantile([.01,.99]); X_valid['gmv'].clip(lo,hi)`. Log1p handles zero and positive skew, but does not make all data normal. Feature scaling is not a remedy for outliers by itself.

### Feature selection & imbalance
Filter methods: variance, correlation, mutual information, ANOVA F score. Wrapper: RFE (iterative model fitting). Embedded: L1 coefficients or tree importance. Fit inside folds. PCA creates new components, not a subset of original columns.

Class weights modify loss; random oversampling duplicates minority samples; undersampling removes majority examples; SMOTE interpolates neighbors (may be unsuitable for mixed categories and sparse data). Use an imbalanced-learn Pipeline when resampling in CV, and keep validation prevalence realistic. Threshold tuning is a separate decision. For high-cardinality categories, frequency encoding uses training counts; target encoding requires out-of-fold training features and a mapping learned from training labels for validation.

Reference: [scikit-learn transformations](https://scikit-learn.org/stable/data_transforms.html).

## 4. Scaling & encoding 

StandardScaler: $z=(x-\mu)/\sigma$；MinMaxScaler: $(x-min)/(max-min)$。LR、SVM、KNN、K-means 常受尺度影響；tree-based 模型通常不需要 scaling。OneHotEncoder 適用 nominal category；OrdinalEncoder 適用有序類別或明確編碼需求，不應任意假設順序。未知類別可用 handle_unknown='ignore'。

## 5. Feature Engineering 

```python
df['aov'] = (df.gmv/df.order_count.replace(0,np.nan)).fillna(0)
df['tenure_days'] = (pd.Timestamp('2025-03-01')-pd.to_datetime(df.join_date)).dt.days
df['active'] = (df.order_count>0).astype(int)
```

AOV=GMV/orders；repeat rate=回購客戶/全部客戶（先定義回購）；比例的總體值通常是分子總和/分母總和，不是各商家比例的簡單平均。所有特徵須符合預測時點，避免未來資訊。

## 6. ColumnTransformer & Pipeline 

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

## 7. Worked examples — scaling, encoding and missingness 

Training values [10,20,30] have mean 20 and population SD ≈8.165. StandardScaler produces approximately [-1.225,0,1.225]. A test value 40 becomes 2.449 using the same parameters, not a newly fit test scaler. MinMaxScaler fit on these values maps [10,20,30] to [0,.5,1]; 40 becomes 1.5 by default (not clipped).

| Tier | basic | pro |
|---|---:|---:|
| basic | 1 | 0 |
| pro | 0 | 1 |
| unseen enterprise | 0 | 0 |

The example assumes OHE fit on basic and pro, `handle_unknown='ignore'`, no dropped column. Unknown levels are not automatically missing values. High-cardinality merchant ID generally needs a justified encoding or exclusion; target encoding must be learned without leaking validation labels.

### Leakage-safe imputation

```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median')
train_num = imp.fit_transform(X_train[['monthly_gmv']])
valid_num = imp.transform(X_valid[['monthly_gmv']])
```

Never fit on combined train/validation. Median is robust to extreme observations but not universally optimal; missingness indicators can help if absence carries signal. Avoid imputing target labels for supervised evaluation.

### Feature selection & imbalance

Supervised selection (e.g. SelectKBest) must be inside the Pipeline. For imbalanced classification, `class_weight='balanced'` may change the loss emphasis but does not guarantee better recall or precision at a chosen threshold. Resampling must occur within training folds, never on validation/test. Evaluate on the distribution relevant to deployment.

### Data quality checklist

| Check | Question |
|---|---|
| Schema | Are numeric fields numeric and dates parsed? |
| Grain | Is a row an order, merchant, or merchant-month? |
| Completeness | Is missing value absence of data or a true zero? |
| Freshness | Was the feature available at prediction time? |
| Validity | Is negative GMV a refund, error, or legitimate value? |
| Duplication | Is the business key unique at the defined grain? |

## 8. Concept check 

Build a pipeline for numeric GMV with missing values and categorical tier with unseen categories. Use median imputation, scaling, most frequent imputation, one-hot encoding and logistic regression.

<details><summary>Answer</summary>

使用第 5 節的 Pipeline，將 num 設為 `['gmv']`、cat 設為 `['tier']`。只在 training fit，validation/test 僅 predict 或 transform。
</details>

## 9. Practical feature construction (P0)

Numeric/categorical column roles should be decided by meaning rather than dtype alone: postal codes and IDs are often categorical; dates usually need extraction; targets must be excluded. For text, CountVectorizer creates token counts; TfidfVectorizer downweights common terms. Fit vocabulary/IDF only on train. Sparse matrices are efficient for one-hot and text. StandardScaler(with_mean=False) avoids centering sparse matrices.

For historical behavioral features, use an explicit as-of time and aggregate only past events. For a label defined in the next 30 days, no future transaction or post-outcome status belongs in X. A lag feature must be shifted within entity after sorting. Check that transformations produce compatible columns and order for unseen data.

<details><summary>Concept check: should every model use the same scaling?</summary>

No. KNN/SVM and regularized linear models often benefit; tree splits are usually invariant to monotonic rescaling. Choose preprocessing for each estimator and fit it within CV.

</details>

[下一章：Model Development & Evaluation →](03-models.md)
