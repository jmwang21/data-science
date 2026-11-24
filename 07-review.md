# 07 — Cheat Sheet & Common Pitfalls

![Chapter 07](assets/07-review.svg)

[返回目錄](README.md)

> 這裡是速查，不是主要教學。需要原因、完整範例及變化題時，請回到各 Lesson。

## 1. pandas exam quick reference

| Task | Syntax |
|---|---|
| Filter | `df.loc[(df.a>0)&df.b.eq('x')]` |
| Date range | `s.between(start,end,inclusive='left')` |
| Missing count / ratio | `df.isna().sum()` / `.mean()` |
| Conditional amount | `df.amount.where(df.status.eq('completed'),0)` |
| Replace zero denominator | `den.replace(0,np.nan)` |
| Replace infinity | `df.replace([np.inf,-np.inf],np.nan)` |
| Aggregate | `df.groupby('id',as_index=False).agg(gmv=('amount','sum'))` |
| Count rows | `size` (count excludes nulls) |
| Distinct | `nunique` |
| Preserve rows | `groupby(...).transform('sum')` |
| Merge | `left.merge(right,on='id',how='left',validate='one_to_one')` |
| Rank per group | `df.groupby('category').gmv.rank(method='dense',ascending=False)` |
| Previous value | `df.groupby('id').gmv.shift(1)` after sorting |
| Past rolling | `g.transform(lambda s:s.shift(1).rolling(3).mean())` |
| Month | `pd.to_datetime(s).dt.to_period('M')` |
| Days | `(end-start).dt.days` |
| Output | `df[cols].sort_values(keys).reset_index(drop=True)` |

## 2. sklearn quick reference

| Task | Syntax |
|---|---|
| Split | `train_test_split(X,y,test_size=.2,stratify=y,random_state=42)` |
| Scaling | `StandardScaler()` |
| Encoding | `OneHotEncoder(handle_unknown='ignore')` |
| Pipeline | `Pipeline([('pre',pre),('model',model)])` |
| Fit | `model.fit(X_train,y_train)` |
| Probability | `model.predict_proba(X_valid)[:,1]` (verify classes) |
| Label | `(prob>=threshold).astype(int)` |
| ROC-AUC | `roc_auc_score(y_true,prob)` |
| AP | `average_precision_score(y_true,prob)` |
| Precision/Recall/F1 | `precision_score(y_true,pred,zero_division=0)` etc. |
| CV | `StratifiedKFold(5,shuffle=True,random_state=42)` |
| Grid | `GridSearchCV(pipe,{'model__C':[.1,1,10]},cv=cv,scoring='roc_auc',refit=True)` |
| Best | `grid.best_params_`, `grid.best_score_`, `grid.best_estimator_` |
| Dict selection | `max(scores,key=scores.get)` for AUC; `min(scores,key=scores.get)` for MAE |

## 3. Common pitfalls & concept corrections

| Topic | Incorrect pattern | Correct principle |
|---|---|---|
| Missing count | `df.isna().count()` | `.sum()` counts True |
| Tenure | date subtraction alone | `.dt.days`; correct reference date |
| Join | left join → remove unmatched rows | clean & aggregate orders → left join merchant universe |
| Conditional GMV | filter cancelled before computing all-order completion rate | use Boolean indicators and conditional amount; denominator must match requirement |
| Growth | `to_numeric(errors='coerce')` on inf | replace denominator 0 or replace ±inf |
| Category rank | global `rank()` | groupby category then rank |
| Output | `df.columns=cols` | `df[cols]` selects/reorders; assignment renames |
| Binary flag | bool output | `.astype(int)` when 0/1 specified |
| Dict | `scores['name']=score` | `scores[name]=score` |
| Model selection | select hyperparameters using test score | choose on validation/CV; optional final refit on development data after selection |
| Threshold | compare `y_valid` to threshold | compare `valid_prob` to threshold |
| Optimization | maximize precision only | filter precision constraint, maximize recall |
| Tie | extra nested conditions | ascending thresholds + recall >= best_recall |
| Grid | best_score_ without instance | `grid.best_score_` |
| Test metrics | validation predictions against test labels | compute test probabilities from X_test |
| Feature selection | fit selector on all X,y before CV | selector inside Pipeline |
| A/B | p-value = probability null true | conditional probability of extreme statistic under H0 |
| Power | probability this result is true | P(reject H0 | specified alternative) |

## 4. Exam workflow

```mermaid
flowchart LR
 A[Read requirements] --> B[Define grain and keys]
 B --> C[Clean / split]
 C --> D[Aggregate / train]
 D --> E[Boundary checks]
 E --> F[Validate output]
 classDef step fill:#dbeafe,stroke:#2563eb,color:#172554
 classDef check fill:#dcfce7,stroke:#16a34a,color:#14532d
 class A,B,C,D step
 class E,F check
```

- [ ] Check missing values, zero denominator, empty subsets, unseen categories.
- [ ] Check key uniqueness, join retention, row counts, expected output order.
- [ ] Check train/valid/test isolation and group/time leakage.
- [ ] Check probability vs label inputs and metric direction.
- [ ] Run code; verify exact output schema and small hand-calculated example.

自我審視：閱讀解法時核對資料粒度、邊界條件與輸出格式，不要求限時完成。

## 5. Reference documentation

- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [SciPy stats](https://docs.scipy.org/doc/scipy/reference/stats.html)

These are links for further reading, not claims that the live documentation was checked for this edition.

## 6. Model & statistics quick reference（P0）

| Topic | Key relationship / choice |
|---|---|
| Regression baseline | MAE → median; MSE → mean |
| Regression metrics | MAE 原單位；RMSE 放大大誤差；R² 可負；MAPE 不適合零值 |
| Classification | ROC-AUC ranking；AP 重視正類；Log Loss 機率品質；F1 threshold dependent |
| Regularization | Ridge/Lasso alpha 越大越強；LogisticRegression C 越小越強 |
| RF | n_estimators、max_features、max_depth、min_samples_leaf |
| Boosting | learning_rate × n_estimators；depth 控制每棵樹 |
| KNN/SVM/PCA | 尺度通常重要 |
| Clustering | K-means inertia 非遞增；silhouette 越高通常越好；DBSCAN 不需指定 K |
| Imputation | training fit，validation/test transform；NaN 不等於 0 |
| Feature selection | supervised selector 放進 CV pipeline |
| Bayes | posterior = likelihood × prior / evidence |
| Independence | P(A∩B)=P(A)P(B)；without replacement 通常不獨立 |
| Binomial | 固定 n、獨立、固定 p；mean=np，variance=np(1-p) |
| Hypergeometric | 有限母體不放回 |
| Poisson / Exponential | 區間事件數 / 等待時間 |
| CLT | sample mean 的抽樣分布近似常態，不是原始資料 |
| SE | SD/sqrt(n) under IID |
| CI | point estimate ± critical value × SE |
| p-value | H0 下至少同樣極端 statistic 的機率 |
| Power | 1−β，特定 alternative 下拒絕 H0 的機率 |
| Sample size | MDE 減半約需四倍；power↑、alpha↓、variance↑ 均提高需求 |
| Test choice | Welch：獨立均值；paired t：配對；proportion z：比例；chi-square：類別關聯 |

### 自我審視：能否解釋而非背誦？
能否說明資料粒度、join fanout、缺失與零、fit/transform、不同 split 的部署含義、模型參數如何影響 bias/variance、指標與商業成本的關係、p-value 與 power 的差別？若不確定，回到相應核心章節閱讀 worked example，而非重複大量題目。
