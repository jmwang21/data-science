# 07 — Cheat Sheet & Common Pitfalls

![Chapter 07](assets/07-review.svg)

[返回目錄](README.md)

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
| Dict selection | `max(scores,key=scores.get)` |

## 3. Common pitfalls & concept corrections

| Topic | Incorrect pattern | Correct principle |
|---|---|---|
| Missing count | `df.isna().count()` | `.sum()` counts True |
| Tenure | date subtraction alone | `.dt.days`; correct reference date |
| Join | left join → remove unmatched rows | clean & aggregate orders → left join merchant universe |
| Conditional GMV | filter subset Series in assign | valid via index alignment on original data; `.where` clearer |
| Growth | `to_numeric(errors='coerce')` on inf | replace denominator 0 or replace ±inf |
| Category rank | global `rank()` | groupby category then rank |
| Output | `df.columns=cols` | `df[cols]` selects/reorders; assignment renames |
| Binary flag | bool output | `.astype(int)` when 0/1 specified |
| Dict | `scores['name']=score` | `scores[name]=score` |
| Model selection | fit already-trained model on validation; score test | use validation predict_proba, no refit in that task |
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

練習目標：限時完成並實際執行測試；核對資料粒度、邊界條件與輸出格式。

## 5. Reference documentation

- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [SciPy stats](https://docs.scipy.org/doc/scipy/reference/stats.html)

These are links for further reading, not claims that the live documentation was checked for this edition.
