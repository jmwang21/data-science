# 03.4 — Hyperparameter Tuning & Model Diagnosis

[← Chapter index](../03-models.md) · [Previous](03-regression.md) · [Next →](05-clustering-calibration.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## Parameters vs hyperparameters

Parameters 由 fit 學習（linear coefficients、tree split）；hyperparameters 由使用者設定（alpha、depth、learning rate）。調參目標不是 training score，而是未見資料表現。

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
X,y=make_regression(n_samples=150,n_features=5,noise=15,random_state=42)
Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.2,random_state=42)
pipe=make_pipeline(StandardScaler(),Ridge())
grid=GridSearchCV(pipe,{'ridge__alpha':[.01,.1,1,10]},
                  cv=KFold(5,shuffle=True,random_state=42),
                  scoring='neg_mean_absolute_error')
grid.fit(Xtr,ytr)
print(grid.best_params_)
print(-grid.best_score_)
print(mean_absolute_error(yv,grid.predict(Xv)))
```

negative MAE 是 sklearn higher-is-better 約定。GridSearchCV 枚舉；RandomizedSearchCV 抽取候選，適合較廣空間。Early stopping 以 development validation 監控迭代，不用 final test。調 learning_rate 時也應考慮迭代數。

| Train / validation error | Plausible cause | Response |
|---|---|---|
| both high | high bias, noisy data | better features, more capacity, inspect labels |
| train low, validation high | variance or shift | constrain complexity, check split, more representative data |
| both low | good fit | compare baseline, check leakage |

<details><summary>Variation: Increasing the number of random forest trees and increasing their depth have the same effect on overfitting. True or false?</summary>

False. Number of trees mainly stabilizes ensemble averaging; depth changes individual tree capacity.

</details>

<details><summary>Variation: You tune 100 parameter sets and report the highest validation score as an unbiased estimate. What's wrong?</summary>

Selection optimism. 用獨立 test 或 nested CV 評估選模流程；validation 分數本身仍可用來選模型。

</details>

[← Chapter index](../03-models.md) · [Next →](05-clustering-calibration.md)
