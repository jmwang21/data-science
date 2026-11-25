# 04.2 — Linear Regression

[← ML index](../04-ml.md) · [Next →](logistic-regression.md)

## 1. Model and objective

**English:** Ordinary least squares estimates coefficients by minimizing the sum of squared residuals.

**Prediction:** ŷ = β₀ + β₁x₁ + … + βₚxₚ

**Residual:** e = y − ŷ

**Objective:** minimize SSE = Σ(yᵢ − ŷᵢ)²

係數代表其他特徵固定時，某個 x 增加一單位的預測差異。Intercept 為所有 x=0 時的預測值；若零不在資料範圍內，可能沒有直接商業意義。係數不是因果效應。

### Worked example

| x | y |
|---:|---:|
| 1 | 2 |
| 2 | 4 |
| 3 | 5 |

Slope = Σ[(xᵢ−x̄)(yᵢ−ȳ)] / Σ(xᵢ−x̄)² = 1.5。Intercept = ȳ−1.5x̄ = 2/3。

| x | Predicted | Residual |
|---:|---:|---:|
| 1 | 2.167 | −.167 |
| 2 | 3.667 | .333 |
| 3 | 5.167 | −.167 |

SSE = 1/6。OLS with intercept 的 training residuals 總和為 0，但 test 不一定。

## 2. Assumptions — prediction vs inference

| Assumption | Why it matters |
|---|---|
| Linearity in parameters | x²、interaction 可作為 feature，仍屬 linear model |
| Zero conditional mean E[ε∣X]=0 | 無偏估計的重要條件 |
| No perfect multicollinearity | 否則係數不唯一 |
| Independent errors | 群組或時間相依需適當推論 |
| Homoskedasticity | 影響傳統 standard error；異質變異本身不必然使係數有偏 |
| Normal errors | 小樣本傳統 t/F 推論；不是 OLS fitting 的必要條件 |

Residual plot 的 U shape 可能缺少非線性結構；fan shape 可能是 heteroskedasticity。高 leverage 且 residual 大的點可能有高 influence。Multicollinearity 會讓 individual coefficients 不穩定，但 prediction 可能仍佳。VIF = 1/(1−R²ⱼ)，其中 R²ⱼ 是用其他 predictors 解釋第 j 個 predictor 的 R²。

## 3. Ridge, Lasso, ElasticNet and polynomial features

| Method | Penalty | Effect |
|---|---|---|
| OLS | none | interpretable baseline |
| Ridge | L2: αΣw² | shrink coefficients; useful for correlated predictors |
| Lasso | L1: αΣabs(w) | can set coefficients to zero |
| ElasticNet | L1 + L2 | sparsity with correlated predictors |

alpha 越大通常 regularization 越強。Penalty 的精確 scaling 因 estimator 而異。標準化有助於不同單位 feature 接受可比較的 penalty。

PolynomialFeatures 建立 x²、x₁x₂ 等欄位，能降低 nonlinear pattern 的 bias，但過高 degree 容易 overfit。

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error
X,y=make_regression(n_samples=200,n_features=4,noise=10,random_state=42)
Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.25,random_state=42)
models={
    'ols':LinearRegression(),
    'ridge':make_pipeline(StandardScaler(),Ridge(alpha=1)),
    'lasso':make_pipeline(StandardScaler(),Lasso(alpha=.1,max_iter=5000)),
    'poly':make_pipeline(PolynomialFeatures(2,include_bias=False),
                         StandardScaler(),Ridge(alpha=1))
}
for name,m in models.items():
    m.fit(Xtr,ytr)
    print(name,mean_absolute_error(yv,m.predict(Xv)))
```

Train loss 可為 squared error，但若指定 MAE，使用 validation MAE 選模型。完整房價流程見 [Case](../cases/03-house-price.md)。

## Concept variations

<details><summary>1. Highly correlated predictors give good predictions but unstable coefficients. Why?</summary>

Multicollinearity. Multiple coefficient combinations produce similar predictions. Ridge may stabilize them.

</details>

<details><summary>2. Does heteroskedasticity necessarily bias OLS coefficients?</summary>

No. 若 zero conditional mean 等條件成立，係數可無偏，但傳統 SE 可能不可靠。

</details>

<details><summary>3. Adding quadratic terms increases training R² but worsens validation MAE. What should you do?</summary>

以指定 validation metric 選擇；可能是額外 capacity 導致 variance。比較正則化或較簡單模型。

</details>
