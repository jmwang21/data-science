# 04.1 — Learning Paradigms & Linear Models

[← Chapter index](../04-ml.md) · [Next →](02-trees-ensembles.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## Learning paradigms 

| Method | Label | Example |
|---|---|---|
| Supervised | 有 target | churn, fraud, GMV regression |
| Unsupervised | 無 target | merchant segmentation, anomaly exploration |
| Semi-supervised | 少量有標籤、大量無標籤 | 商品分類 |

Clustering 不會自動找出「高潛商家」；需以商業指標驗證 cluster profiling。若只有採用廣告的商家有 label，可能有 selection bias，模型未必能泛化到全部商家。

## Linear Regression 

模型：$\hat y=\beta_0+\sum_j\beta_jx_j$；OLS 最小化 SSE。係數代表其他特徵固定時，x 增加一單位的預測差異，並不必然代表因果。

- 線性指條件平均對參數線性，不要求所有原始 x 與 y 都呈直線（可做多項式特徵）。
- 誤差獨立、零條件均值等是推論假設；同方差影響傳統標準誤，常態誤差主要關係有限樣本推論。
- Multicollinearity 使係數不穩定；高 R² 不代表因果或泛化好。

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
lr = LinearRegression().fit(X_train,y_train)
pred = lr.predict(X_test)
```

## Logistic Regression 

$z=\beta_0+\beta^Tx$；$p=1/(1+e^{-z})$；$\log(p/(1-p))=z$。

係數增加 1 單位的 odds ratio 為 $e^{\beta_j}$（其餘固定）。例如 β=.693，odds 約乘 2，不代表 probability 加倍。

Binary cross-entropy：$-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]$。

| Regularization | Penalty | 效果 |
|---|---|---|
| L1 | λΣ abs(β) | 可產生零係數，稀疏選特徵 |
| L2 | λΣ β² | 平滑收縮，通常不歸零 |

sklearn LogisticRegression 的 `C` 為 regularization strength 的反向控制：C 小 → stronger regularization。Ridge/Lasso 的 alpha 則大 → stronger。不同模型 penalty 的精確 scaling 定義可能不同。

## Loss, coefficients and regularization

Linear regression predicts continuous target; logistic regression predicts probability via sigmoid. OLS minimizes squared residuals, logistic regression minimizes negative log likelihood (cross-entropy). Linear models can use nonlinear transformed features.

For ridge, objective is SSE + alpha·sum(w²); Lasso uses absolute coefficient penalty (sklearn conventions may scale data loss differently). L1 can set coefficients to zero; L2 shrinks. ElasticNet combines both. LogisticRegression C is inverse regularization strength.

Example: logistic score z=0 means p=.5; z=ln(3) means odds=3 and p=.75. Adding coefficient ln(2) doubles odds to 6, so p=6/7=.857, not 1.5.

### Worked regression variation

A model predicts price (in thousands): price=50+3×area−2×age. For area=80 and age=10, prediction=270. If area grows by 10 with age fixed, predicted price increases by 30; this is conditional association, not proof of causation. Residual=y−prediction. If actual=300, residual=30, squared error=900.

Ridge shrinks coefficients; Lasso may choose zero coefficients. When two features are nearly identical, Lasso may arbitrarily choose one, Ridge tends to distribute weight. Neither coefficient magnitude alone nor feature importance establishes causal effects.

Gradient descent updates weights opposite the gradient. Learning rate too high can overshoot; too small can converge slowly. Batch vs mini-batch vs stochastic use different sample subsets.

<details><summary>Variation: In a linear regression, highly correlated predictors yield good predictive accuracy but unstable coefficients. Why?</summary>

Multicollinearity. Many coefficient combinations may produce similar predictions; Ridge can stabilize estimation. High R² does not establish causal effects.

</details>

<details><summary>Variation: A logistic model has coefficient 0.693 for a binary feature. Does its predicted probability double when the feature changes from 0 to 1?</summary>

No. Odds approximately double, conditional on other predictors; probability change depends on baseline odds.

</details>

[← Chapter index](../04-ml.md) · [Next →](02-trees-ensembles.md)
