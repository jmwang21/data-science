# 03.3 — Regression Models, MAE & Error Analysis

[← Chapter index](../03-models.md) · [Previous](02-classification-metrics.md) · [Next →](04-tuning.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## Regression metrics 

| Metric | Formula / meaning | 注意事項 |
|---|---|---|
| MAE | mean(abs(y-pred)) | 原單位，較不受大誤差支配 |
| MSE | mean((y-pred)^2) | 大誤差懲罰較大 |
| RMSE | sqrt(MSE) | 原單位，受極端誤差影響 |
| R² | 1-SSE/SST | 可為負，常數 target 需特別處理 |
| MAPE | mean(abs((y-pred)/y)) | y=0 或接近 0 時不適合 |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred)**0.5
r2 = r2_score(y_test, pred)
```

## How the loss changes the answer

| Actual | Model A | Model B |
|---:|---:|---:|
| 100 | 110 | 125 |
| 200 | 210 | 225 |
| 300 | 250 | 275 |

A: MAE=70/3=23.33, RMSE=30；B: MAE=25, RMSE=25。不同 objective 可得到不同選模結果。

常數 baseline：MAE 最佳常數是 median，MSE 最佳常數是 mean。例如 [100,110,120,1000]，mean=332.5，median=115。最小化 MAE 的常數可在 [110,120] 內。模型訓練 loss 可以與評分 metric 不同，但以指定 validation metric 選模型。

| Model | What to tune first |
|---|---|
| Ridge / Lasso | alpha |
| RandomForestRegressor | depth, leaf size, max_features |
| GradientBoostingRegressor | learning_rate, n_estimators, depth |
| HistGradientBoostingRegressor | max_iter, leaf nodes, learning_rate |

詳見 [原理](../04-ml.md) 與 [完整房價案例](../cases/03-house-price.md)。

### Error analysis

Residual=y−prediction。正值代表低估。全體 MAE 可能掩蓋某個地區或房價區間的錯誤。檢查 residual vs predicted value、分群 MAE、bias mean residual，並比較 baseline。對右偏 target 可嘗試 log1p/expm1，但一定在原始價格尺度評分；反轉換不保證無偏。

<details><summary>Variation: The model has R² = −0.2 on validation. Is this impossible?</summary>

No. R²=1−SSE/SST；比使用 validation mean 的常數基準還差時可為負。R² 不是一般意義的 accuracy。

</details>

<details><summary>Variation: Test labels are hidden. How can you choose between models?</summary>

Use validation MAE (or CV on training); never pretend to calculate test MAE. Submit predictions following required schema.

</details>

[← Chapter index](../03-models.md) · [Next →](04-tuning.md)
