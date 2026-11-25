# 04.4 — Loss Functions & Optimization

[← ML index](../04-ml.md) · [Next →](decision-tree.md)

## 1. Loss vs objective vs metric

Loss 衡量 prediction error，objective 通常是平均 loss 加 penalty；metric 用來評估未見資料。可用 squared-error training，但以 validation MAE 選模型。

| Loss | Expression | Common use |
|---|---|---|
| Squared error | (y−ŷ)² | OLS, regression |
| Absolute error | abs(y−ŷ) | MAE, robust regression |
| Huber | quadratic near zero, linear outside | robust regression |
| Binary cross-entropy | −y log(p) − (1−y) log(1−p) | binary classification |
| Categorical cross-entropy | −log(p_true) | multiclass softmax |
| Hinge | max(0,1−y×score), y in {−1,+1} | margin classification |

For [10,20,90], mean=40, median=20. Prediction 40 yields MAE=33.33 and MSE=1266.67; prediction 20 yields MAE=26.67 and MSE=1666.67. MAE 最佳常數是 median，MSE 是 mean。

## 2. Gradient descent

Update: w_next = w − η × gradient

J(w)=(w−3)², gradient=2(w−3), start w=0.

| Step | w, η=.1 | Loss |
|---|---:|---:|
| 0 | 0 | 9 |
| 1 | .6 | 5.76 |
| 2 | 1.08 | 3.6864 |
| 3 | 1.464 | 2.359296 |

η=1.1 時，w 從 0 到 6.6 再到 −1.32，loss 增加。Learning rate too high may overshoot/diverge; too low may converge slowly.

For linear model ŷ=wx, x=2,y=4,w=1, half squared loss gradient=(ŷ−y)x=−4。η=.1 時 w=1.4，prediction=2.8，loss 從 2 到 .72。

```python
w=0.
for step in range(8):
    w -= .1 * 2*(w-3)
    print(step+1,round(w,4),round((w-3)**2,4))
```

## 3. Optimizer and training vocabulary

| Method | Meaning |
|---|---|
| Batch GD | full training data per update |
| SGD | one observation per update |
| Mini-batch | subset per update |
| Momentum | accumulates past gradients |
| Adam | adaptive steps using gradient moments |

Epoch = one pass through data；batch size = rows per update；iteration = one parameter update。1030 rows, batch size 100, keep partial batch → 11 iterations per epoch. Feature scaling often helps convergence. Convex optimization may reach a global optimum under appropriate conditions; neural nets are generally nonconvex.

Regularization limits capacity through penalty or other constraints. Early stopping monitors development performance. Hyperparameter search is not the same as gradient descent.

## Concept variations

<details><summary>1. Loss oscillates and increases after raising learning rate. What should you investigate?</summary>

Step size too large, numerical problems or feature scales. Lower LR and inspect convergence.

</details>

<details><summary>2. Does the lowest training loss imply best validation loss?</summary>

No. Capacity can fit noise; compare validation and generalization.

</details>

<details><summary>3. Is a model trained with MSE invalid when the assessment uses MAE?</summary>

No. Choose candidates by validation MAE; optionally compare absolute/Huber loss when available.

</details>
