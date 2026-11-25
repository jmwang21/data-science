# 04.3 — Logistic Regression

[← ML index](../04-ml.md) · [Next →](loss-optimization.md)

## 1. From score to probability

**English:** Logistic regression models the log-odds of a binary outcome as a linear function of predictors.

- Linear score: z = b + wᵀx
- Sigmoid: p = 1 / (1 + exp(−z))
- Odds: p / (1−p) = exp(z)
- Log-odds: log[p/(1−p)] = z

| z | odds | p |
|---:|---:|---:|
| −1.386 | .25 | .20 |
| 0 | 1 | .50 |
| .693 | 2 | .667 |
| 1.386 | 4 | .80 |

Coefficient w 表示 feature 增加一單位時，odds 乘以 exp(w)，其他變數不變。若原 p=.2、w=ln(2)，odds 從 .25 變 .5，所以 p=1/3，不是 .4。

Threshold=.5 的 decision boundary 是 z=0；threshold=.8 則 z=ln(4)≈1.386。改 threshold 不等於 retrain。加入 nonlinear features 後，在原始 feature space 可形成非線性邊界。

## 2. Maximum likelihood and cross-entropy

每筆 Bernoulli likelihood 為 pʸ(1−p)^(1−y)。獨立資料 likelihood 是乘積；最大化 log likelihood 等同最小化 negative log-likelihood。

> Binary cross-entropy = −mean[y log(p) + (1−y) log(1−p)]

當 y=1，p=.9 loss=.105；p=.1 loss=2.303。錯且自信的預測懲罰大。Linear logits + BCE 為 convex objective；非所有分類模型都是 convex。Logistic regression 不要求 predictors 常態或互相獨立。

## 3. Important parameters

| Parameter | Effect |
|---|---|
| C | inverse regularization strength; smaller = stronger |
| L1 / L2 | sparse coefficients / smooth shrinkage |
| solver | algorithm for optimization; compatible with penalty |
| class_weight | changes each class's loss contribution |
| max_iter | iteration limit, not model complexity |

Perfect separation 可能讓未正則化 MLE 係數無界；regularization 有助穩定。Convergence warning 先查 scaling、solver，再考慮 max_iter。class_weight 不保證 validation recall 一定提升，也可能影響 calibration。

## 4. Implementation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss,recall_score
X,y=make_classification(n_samples=300,n_features=6,weights=[.85,.15],
                        random_state=42)
Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
for weight in [None,'balanced']:
    m=make_pipeline(StandardScaler(),LogisticRegression(
        C=1,class_weight=weight,max_iter=1000))
    m.fit(Xtr,ytr)
    p=m.predict_proba(Xv)[:,1] # labels are 0/1
    print(weight,roc_auc_score(yv,p),log_loss(yv,p),recall_score(yv,p>=.5))
```

predict_proba 回傳機率；decision_function 回傳 binary logit；predict 回傳 label。StandardScaler 後的 coef_ 在標準化單位下解釋。

## Concept variations

<details><summary>1. Does a coefficient of ln(2) imply the predicted probability doubles?</summary>

No. Odds doubles, not probability. 機率變化取決於 baseline。

</details>

<details><summary>2. High AUC but low recall at .5: must the model be retrained?</summary>

No. 先在 validation 檢查 threshold、成本與限制；排序好不代表 operating point 好。

</details>

<details><summary>3. Which direction of C strengthens regularization?</summary>

Smaller C. Ridge/Lasso 的 alpha 方向相反。

</details>

<details><summary>4. Do one-hot features violate a normal predictor assumption?</summary>

No. Logistic regression does not require normally distributed predictors.

</details>
