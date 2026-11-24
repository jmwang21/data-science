# 03.2 — Classification Metrics & Threshold

[← Chapter index](../03-models.md) · [Previous](01-workflow-validation.md) · [Next →](03-regression.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## Confusion Matrix（固定方向） 

| Actual ↓ / Predicted → | Positive | Negative |
|---|---:|---:|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | 全體正確率 |
| Precision | TP/(TP+FP) | 預測正類中有多少為真 |
| Recall / TPR | TP/(TP+FN) | 真正正類中找出多少 |
| Specificity / TNR | TN/(TN+FP) | 真負類中排除多少 |
| FPR | FP/(FP+TN) | 真負類被誤報的比例 |
| F1 | 2PR/(P+R) | Precision/Recall 調和平均 |

例如 TP=80, FN=20, FP=40, TN=860：Accuracy=.94, Precision=.667, Recall=.8, FPR=.0444, F1=.7273。Fraud detection 通常需要關注 recall 與 review capacity；廣告高潛商家推薦可優先 precision 或 precision@K；不能僅因資料不平衡就認定某個 metric 永遠最佳。

```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
# sklearn 原始矩陣為 labels=[0,1]：[[TN,FP],[FN,TP]]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
# 若要符合本教材表格，直接使用 labels=[1,0]：[[TP,FN],[FP,TN]]
```

## ROC-AUC vs PR-AUC vs F1 

| Metric | Input | 意義 | 使用情境 |
|---|---|---|---|
| ROC-AUC | probabilities / scores | 隨機正類排名高於隨機負類的機率（ties 半分） | 整體排序能力 |
| Average Precision (AP) | probabilities / scores | 以 recall 增量加權的 precision | 正類稀少，重視正類排序 |
| F1 | binary labels | 特定 threshold 的 P/R 平衡 | 有明確決策 threshold |
| Precision@K | top K | 前 K 個候選中的正類比例 | 名單或人力上限 |

ROC 橫軸 FPR、縱軸 TPR；隨機模型 AUC 約 .5。PR 橫軸 recall、縱軸 precision；隨機排序 baseline 約正類比例。AP 不完全等於梯形積分 PR-AUC。ROC-AUC 高不代表營運 threshold 下 precision 足夠，也不代表 probability calibration 好。

```python
from sklearn.metrics import roc_auc_score, average_precision_score
auc = roc_auc_score(y_valid, valid_prob)
ap = average_precision_score(y_valid, valid_prob)
```

降低 threshold：TP 與 FP 不減、FN 與 TN 不增，因此 recall 不減、FPR 不減；precision 可能增加、下降或不變，並非數學上必然下降。

## Log Loss / Binary Cross-Entropy 

Log Loss 評估機率預測的品質，越低越好；對自信但錯誤的預測懲罰特別大。

$$L=-\frac{1}{n}\sum_{i=1}^n[y_i\ln(p_i)+(1-y_i)\ln(1-p_i)]$$

| True label | Predicted positive probability | Loss |
|---|---:|---:|
| 1 | 0.9 | 0.1054 |
| 1 | 0.5 | 0.6931 |
| 1 | 0.1 | 2.3026 |
| 0 | 0.9 | 2.3026 |

```python
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
y = np.array([0, 0, 1, 1])
a = np.array([.1, .2, .8, .9])
b = np.array([.41, .42, .58, .59])
for p in [a,b]:
    print(roc_auc_score(y,p), log_loss(y,p))
```

兩組 AUC 都是 1，Log Loss 約為 0.1643 和 0.5362。AUC 評估排序，不保證機率校準好；Log Loss 評估給真實類別的機率。Accuracy 則依賴分類閾值。`scoring='neg_log_loss'` 在 sklearn 搜尋中取較大值，實際 loss 為其負值。手動計算可用 `np.clip(p, 1e-15, 1-1e-15)` 防止 log(0)。

> [!WARNING]
> 不要將 threshold 後的 0/1 label 當成機率傳入 Log Loss。AUC 也應使用連續分數。

**Practice:** Two classifiers have identical ROC-AUC but different log loss. Must their calibration be identical?

<details><summary>Answer</summary>

No. A strictly increasing transformation preserves ranking and therefore AUC, but changes the probabilities and generally changes log loss. Log loss alone does not isolate calibration from discrimination.

</details>

## Threshold optimization 

Business requirement: Maximize recall subject to precision >= 0.60. If tied, choose the highest threshold.

```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
valid_prob = best_model.predict_proba(X_valid)[:,1]
best_threshold = None
best_recall = -1
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (valid_prob >= threshold).astype(int)
    precision = precision_score(y_valid, pred, zero_division=0)
    recall = recall_score(y_valid, pred, zero_division=0)
    if precision >= .60 and recall >= best_recall:
        best_recall = recall
        best_threshold = threshold
if best_threshold is None:
    raise ValueError('No feasible threshold')
test_prob = best_model.predict_proba(X_test)[:,1]
test_pred = (test_prob >= best_threshold).astype(int)
test_metrics = {
    'precision': precision_score(y_test,test_pred,zero_division=0),
    'recall': recall_score(y_test,test_pred,zero_division=0),
    'f1': f1_score(y_test,test_pred,zero_division=0),
    'roc_auc': roc_auc_score(y_test,test_prob)
}
```

`>= best_recall` 可簡化 tie-breaking，因 thresholds 已由小到大排列。若順序不固定，使用 tuple 比較 `(recall, threshold)`。不能對 `y_valid` 真實 labels 做 threshold；也不能在 test 選 threshold。模型若重新 fit，機率尺度可能改變，原 threshold 不必然仍有效；本例直接評估已選定的模型。

## Variation: prevalence, threshold and ranking

| Actual ↓ / Predicted → | Positive | Negative |
|---|---:|---:|
| Actual Positive | 90 | 10 |
| Actual Negative | 45 | 855 |

Precision=90/135=66.7%；Recall=.9；FPR=.05。若新市場 prevalence 從 10% 變 1%，且 TPR=.9、FPR=.05 不變，precision=.009/(.009+.0495)=15.38%。ROC 點不變，precision 改變。AP 與 prevalence 相關，不能把 ROC-AUC 與 AP 直接當同一指標。

### Worked threshold example

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score
y=np.array([1,0,1,1,0])
p=np.array([.9,.8,.6,.4,.1])
for t in [.5,.7,.85]:
    pred=(p>=t).astype(int)
    print(t,precision_score(y,pred),recall_score(y,pred))
# .5: 2/3, 2/3; .7: 1/2, 1/3; .85: 1, 1/3
```

Lowering threshold increases or preserves recall and FPR; precision 不保證單調。若要求 precision≥.6，再最大化 recall，.5 是此候選中的最佳解。若沒有可行 threshold，明確回報無可行解，而非偷偷改限制。

<details><summary>Variation: A strictly increasing transform is applied to every probability. Which metric is invariant: ROC-AUC or Log Loss?</summary>

ROC-AUC（排序與 ties 不變）。Log Loss 評估 probability magnitude，因此可能改變；同時 threshold=.5 的預測也可能改變。

</details>

<details><summary>Variation: For a rare disease screening system, should you always maximize F1?</summary>

No. 漏診、誤報的成本、可接受 false positive rate、醫療資源與校準都重要。可最大化 recall 並加 precision/FPR constraint。

</details>

[← Chapter index](../03-models.md) · [Next →](03-regression.md)
