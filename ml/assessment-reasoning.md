# ML Assessment Reasoning — 核心觀念與英文多選題

[← ML index](../04-ml.md) · [Model evaluation](../03-models.md)

這是一份跨模型觀念的閱讀單元，不是新的一套演算法教材，也不是正式試題。先讀每個情境的 reasoning，再展開英文多選題解析。以下六組是原創練習，選項可能有多個正確答案。

## 1. Generalization: diagnose the gap, not just name overfitting

| Training accuracy | Validation accuracy | Plausible interpretation |
|---:|---:|---|
| 98% | 65% | high variance, shift, invalid split, noisy validation |
| 66% | 64% | high bias, weak signal or noisy labels |
| 98% | 98% | good performance OR leakage / unrepresentative evaluation |

High variance 表示 fitted model 對訓練樣本變動敏感。Overfitting 是模型捕捉訓練特有的噪音而泛化不佳；gap 是證據之一，不是充分條件。比較相同 metric、可代表部署的資料、相同 inference 條件；檢查 split、label quality、leakage 後，再決定減少 capacity、加強 regularization 或增加代表性資料。

若 train accuracy 與 val accuracy 分別 98%、65%，不能單憑數字斷定 high variance；若 training 在含 dropout 的模式計分、validation 關閉 dropout，validation 甚至可能高於 train。

### Select all statements that are true

A. A large train-validation gap is consistent with overfitting.  
B. A large gap proves the training and validation distributions are identical.  
C. Increasing regularization may improve validation performance.  
D. More representative training data may help if variance is the main problem.  
E. Increasing training accuracy necessarily improves generalization.

<details><summary>Answer and option-by-option reasoning</summary>

**A, C, D.**

- A — True: overfitting is a plausible cause, not the only one.
- B — False: a distribution shift may itself explain the gap.
- C — True: stronger regularization may reduce variance; too much can underfit.
- D — True: representative data can stabilize estimates; it cannot guarantee a cure.
- E — False: fitting training noise can worsen performance on unseen samples.

</details>

[Full lesson: Bias & Variance](regularization-generalization.md)

## 2. Ensembles: distinguish how models are trained and combined

Bagging creates diverse independently trained models, mainly reduces variance through averaging. Random forest adds random candidate features at each split. Boosting sequentially improves the current ensemble objective; the next learner depends on previous predictions. Voting combines already-trained predictions; stacking learns a combination using out-of-fold base predictions.

Worked variation: each model has variance 1. Averaging 10 equally correlated models with correlation .2 yields variance .2+.8/10=.28; with correlation .9 yields .91. This explains why model diversity matters more than blindly adding trees.

Three classifiers predict probabilities .51, .51, .01 for class 1. Hard voting predicts 1; soft voting mean=.343 predicts 0 at threshold .5. Their combination rules differ.

### Select all statements that are true

A. Bagging can reduce variance without guaranteeing lower bias.  
B. In random forests, candidate feature randomness can reduce correlation among trees.  
C. Gradient boosting fits independent trees that can all be trained before calculating the current ensemble predictions.  
D. Boosting always fits raw residuals, regardless of the loss function.  
E. A stacking meta-model can be trained using out-of-fold base predictions.

<details><summary>Answer and option-by-option reasoning</summary>

**A, B, E.**

- A — True: averaging partially decorrelated learners reduces variance; bias need not decrease.
- B — True: random features can promote diversity, but predictive strength may also change.
- C — False: standard gradient boosting is sequential.
- D — False: raw residuals correspond to squared-error boosting; general differentiable loss uses negative gradients (up to loss scaling).
- E — True: OOF predictions help the meta-model learn from held-out base predictions rather than in-sample optimism.

</details>

[Full lesson: Ensemble Learning](ensembles.md)

## 3. ROC-AUC: compare rankings, not just accuracy

ROC curve plots TPR against FPR as threshold changes. AUC is the probability that a random positive is ranked above a random negative, giving half credit for ties.

| Actual class | Score |
|---|---:|
| Positive | .8 |
| Positive | .4 |
| Negative | .4 |
| Negative | .2 |

Four positive-negative pairs: .8 beats both negatives (2 points); .4 ties .4 (.5 points) and beats .2 (1 point). AUC=3.5/4=.875. At threshold .5 only one positive is predicted positive; recall=.5. AUC=.875 is not accuracy=.875 or recall=.875.

Strictly increasing score transformations preserve pairwise ordering and AUC on the same sample. They may change hard predictions at a fixed numerical threshold. AUC can use logits or other real-valued ranking scores; probabilities are not required. Prevalence alone does not change the population ROC when class-conditional score distributions stay fixed; sampling uncertainty and other shifts may affect observed AUC. PR/AP depends strongly on prevalence.

### Select all statements that are true

A. A strictly increasing transformation of scores preserves ROC-AUC on the same dataset.  
B. An AUC of .9 implies 90% accuracy at threshold .5.  
C. Ties receive half credit in the pairwise interpretation.  
D. ROC-AUC can be calculated from a suitable decision_function score.  
E. Higher AUC guarantees higher precision at every operating threshold.

<details><summary>Answer and option-by-option reasoning</summary>

**A, C, D.**

- A — True: pairwise ranks and ties remain unchanged.
- B — False: accuracy depends on threshold and class mix.
- C — True: equally scored positive-negative pairs contribute .5.
- D — True: ranking scores need not be probabilities.
- E — False: ROC curves can cross, thresholds may use different score scales, and precision depends on prevalence and the operating point.

</details>

[Full lesson: Classification Metrics](../modeling/02-classification-metrics.md)

## 4. Log Loss: confidence matters

Binary log loss is the average negative log probability assigned to the true label. Lower is better. For y=1, p=.9 gives loss .105; p=.01 gives 4.605. For y=0, p=.99 gives 4.605 too.

| Actual labels | Model A probabilities | Model B probabilities |
|---|---|---|
| [0,0,1,1] | [.1,.2,.8,.9] | [.41,.42,.58,.59] |

Both rank perfectly (AUC=1) and at .5 classify perfectly. A's mean log loss is about .1643, B's about .5362. Different confidence, same ranking and hard labels. But lower log loss alone does not prove perfect calibration; calibration asks whether among observations assigned probability p, about fraction p are positive.

Example: among 100 predictions near .8, 50 are positive → observed fraction=.5, indicating overconfidence in that bin (sampling noise matters).

### Select all statements that are true

A. Log loss penalizes a confidently wrong prediction more than an uncertain wrong prediction.  
B. Identical ROC-AUC implies identical log loss.  
C. Two models can have identical accuracy and different log loss.  
D. A model with the lowest log loss is necessarily perfectly calibrated.  
E. Binary log loss should normally be evaluated on positive-class probabilities, not thresholded hard labels.

<details><summary>Answer and option-by-option reasoning</summary>

**A, C, E.**

- A — True: negative log of a small true-class probability is large.
- B — False: ranking ignores confidence magnitude.
- C — True: probabilities can differ without changing decisions.
- D — False: log loss reflects overall probabilistic quality; calibration is a separate property.
- E — True: hard labels discard uncertainty and may yield very large loss when wrong.

</details>

## 5. Binary classification: threshold, prevalence and operating objective

Fixed scores, threshold decreases from .7 to .5: predicted positives can only increase (or stay unchanged). TP and FP cannot decrease; FN and TN cannot increase. Thus recall and FPR cannot decrease. Precision can rise, fall or stay unchanged depending on the newly admitted observations. F1 and accuracy are not generally monotonic.

| Actual ↓ / Predicted → | Positive | Negative |
|---|---:|---:|
| Actual Positive | 80 | 20 |
| Actual Negative | 90 | 810 |

Recall=.8, FPR=.1, precision=80/170≈.471. If prevalence decreases from .10 to .01 with same TPR/FPR, precision=.008/(.008+.099)≈.0748. The classifier need not have changed; the population mix did.

For a review capacity of 100 merchants, precision@100 may matter more than accuracy. For fraud missed cases, consider recall constrained by false positives, review capacity or cost. Class weighting changes training objective; threshold tuning changes decisions for a fitted model. Neither guarantees all metrics improve.

### Select all statements that are true

A. For fixed scores, lowering the threshold cannot decrease recall.  
B. Lowering the threshold always decreases precision.  
C. Class weighting and threshold tuning are identical operations.  
D. With 1% positives, predicting everything negative achieves 99% accuracy.  
E. The threshold should be chosen using development data rather than the final test set.

<details><summary>Answer and option-by-option reasoning</summary>

**A, D, E.**

- A — True when actual positives exist; newly admitted positives cannot reduce TP.
- B — False: added observations may have a higher positive fraction than previous predictions.
- C — False: class weighting changes fitted model; threshold changes the decision rule.
- D — True: but recall is zero, so this can be useless for detection.
- E — True: using test labels to choose threshold compromises independent evaluation.

</details>

## 6. Code interpretation: recognize a correct workflow

```python
# Assume X, y contain numeric features and binary labels.
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

Xtr, Xt, ytr, yt = train_test_split(X, y, test_size=.2,
                                    stratify=y, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
search = GridSearchCV(pipe, {'logisticregression__C':[.1,1,10]},
                      cv=StratifiedKFold(5,shuffle=True,random_state=42),
                      scoring='roc_auc', refit=True)
search.fit(Xtr, ytr)
p = search.predict_proba(Xt)[:,1]  # assumes classes [0, 1]
auc = roc_auc_score(yt,p)
loss = log_loss(yt,p)
```

The scaler fits only on each CV training fold. GridSearchCV uses mean validation AUC for selection and refits best pipeline on all Xtr. Test used only after selection. If test labels are absent, skip metrics and output predictions. If rows share entities or time dependence, replace IID split/CV with appropriate design.

### Select all statements that are true

A. StandardScaler must be fitted separately on every training fold.  
B. refit=True means the selected pipeline is fitted on the entire dataset passed to search.fit.  
C. best_score_ is an unbiased estimate of the final test performance.  
D. Replacing probabilities with predicted labels is generally undesirable for log loss and ranking AUC.  
E. StratifiedKFold guarantees no entity ID overlap.

<details><summary>Answer and option-by-option reasoning</summary>

**A, B, D.**

- A — True: prevents validation statistics leaking into preprocessing.
- B — True: here the entire Xtr, not Xt.
- C — False: it is the selected mean CV score and can be optimistic.
- D — True: hard labels lose probability and fine ranking information.
- E — False: stratification preserves label ratios; group isolation requires a group-aware split.

</details>

## English wording — how to read options

| Phrase | Interpretation |
|---|---|
| may / can | possible; not a guarantee |
| always / necessarily / guarantees | universal claim: look for counterexamples |
| all else being equal | hold other factors fixed |
| subject to a constraint | satisfy the constraint, then optimize objective |
| on the same dataset | comparison is conditional on fixed observations |
| assuming class-conditional distributions remain unchanged | prevalence variation with fixed conditional distributions |
| out-of-fold predictions | predictions for rows excluded from that fitted fold |
| select all that apply | assess each statement independently, not just the best option |

Words such as always do not automatically make an option false: recall monotonicity under fixed scores is a valid guarantee. Check assumptions and the exact claim.

For further review: [model workflow](../modeling/01-workflow-validation.md), [calibration](../modeling/05-clustering-calibration.md), [optional timed mock](../09-mock.md).
