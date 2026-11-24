# 09 — Timed Mock Assessment

![Chapter 09](assets/09.svg)

[返回目錄](README.md)

> **90 minutes · 11 original questions · 100 practice points**
> This is a synthetic practice assessment, not an official test or a reproduction of actual questions. Suggested timing: Q1–2 (12 min), Q3–8 (18 min), Q9 (30 min), Q10 (18 min), Q11 (12 min). Run the code and verify outputs before time runs out.

> **Optional:** 這是獨立原創限時模擬考，不是閱讀進度要求。此處 Q11 是分類評估；若要複習 free-form regression、70/15/15 split 與 MAE，另見 [House Price Regression](cases/03-house-price.md)。

## Probability & Statistics — 2 questions (10 points each)

### Q1 — Conditional Bernoulli trials

A box is selected once: A with probability 0.4, B with probability 0.6. Conditional on the selected box, three draws with replacement are independent. The probability of a red card is 0.75 for A and 0.25 for B. What is the probability of exactly two red cards? Given exactly two red cards, what is the probability box A was selected?

<details><summary>Solution & concept correction</summary>

P(E|A)=3×.75²×.25=.421875; P(E|B)=3×.25²×.75=.140625. Total P(E)=.4×.421875+.6×.140625=.253125. Bayes: P(A|E)=.16875/.253125=2/3. Do not use the unconditional red probability in a binomial model: draws share a latent box and are not marginally independent.

</details>

### Q2 — Expected value with repeated cards

Draw four cards independently, uniformly from five types, with replacement. Earn $10 for each distinct type appearing and an additional $20 if all four draws are different. Calculate expected reward.

<details><summary>Solution & concept correction</summary>

E[distinct]=5(1-(4/5)^4)=2.952. P(all different)=5×4×3×2/5⁴=.192. Expected reward=10×2.952+20×.192=33.36. Linearity works even when the distinct-type indicators and bonus event are dependent.

</details>

## Machine Learning — 6 multiple-select questions (5 points each)

Select **all** correct statements. Suggested scoring: 5 points for the exact set, 0 otherwise; review individual options afterwards.

### Q3 — Generalization

A. A large train-validation accuracy gap can indicate overfitting.  
B. A gap always proves overfitting.  
C. Regularization may improve validation performance.  
D. A distribution shift can also explain the gap.

<details><summary>Answer & reasoning</summary>

A, C, D. B is too absolute; compare splits, label quality and distributions before attributing the gap.

</details>

### Q4 — Ensemble

A. Bagging can reduce variance.  
B. Random forests use randomized feature subsets.  
C. Gradient boosting trains all trees independently.  
D. Stacking should use out-of-fold base predictions to train the meta-model.

<details><summary>Answer & reasoning</summary>

A, B, D. Boosting is sequential. Stacking using in-sample predictions can leak training information into meta-model features.

</details>

### Q5 — AUC & Log Loss

A. A strictly increasing transform of scores preserves ROC-AUC.  
B. Identical AUC guarantees identical log loss.  
C. Log loss penalizes confident wrong predictions.  
D. AUC requires scores in [0,1].

<details><summary>Answer & reasoning</summary>

A, C. AUC depends on ranking and ties; probabilities can have different quality with identical ranking. Scores need not be probabilities.

</details>

### Q6 — Thresholds & class imbalance

A. Lowering a threshold on fixed probabilities cannot reduce recall.  
B. Lowering the threshold always reduces precision.  
C. Accuracy can hide poor minority-class performance.  
D. Precision-recall metrics can be useful for rare positives.

<details><summary>Answer & reasoning</summary>

A, C, D. Newly positive observations may include enough true positives for precision to increase, so B is false. Recall is nondecreasing as threshold falls, assuming positives exist.

</details>

### Q7 — Leakage

A. Fit the scaler on all data before splitting.  
B. Use Pipeline inside CV.  
C. Select the highest test AUC from 12 models and report it as unbiased.  
D. Use group-based splits when generalizing to unseen entities with repeated rows.

<details><summary>Answer & reasoning</summary>

B, D. A leaks statistics; C uses the test set for selection, introducing optimistic selection bias.

</details>

### Q8 — Binary classification

A. With binary labels 0 and 1, logistic regression with the usual sigmoid models the probability of class 1.  
B. Increasing C in sklearn LogisticRegression strengthens regularization.  
C. `predict_proba` columns follow `classes_`.  
D. F1 is the harmonic mean of precision and recall.

<details><summary>Answer & reasoning</summary>

A, C, D. C is inverse regularization strength in sklearn; larger C weakens regularization. Check the class ordering before selecting a probability column.

</details>

## Coding — 3 linked tasks (50 points total)

### Q9 — Build a clean feature table (25 points)

Use the seven CSV files in [data/](data/). All IDs are strings, including leading zeros. Four logical tables:

| Files | Grain | Keys |
|---|---|---|
| merchants.csv | merchant | merchant_id |
| orders_1.csv, orders_2.csv, orders_3.csv | order | order_id, merchant_id |
| tickets_1.csv, tickets_2.csv | ticket | ticket_id, merchant_id |
| labels.csv | merchant | merchant_id |

The target `churn` refers to a synthetic future outcome. Features must use only January and February 2025. Some headers have inconsistent case/whitespace, and duplicate event IDs represent identical events. Missing or invalid amounts are not valid orders.

- Load all files, normalize headers, preserve leading zeros, concatenate partitions, and deduplicate events by ID (6 points).
- Convert date/amount fields, filter completed finite-amount orders to `[2025-01-01, 2025-03-01)`; count tickets in the same window (4 points).
- Aggregate orders by merchant: total_gmv, total_orders, unique_customers, jan_gmv, feb_gmv. Aggregate tickets separately: ticket_count, open_tickets (6 points).
- Left join aggregates and labels onto merchant dimension, preserving all merchants and avoiding event fan-out; zero-fill missing aggregates (5 points).
- Compute tenure_days at 2025-03-01 and gmv_growth (NaN if jan_gmv=0). Return sorted table with schema below, validate uniqueness and row count (4 points).

```python
['merchant_id','category','tier','tenure_days','total_gmv','total_orders',
 'unique_customers','jan_gmv','feb_gmv','gmv_growth','ticket_count',
 'open_tickets','churn']
```

### Q10 — Train a binary classifier (15 points)

Use Q9 output. Exclude merchant_id and churn from features. Stratified 75/25 train/test split, random_state=42. Numeric columns: median impute, StandardScaler. Categorical columns (`category`, `tier`): most-frequent impute and OneHotEncoder(handle_unknown='ignore'). Build ColumnTransformer + Pipeline + LogisticRegression(max_iter=1000, random_state=42). Fit training only. Return model and test inputs.

Scoring: split/features 4; numeric preprocessing 3; categorical preprocessing 3; combined pipeline 3; fit and output 2.

### Q11 — Evaluate classifier (10 points)

Get probabilities for positive class 1 and generate binary labels at threshold .5. Return dictionary containing accuracy, precision, recall, f1, roc_auc and log_loss. Use zero_division=0 where applicable. Do not refit or tune on test data.

Scoring: correct probability 2; labels 2; label metrics 3; probability metrics 2; exact dictionary 1.

<details><summary>Reference solution & executable checks — open after attempting</summary>

Run `python solutions/mock_solution.py` from repository root after installing numpy, pandas and scikit-learn. The full source is [here](solutions/mock_solution.py). Expected feature table: 240 rows × 13 columns, unique merchant_id; test: 60 rows. In the tested environment accuracy≈0.7333, ROC-AUC≈0.7644, log loss≈0.5934. Minor version-dependent numerical differences are possible. Do not optimize to these synthetic test metrics.

Common pitfalls: normalizing headers after type-specific parsing loses leading-zero IDs; joining raw orders with raw tickets multiplies rows; dropping rows after left joins removes inactive merchants; filling growth with zero hides undefined growth; fitting imputation/scaling on the full dataset leaks; thresholding y_test instead of predicted probabilities is invalid; passing class labels to AUC or log loss discards ranking/confidence information.

</details>
