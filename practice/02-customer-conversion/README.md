# Practice 02 — Customer Conversion Prediction

[← Practice index](../README.md)

**English assessment · 3 linked coding tasks · suggested 60 minutes**

Original synthetic practice inspired by the reported end-to-end coding workflow. Unlike Practice 01, this task includes historical revisions, temporal features and validation-based model selection.

## Dataset

Use seven CSVs in [data/](data/).

| Files | Grain / columns |
|---|---|
| customers.csv | customer: customer_id, channel, plan, signup_date |
| sessions_1/2.csv | session: session_id, customer_id, session_date, duration_sec |
| purchases_1/2/3.csv | purchase revision: purchase_id, customer_id, purchase_date, amount, status, updated_at |
| labels.csv | customer: customer_id, converted (March 2025 binary outcome) |

Customer identifiers have leading zeros. Headers contain inconsistent case/whitespace. Sessions may repeat identically; purchases may have revisions or identical duplicates. All timestamps are in the same local time zone. Feature window is [2025-01-01, 2025-03-01). Prediction cutoff is 2025-03-01 00:00:00.

## Q1 — Data cleaning and feature table (suggested 30 min)

Implement `build_features(data_dir)`.

1. Normalize headers, preserve IDs as strings and concatenate partitions of each logical table.
2. Deduplicate sessions by session_id. Parse purchase updated_at and discard revisions at or after cutoff (or invalid updated_at); then retain the latest revision per purchase_id. Ties in latest timestamp are identical. **Resolve revisions before filtering event date, status or amount.**
3. Retain sessions with valid date in feature window and finite nonnegative duration. Retain purchases with valid date in the window, finite nonnegative amount, and status completed or refunded. Ignore unknown customer IDs.
4. Aggregate sessions: session_count, active_days (distinct session dates), avg_duration, last_seen_days (cutoff date minus latest valid session date). Aggregate purchases: purchase_count (completed only), total_spend (completed only), jan_spend, feb_spend (completed only), refund_rate (refunded / all valid completed or refunded purchases).
5. Left join both aggregate tables and labels onto customers. Fill missing counts and spend with zero. Keep avg_duration, last_seen_days and refund_rate missing when no qualifying events exist; leave plan missing. Compute tenure_days from signup_date to cutoff. Return exactly one row per customer sorted by customer_id.

Output schema:

```python
['customer_id','channel','plan','tenure_days','session_count','active_days',
 'avg_duration','last_seen_days','purchase_count','total_spend','jan_spend',
 'feb_spend','refund_rate','converted']
```

## Q2 — Model development and selection (suggested 20 min)

Implement `train_model(features)`.

- Exclude customer_id and converted from X.
- First split off 30% with stratify=y and random_state=42; split those rows 50/50 into validation and test, stratifying again and using random_state=42. Use the first returned half as validation.
- Numeric: median imputation and StandardScaler. Categorical (channel, plan): most-frequent imputation and OneHotEncoder(handle_unknown='ignore').
- Train two independent pipelines on training data: LogisticRegression(max_iter=1000, random_state=42) and RandomForestClassifier(n_estimators=100, min_samples_leaf=4, random_state=42).
- Compare validation **ROC-AUC** using positive-class probabilities. Select higher AUC; ties go to logistic regression. Do not refit on validation or test for this task.
- Return selected model, selected model name ('logistic' or 'forest'), validation score dictionary, X_test, y_test.

## Q3 — Model evaluation (suggested 10 min)

Implement `evaluate(model, X_test, y_test)`. At threshold .5 inclusive compute accuracy, precision, recall, F1, specificity, ROC-AUC, average precision and log loss. Use zero_division=0 where applicable. Return a dictionary with exactly:

```python
['accuracy','precision','recall','f1','specificity','roc_auc','average_precision','log_loss']
```

Also return a DataFrame `predictions` with customer_id, probability and prediction for test rows in X_test's order (retrieve IDs from features using original test indexes; do not use ID as a feature). The function may return metrics only; create the prediction table in the main script. Do not tune on test.

## Self-check (no solution)

- Features shape: (300, 14); unique customer_id; split sizes 210/45/45.
- Growth is not requested. Refund rate is between 0 and 1 when defined.
- Check event revisions were resolved as of cutoff, not by arbitrary file order.
- All test prediction probabilities must be finite and between 0 and 1.
- Run end-to-end with no errors.

Start with [starter.py](starter.py). Reference solution is separately stored under [solution/](solution/); open after attempting.
