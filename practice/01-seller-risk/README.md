# Practice 01 — Seller Risk Prediction

[← Practice index](../README.md)

**English assessment · 3 linked coding tasks · suggested 55–60 minutes**

This is an original synthetic exercise inspired by the reported seven-CSV data manipulation → binary classification → evaluation workflow. Not an actual exam question. Use pandas, NumPy and scikit-learn. Do not read the reference solution before attempting.

## Dataset

Use the seven CSV files under [data/](data/). Four logical tables:

| Files | Grain / columns |
|---|---|
| sellers.csv | one seller: seller_id, category, tier, join_date |
| orders_1/2/3.csv | order: order_id, seller_id, order_date, status, amount, buyer_id |
| tickets_1/2.csv | support ticket: ticket_id, seller_id, created_at, status |
| labels.csv | one seller: seller_id, high_risk (binary outcome in March 2025) |

IDs are strings with leading zeros. Headers may have inconsistent case or surrounding spaces. Repeated event IDs are identical records. Some dates and amounts are invalid; some events belong to unknown sellers. All seller labels are available. Feature window is January 1 inclusive to March 1 exclusive, 2025; cutoff is March 1.

## Q1 — Data manipulation (suggested 30 min)

Implement `build_features(data_dir)`, returning a DataFrame with **one row per seller**, including those with no valid events.

1. Load and normalize all file headers; preserve identifiers; concatenate partitions of the same logical table and remove duplicate event IDs.
2. Parse dates and amounts. Retain orders with status completed, finite numeric amount and date within the feature window. Retain all support tickets with valid dates in the same window. Ignore events whose seller is absent from sellers.csv.
3. Aggregate orders separately: completed_orders, total_gmv, unique_buyers, jan_gmv, feb_gmv. Aggregate tickets separately: ticket_count, open_tickets.
4. Join the aggregate tables and labels to the seller dimension. Fill missing count and GMV aggregates with zero; leave missing tier as missing. Compute tenure_days as calendar days from join_date to cutoff, and gmv_growth=(feb_gmv−jan_gmv)/jan_gmv; undefined growth must remain NaN.
5. Return rows sorted by seller_id, columns in exact order:

```python
['seller_id','category','tier','tenure_days','completed_orders','total_gmv',
 'unique_buyers','jan_gmv','feb_gmv','gmv_growth','ticket_count','open_tickets',
 'high_risk']
```

## Q2 — Binary model development (suggested 15 min)

Implement `train_model(features)`. Exclude seller_id and high_risk from X. Use stratified train/test split with test_size=0.25 and random_state=42. Numeric columns: median imputation then StandardScaler. Categorical columns (category, tier): most-frequent imputation then OneHotEncoder(handle_unknown='ignore'). Build a ColumnTransformer and Pipeline with LogisticRegression(max_iter=1000, random_state=42). Fit training rows only. Return model, X_test, y_test.

## Q3 — Prediction and evaluation (suggested 10 min)

Implement `evaluate(model, X_test, y_test)`. Get positive-class probabilities, use threshold 0.5 (inclusive) to produce predictions, and return a dictionary with keys:

```python
['accuracy','precision','recall','f1','roc_auc','log_loss']
```

Use zero_division=0 where applicable. Do not tune on test. Report which metrics use scores and which use labels.

## Self-check (no solution)

- Output shape: (240, 13); unique seller_id; 60 test rows.
- No infinity in growth or numerical features; NaN growth allowed.
- All rows must remain, including sellers without events.
- No target or ID in model features.
- Check that your Python file runs from start to finish.

Start with [starter.py](starter.py). Reference solution is separately stored under [solution/](solution/); open after attempting.
