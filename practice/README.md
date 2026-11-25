# End-to-End Coding Practice

[← Handbook](../README.md)

Two original English practice sets, inspired by reported data manipulation → model training → evaluation tasks. Each includes seven CSVs, four logical tables, three linked tasks, starter code, and a separate reference solution.

| Practice | Main variation | Time target |
|---|---|---|
| [01 Seller Risk Prediction](01-seller-risk/README.md) | duplicates, invalid dates/amounts, missing categories, fanout, binary Logistic Regression, six metrics | 55–60 min |
| [02 Customer Conversion Prediction](02-customer-conversion/README.md) | latest revision as of cutoff, temporal features, refund rate, LR vs RF validation selection, eight metrics | 60 min |

## How to practice

Clone or download the repository. Install dependencies if needed:

```bash
pip install pandas numpy scikit-learn
```

Run your completed starter from each directory:

```bash
python practice/01-seller-risk/starter.py
python practice/02-customer-conversion/starter.py
```

Do not open solution/reference.py before attempting. Afterward, compare with the reference and inspect any failing assertions or metric discrepancies. Output row counts and schemas are provided in each problem, not model scores. Suggested times are practice targets, not official CodeSignal allocations.

All data are synthetic; no actual exam question, dataset, or personal data is included.
