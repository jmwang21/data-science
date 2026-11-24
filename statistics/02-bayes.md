# 05.2 — Bayes' Theorem & Total Probability

[← Chapter index](../05-statistics.md) · [Previous](01-probability.md) · [Next →](03-counting-distributions.md)

> Core review · 中文觀念、英文題目、逐步解析與變化題。

## Bayes' theorem & total probability 

$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$$

若 $A_1,\ldots,A_k$ 互斥且涵蓋全部：$P(B)=\sum_i P(B\mid A_i)P(A_i)$。

### Worked example — Fraud detection

Fraud prevalence=1%, sensitivity=90%, FPR=5%。10,000 筆交易：

| Actual ↓ / Predicted → | Flagged | Not flagged |
|---|---:|---:|
| Fraud | 90 | 10 |
| Legitimate | 495 | 9405 |

$$P(Fraud\mid Flagged)=\frac{.9\times .01}{.9\times .01+.05\times .99}=15.38\%$$

即使 recall 高，因低 base rate，precision 仍可能低。當新市場 prevalence 改變，若 TPR/FPR 不變，precision 仍會改變。Bayes 不能隨意假設證據 conditionally independent；只有已知給定類別下獨立，才能將 likelihood 相乘。

## Worked variation — changing the base rate

Fraud prevalence=.02, TPR=.9, FPR=.05. Among 10,000 transactions, TP=180 and FP=490. Precision=.018/(.018+.049)=26.87%. If prevalence becomes .10 while TPR/FPR stay fixed, precision=.09/(.09+.045)=66.67%. Same model operating characteristics, different posterior.

## Worked variation — shared latent source

Choose a box once: A with probability .4, B with .6. Within A red probability=.75, within B=.25; three draws with replacement are conditionally independent. Event E=exactly two red.

P(E|A)=3(.75²)(.25)=.421875; P(E|B)=3(.25²)(.75)=.140625.
P(E)=.4×.421875+.6×.140625=.253125.
P(A|E)=.16875/.253125=2/3.

The unconditional red probability is .45, but using Binomial(3,.45) is incorrect: all draws share the same latent box. Distinguish marginal independence from conditional independence.

<details><summary>Variation 1: A test has 80% sensitivity, 90% specificity, and prevalence 5%. What is the positive predictive value?</summary>

FPR=.1. PPV=.8×.05/(.8×.05+.1×.95)=.04/.135≈29.63%.

</details>

<details><summary>Variation 2: Can you multiply the likelihoods of two positive results without any further assumption?</summary>

No. Multiplication of individual likelihoods requires conditional independence given each hypothesis. Otherwise use joint likelihoods.

</details>

[← Chapter index](../05-statistics.md) · [Next →](03-counting-distributions.md)
