# 04.9 — Naive Bayes

[← ML index](../04-ml.md) · [Next →](clustering.md)

## 1. Model assumptions and prediction

**English:** Naive Bayes assumes features are conditionally independent given the class.

Bayes: P(c|x) proportional to P(c) × P(x|c). Under conditional independence, P(x|c)=Π P(xⱼ|c). This does not imply marginal independence.

Example P(spam)=.2, P(word|spam)=.8, P(word|ham)=.1. Posterior=.16/(.16+.08)=2/3. If a second word has respective probabilities .5 and .1 and conditional independence holds, posterior=(.2×.8×.5)/(.2×.8×.5+.8×.1×.1)=.9091. If both words always occur together, multiplication double-counts evidence.

| Variant | Input assumption |
|---|---|
| GaussianNB | continuous Gaussian class-conditional features |
| MultinomialNB | nonnegative counts / suitable nonnegative features |
| BernoulliNB | binary feature presence/absence |

Laplace smoothing prevents zero likelihood for unseen feature counts: for class token counts (2,0) and vocabulary size 2, alpha=1 gives probabilities (3/4,1/4). Without smoothing, unseen word probability zero can eliminate a class. Computation often uses log likelihood to avoid multiplying tiny numbers.

## 2. Practical comparison

Fast to train, useful as text baseline and small datasets; conditional independence is often unrealistic, probability calibration may be poor even if classification works. GaussianNB distributional assumption is within class, not necessarily global feature normality.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
m=make_pipeline(CountVectorizer(),MultinomialNB(alpha=1))
m.fit(['win prize','claim reward','project meeting','weekly meeting'],[1,1,0,0])
print(m.predict(['claim prize','weekly project']))
```

## Concept variations

<details><summary>1. Does Naive Bayes require marginal independence of features?</summary>

No. The assumption is independence conditional on each class.

</details>

<details><summary>2. Why might duplicated, highly correlated features distort probabilities?</summary>

They may count the same evidence multiple times, producing overconfident estimates.

</details>

<details><summary>3. What happens if a class has never observed a vocabulary token and no smoothing is used?</summary>

Its likelihood for that token is zero; smoothing avoids eliminating the class solely due to finite sample absence.

</details>
