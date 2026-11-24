# 05.4 — Expected Value, Variance & Descriptive Statistics

[← Chapter index](../05-statistics.md) · [Previous](03-counting-distributions.md) · [Next →](05-inference.md)

> Core review · 中文觀念、英文題目、逐步解析與變化題。

## Expected value & variance 

$$E[X]=\sum_x xP(X=x)\qquad Var(X)=E[X^2]-E[X]^2$$

$E[aX+b]=aE[X]+b$；$Var(aX+b)=a^2Var(X)$。Expectation 可線性相加，不需要獨立；$Var(X+Y)=Var(X)+Var(Y)+2Cov(X,Y)$。獨立 ⇒ covariance=0，反向通常不成立。

Example: 20% earn $100, 80% earn $0. E=20，E[X²]=2000，Var=1600，SD=40。若兩個獨立 campaign，總期望 40，variance 3200。

```python
import numpy as np
x = np.array([0,100]); p = np.array([.8,.2])
mean = np.sum(x*p)
var = np.sum((x-mean)**2*p)
```

Population variance 分母 n；unbiased sample variance 分母 n−1（IID 假設）。`np.var(x,ddof=1)`，pandas `.var()` 預設 ddof=1。

## Bernoulli, indicator variables & tricky expectation 

Let $I_A=1$ when event A occurs, otherwise 0. Then $E[I_A]=P(A)$ and $Var(I_A)=p(1-p)$. Linearity $E[\sum_i I_i]=\sum_i E[I_i]$ does **not** require independence. Variance additivity generally requires zero covariance.

### Example 1 — dependent Bernoulli trials

Two cards are drawn without replacement from 4 red and 6 blue cards. Let X count red cards. Each position has marginal red probability 0.4; E[X]=0.8 even though draws are dependent. P(X=2)=4/10 × 3/9=2/15, not 0.16.

### Example 2 — random reward and conditional expectation

A customer is premium with probability 0.3. Conversion probabilities are 0.6 for premium and 0.2 otherwise. Reward per conversion is 100 for premium, 40 otherwise.

P(convert)=0.3×0.6+0.7×0.2=0.32. Expected reward is 0.3×0.6×100+0.7×0.2×40=23.6. Multiplying the unconditional mean reward amount by conversion probability would incorrectly ignore dependence.

### Example 3 — collect all types (independent uniform draws)

There are 4 equally likely card types, with replacement. After k distinct types have been collected, probability of a new type is (4-k)/4. The waiting time is geometric (counting trials including success), with mean 4/(4-k). Expected draws to complete the collection: 1+4/3+2+4=25/3.

If probabilities are unequal, this simple harmonic formula does not apply. Without replacement is a different sampling process.

**Practice:** Five independent draws from three equally likely types. What is the expected number of distinct types? What is the probability all three appear?

<details><summary>Solution</summary>

Use indicator variables: each type appears with probability 1-(2/3)^5, hence E[distinct]=3[1-(2/3)^5]=211/81≈2.6049. Inclusion-exclusion gives P(all)=1-3(2/3)^5+3(1/3)^5=150/243≈0.6173. The indicators for distinct types are not independent.

</details>

## Descriptive statistics and joint distributions (P0) 

Mean uses all values and is sensitive to extremes; median is the 50th percentile; mode is the most frequent. For [1,2,3,4,100], mean=22, median=3. Sample variance divides squared deviations by n−1; standard deviation is its square root. IQR=Q3−Q1. Quantile algorithms can differ for small samples.

Covariance measures joint variation in original units. Pearson r=Cov(X,Y)/(SD(X)SD(Y)) measures linear association; Spearman measures monotonic association via ranks. Zero correlation does not imply independence (e.g. symmetric X and Y=X²). Correlation does not establish causality; confounding and Simpson's paradox can reverse aggregate patterns.

Conditional expectation E[Y|X] is the mean within a specified X; law of total expectation E[Y]=E[E[Y|X]], total variance Var(Y)=E[Var(Y|X)]+Var(E[Y|X]). Example: half customers have conditional mean 10, half 20, both conditional variance 4. Overall mean 15, variance 4+25=29.

### Worked descriptive variation — Simpson's paradox

Group A: 9 of 10 successes for easy cases, 1 of 10 for hard cases. Group B: 80 of 100 for easy, 0 of 1 for hard. A is better within both strata (.9>.8, .1>0), but aggregate A=10/20=.5 and B=80/101≈.792. Group composition reverses the comparison. Report stratified metrics and investigate confounding.

## Worked variation — transformation and dependence

A lottery pays 100 with probability .2 and zero otherwise. E[X]=20, Var(X)=100²×.2−20²=1600. A fee of 5 yields E[X−5]=15, variance unchanged. Doubling payoff yields E[2X]=40, Var(2X)=6400.

For independent X,Y with variances 4 and 9, Var(X+Y)=13. If covariance=2, Var(X+Y)=17; Var(X−Y)=9. Variance nonnegative; covariance signed.

<details><summary>Variation 1: Four independent draws with replacement from five equally likely card types. Expected number of distinct types?</summary>

Use an indicator for each type appearing. E[distinct]=5[1−(.8)^4]=2.952. Independence among the indicators is not required.

</details>

<details><summary>Variation 2: Average order value is 50 for 10 orders and 100 for 90 orders. What is overall average?</summary>

(50×10+100×90)/100=95, not 75. Weight by number of orders.

</details>

[← Chapter index](../05-statistics.md) · [Next →](05-inference.md)
