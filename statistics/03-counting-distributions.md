# 05.3 — Counting & Probability Distributions

[← Chapter index](../05-statistics.md) · [Previous](02-bayes.md) · [Next →](04-expectation.md)

> Core review · 中文觀念、英文題目、逐步解析與變化題。

## Permutations & combinations 

| 是否在意順序 | 是否可重複 | 公式 |
|---|---|---|
| 是 | 否 | `P(n,k) = n!/(n−k)!` |
| 否 | 否 | `C(n,k) = n!/[k!(n−k)!]` |
| 是 | 是 | `n^k` |
| 否 | 是 | `C(n+k−1,k)` |

Example：12 fraud、28 legitimate，隨機抽 5 筆不放回，恰好 2 fraud：

**C(12,2) × C(28,3) / C(40,5)**

也可用固定排列的連乘機率乘以 C(5,2)。至少一個使用 complement 更簡單；不是所有題目都需要列舉組合。

## Distributions — 核心複習與補充 

| Distribution | When to use | Mean | Variance |
|---|---|---:|---:|
| Bernoulli(p) | 單次成功/失敗 | p | p(1-p) |
| Binomial(n,p) | n 次獨立、固定 p | np | np(1-p) |
| Hypergeometric(N,K,n) | 有限母體、不放回抽樣 | nK/N | np(1-p)(N-n)/(N-1) |
| Poisson(λ) | 固定區間事件數、理想化常率獨立增量 | λ | λ |
| Normal(μ,σ²) | 連續鐘形分布 | μ | σ² |
| Exponential(λ) | Poisson process 等待時間 | 1/λ | 1/λ² |
| Geometric(p) | Trials until first success | 1/p | (1-p)/p² |
| Uniform(a,b) | Constant density on interval | (a+b)/2 | (b-a)²/12 |

Hypergeometric 的 p=K/N，N>1。

### Binomial

**P(X=k) = C(n,k) × p^k × (1−p)^(n−k)**

n=8,p=.2：恰好 3 次成功=C(8,3)(.2³)(.8⁵)；至少一次=1-.8⁸；期望=1.6。

### Hypergeometric

**P(X=k) = C(K,k) × C(N−K,n−k) / C(N,n)**

若抽樣比重很小，可近似 Binomial，但精確答案仍使用 hypergeometric。

### Poisson & Exponential（補充）

`P(X=k) = exp(−λ) × λ^k / k!`。若平均每小時 3 筆 ticket，Poisson 假設下 0 筆機率=e⁻³。等候下一件的時間 T：`P(T>t) = exp(−λt)`。單位一致；若 arrival rate 隨時段變動或事件群聚，不應直接採固定 λ。

```python
from scipy.stats import binom, hypergeom, poisson, norm
binom.pmf(3,8,.2)          # exactly 3
binom.sf(0,8,.2)           # at least one
hypergeom.pmf(2,40,12,5)
poisson.pmf(0,3)
norm.cdf(1.96)
```

`pmf` 離散點機率；`pdf` 連續密度，不是點機率；`cdf` P(X<=x)；`sf` P(X>x)；`ppf` 分位數。At least k = sf(k-1)（離散）。

## Normal distribution 

`Z = (X−μ)/σ`。Standard Normal mean=0, variance=1。約 68%、95%、99.7% 落於 mean ±1、2、3 SD（常態假設）。

Example: order amount normally distributed, μ=100, σ=20. P(X>130)=1−Φ(1.5)≈.0668。

```python
norm.sf(130,loc=100,scale=20)
norm.ppf(.975)  # ≈1.96
```

> ⚠️ 真實 GMV 常右偏，不可因為需要常態公式就假設原始 GMV 正態。

## Geometric and Uniform (P1)

Geometric: P(T=k)=(1−p)^(k−1)p, with T counting trials including success. Uniform(a,b): interval probability=(d−c)/(b−a) for [c,d] inside support. Compare the wording "number of successes in 10 attempts" (binomial) and "number of attempts until first success" (geometric).

## Worked variations: same wording, different assumptions

Exactly 3 successes in 4 independent draws, p=.3: C(4,3)×.3³×.7=.0756.
From a population of 10 with 3 positives, select 4 without replacement: C(3,3)C(7,1)/C(10,4)=7/210=.0333.
With Poisson rate 2/hour, at least one event in one hour: 1−e^(−2).

```python
from scipy.stats import binom, hypergeom, poisson, norm
binom.pmf(3,4,.3)
hypergeom.pmf(3,10,3,4)
poisson.sf(0,2)
norm.sf(130,loc=100,scale=20)
```

**English variations:** exactly k → pmf(k); at most k → cdf(k); at least k → sf(k−1), for discrete variables. For continuous variables, above threshold → sf(threshold).

<details><summary>Variation 1: A committee of 3 is selected from 5 analysts and 4 engineers. How many committees contain exactly two analysts?</summary>

C(5,2)C(4,1)=40. If all committees are equally likely, probability=40/C(9,3)=40/84.

</details>

<details><summary>Variation 2: A machine succeeds with probability .2 per independent attempt. Expected trials until first success? Probability first success on third attempt?</summary>

E[T]=5; P(T=3)=.8²×.2=.128. Geometric, not binomial: the number of attempts is random.

</details>

[← Chapter index](../05-statistics.md) · [Next →](04-expectation.md)
