# 05 — Probability & Statistics

![Chapter 05](assets/05-statistics.svg)

[返回目錄](README.md)


## 1. Basic probability & set operations

| Concept | Formula |
|---|---|
| Complement | $P(A^c)=1-P(A)$ |
| Union | $P(A\cup B)=P(A)+P(B)-P(A\cap B)$ |
| Conditional | $P(A\mid B)=P(A\cap B)/P(B)$ |
| Multiplication | $P(A\cap B)=P(A\mid B)P(B)$ |
| At least one | $1-P(\text{none})$ |

Example: 40% buy A, 30% buy B, 10% buy both. P(A or B)=.6, P(A only)=.3, P(A|B)=1/3. Mutually exclusive means intersection=0；independent means intersection=P(A)P(B)。非零機率的互斥事件不獨立。

## 2. Bayes' theorem & total probability

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

## 3. Independence

Independent events: $P(A\cap B)=P(A)P(B)$；等價於 $P(A\mid B)=P(A)$（分母非零）。Conditional independence: $P(A,B\mid C)=P(A\mid C)P(B\mid C)$，不代表 marginal independence。

Three independent attempts, each success=.2: none=.8³=.512, at least one=.488, exactly one=3(.2)(.8²)=.384。

> ⚠️ Without replacement 通常不獨立。若題目未給獨立，不可直接把 marginal probabilities 相乘。

## 4. Permutations & combinations

| 是否在意順序 | 是否可重複 | 公式 |
|---|---|---|
| 是 | 否 | $P(n,k)=n!/(n-k)!$ |
| 否 | 否 | $C(n,k)=n!/[k!(n-k)!]$ |
| 是 | 是 | $n^k$ |
| 否 | 是 | $C(n+k-1,k)$ |

Example：12 fraud、28 legitimate，隨機抽 5 筆不放回，恰好 2 fraud：

$$\frac{\binom{12}{2}\binom{28}{3}}{\binom{40}{5}}$$

也可用固定排列的連乘機率乘以 C(5,2)。至少一個使用 complement 更簡單；不是所有題目都需要列舉組合。

## 5. Distributions — 核心複習與補充

| Distribution | When to use | Mean | Variance |
|---|---|---:|---:|
| Bernoulli(p) | 單次成功/失敗 | p | p(1-p) |
| Binomial(n,p) | n 次獨立、固定 p | np | np(1-p) |
| Hypergeometric(N,K,n) | 有限母體、不放回抽樣 | nK/N | np(1-p)(N-n)/(N-1) |
| Poisson(λ) | 固定區間事件數、理想化常率獨立增量 | λ | λ |
| Normal(μ,σ²) | 連續鐘形分布 | μ | σ² |
| Exponential(λ) | Poisson process 等待時間 | 1/λ | 1/λ² |

Hypergeometric 的 p=K/N，N>1。

### Binomial

$$P(X=k)=\binom nk p^k(1-p)^{n-k}$$

n=8,p=.2：恰好 3 次成功=C(8,3)(.2³)(.8⁵)；至少一次=1-.8⁸；期望=1.6。

### Hypergeometric

$$P(X=k)=\frac{\binom Kk\binom{N-K}{n-k}}{\binom Nn}$$

若抽樣比重很小，可近似 Binomial，但精確答案仍使用 hypergeometric。

### Poisson & Exponential（補充）

$P(X=k)=e^{-\lambda}\lambda^k/k!$。若平均每小時 3 筆 ticket，Poisson 假設下 0 筆機率=e⁻³。等候下一件的時間 T：$P(T>t)=e^{-\lambda t}$。單位一致；若 arrival rate 隨時段變動或事件群聚，不應直接採固定 λ。

```python
from scipy.stats import binom, hypergeom, poisson, norm
binom.pmf(3,8,.2)          # exactly 3
binom.sf(0,8,.2)           # at least one
hypergeom.pmf(2,40,12,5)
poisson.pmf(0,3)
norm.cdf(1.96)
```

`pmf` 離散點機率；`pdf` 連續密度，不是點機率；`cdf` P(X<=x)；`sf` P(X>x)；`ppf` 分位數。At least k = sf(k-1)（離散）。

## 6. Expected value & variance

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

## 7. Normal distribution

$Z=(X-\mu)/\sigma$。Standard Normal mean=0, variance=1。約 68%、95%、99.7% 落於 mean ±1、2、3 SD（常態假設）。

Example: order amount normally distributed, μ=100, σ=20. P(X>130)=1−Φ(1.5)≈.0668。

```python
norm.sf(130,loc=100,scale=20)
norm.ppf(.975)  # ≈1.96
```

> ⚠️ 真實 GMV 常右偏，不可因為需要常態公式就假設原始 GMV 正態。

## 8. Sampling distribution & CLT

IID、有限 variance 下，sample mean 的期望 μ，variance σ²/n，standard error σ/√n。CLT 說明標準化 sample mean 隨 n 增大趨近常態；不是原始資料變成常態，也沒有普遍適用的 n≥30 保證。

例：σ=20,n=100，SE=2；n=400，SE=1。樣本數 ×4，SE 減半。若同 merchant 多筆觀察或時序相依，IID standard error 可能低估不確定性，需要 cluster-aware 方法。

## 9. Confidence interval

已知 σ 的常態或大樣本近似：$\bar x\pm z_{1-\alpha/2}\sigma/\sqrt n$。未知 σ、常態 IID 資料時，用 t 分位數與 sample SD：$\bar x\pm t_{n-1,1-\alpha/2}s/\sqrt n$。

例：mean=100, SE=2，95% z CI=[96.08,103.92]。Frequentist 95% CI 意思是重複抽樣並用同一方法建 CI，長期約 95% 區間覆蓋真參數；不是已得到的固定區間有 95% 機率包含固定真值。

比例的大樣本近似：$\hat p\pm z\sqrt{\hat p(1-\hat p)/n}$；小樣本/接近 0 或 1 時可使用 Wilson interval。

## 10. Hypothesis testing & p-value

1. 先定義 H0、H1、alpha、primary metric、分析單位與停止規則。
2. 算 statistic 與 null distribution；p-value 是在 H0 與模型假設下，觀察到至少同樣極端 statistic 的機率。
3. p≤alpha 時 reject H0；否則 fail to reject（不是接受 H0 為真）。

p-value 不是 P(H0 true | data)、不是 effect size、也不是結果可重複機率。Statistically significant 不等於 commercially meaningful。

### Type I / Type II（固定矩陣方向）

Actual positive = H0 false；Actual negative = H0 true；Predicted positive = reject H0。

| Actual ↓ / Decision → | Reject H0 (Positive) | Fail to reject (Negative) |
|---|---|---|
| H0 false | TP / Power = 1−β | FN / Type II = β |
| H0 true | FP / Type I = α* | TN / 1−α* |

*在固定 null 情境中，實際 Type I error rate 可小於設定 α（例如離散檢定）；精確達到 α 時才等號成立。這些是條件於真實狀態的機率，並非無條件的 P(FP)。Power 與特定 alternative effect size 有關。不能把 80% power 說成研究結果有 80% 機率是真的。

## 11. A/B testing — worked example

Control A: 800 conversions / 8000；Treatment B: 920 / 8000。

| Metric | Value |
|---|---:|
| A conversion | 10.0% |
| B conversion | 11.5% |
| Absolute uplift | +1.5 percentage points |
| Relative uplift | +15% |

Two-sided two-proportion z-test：H0 pA=pB；H1 pA≠pB。

$$\hat p=\frac{x_A+x_B}{n_A+n_B},\quad SE_0=\sqrt{\hat p(1-\hat p)(1/n_A+1/n_B)}$$

$$z=\frac{\hat p_B-\hat p_A}{SE_0}$$

```python
from scipy.stats import norm
from math import sqrt
xA,nA,xB,nB=800,8000,920,8000
pA,pB=xA/nA,xB/nB
p=(xA+xB)/(nA+nB)
z=(pB-pA)/sqrt(p*(1-p)*(1/nA+1/nB))
pvalue=2*norm.sf(abs(z))
se=sqrt(pA*(1-pA)/nA+pB*(1-pB)/nB)
ci=((pB-pA)-1.96*se,(pB-pA)+1.96*se)
```

z≈3.06、p≈.0022、absolute uplift CI 約 [0.0054,0.0246]（大樣本近似，CI 用 unpooled SE）。有統計證據不代表一定值得上線：需估算 incremental profit、成本、guardrails 及長期影響。

### Test design pitfalls

| Risk | Example / remedy |
|---|---|
| Peeking | 每天看 p<.05 就停；固定 horizon 或預先設計 sequential test |
| Multiple comparisons | 多 metrics/variants；預先主指標，必要時 Holm/Bonferroni 或 FDR |
| SRM | Sample ratio mismatch；先檢查 assignment/instrumentation |
| Interference | 商家/用戶互相影響；考慮 cluster randomization |
| Novelty / seasonality | 新奇效果、週末活動；設計足夠觀察期間 |
| Post-treatment selection | 僅比較實驗後 active users 可能造成偏誤；優先 ITT |

## 12. Power, MDE & sample size

MDE 是設計想偵測的最小效果，不是觀察到的效果。固定其他條件：

| 改變 | 所需 sample size |
|---|---|
| MDE ↓ | ↑（約與 1/MDE² 成比例） |
| Power ↑ | ↑ |
| α ↓ | ↑ |
| Variance ↑ | ↑ |

兩組等樣本、近似常態且共同 variance σ² 的雙尾檢定：

$$n_{per\ group}\approx\frac{2\sigma^2(z_{1-\alpha/2}+z_{1-\beta})^2}{\delta^2}$$

二元轉換率規劃可用 baseline p 的 variance p(1-p) 做簡化近似。Baseline=.10，MDE=.01 absolute，alpha=.05，power=.8，約 14,128/group（使用 1.96/.8416 的常態分位數；精確規劃可能不同）。要考慮失訪、cluster effect、分流比例。

## 13. Practice questions

1. Fraud prevalence=.02, TPR=.8, FPR=.04. Compute precision.
2. Ten independent trials with p=.3. Probability of at least one success?
3. Mean=50, population SD=12, n=36. Find SE and approximate 95% CI around observed mean 52.
4. An A/B test returns p=.03, but estimated profit uplift is below operating cost. Should the team ship automatically?
5. What happens to required sample size if MDE is halved?

<details><summary>Reference answers</summary>

1. .016/(.016+.0392)=.2899.
2. 1−.7¹⁰≈.97175.
3. SE=2; CI=[48.08,55.92] under stated approximation.
4. No. Evidence of a difference is not sufficient for positive business value; assess CI, cost and guardrails.
5. Approximately quadruples, all else fixed.
</details>

[下一章：Integrated Cases →](06-cases.md)

## 14. Bernoulli, indicator variables & tricky expectation

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

## 15. Descriptive statistics and joint distributions (P0)

Mean uses all values and is sensitive to extremes; median is the 50th percentile; mode is the most frequent. For [1,2,3,4,100], mean=22, median=3. Sample variance divides squared deviations by n−1; standard deviation is its square root. IQR=Q3−Q1. Quantile algorithms can differ for small samples.

Covariance measures joint variation in original units. Pearson r=Cov(X,Y)/(SD(X)SD(Y)) measures linear association; Spearman measures monotonic association via ranks. Zero correlation does not imply independence (e.g. symmetric X and Y=X²). Correlation does not establish causality; confounding and Simpson's paradox can reverse aggregate patterns.

Conditional expectation E[Y|X] is the mean within a specified X; law of total expectation E[Y]=E[E[Y|X]], total variance Var(Y)=E[Var(Y|X)]+Var(E[Y|X]). Example: half customers have conditional mean 10, half 20, both conditional variance 4. Overall mean 15, variance 4+25=29.

### Additional distributions
Geometric(p), trials until first success: P(T=k)=(1−p)^(k−1)p, E[T]=1/p, Var(T)=(1−p)/p². Uniform(a,b): mean (a+b)/2, variance (b−a)²/12; P(c≤X≤d)=(d−c)/(b−a) within support. For geometric p=.2, first success on third trial=.8²×.2=.128. For continuous uniform 0–10, probability of 2–5=.3.

## 16. Choosing statistical tests (P0)

| Question | Method | Key assumptions / interpretation |
|---|---|---|
| One mean vs target | One-sample t-test | IID; exact under normal population |
| Two independent means | Welch t-test | Independent observations, unequal variance allowed |
| Before/after same subject | Paired t-test | Analyze within-pair differences |
| Two proportions | Two-proportion z-test | Sufficient expected counts; independent units |
| Categorical association | Chi-square independence | Adequate expected cell counts; Fisher exact for small 2×2 |
| Three or more means | ANOVA | Classical version assumes independent normal errors and equal variance |
| Rank comparison | Mann–Whitney U (P1) | Independent samples; not universally a median test |

Welch statistic=(mean_A−mean_B)/sqrt(s_A²/n_A+s_B²/n_B). Example: means 10 and 12, each n=100 and SD=5, SE difference≈.707, t≈−2.83. A two-sided test considers either direction. Use effect size and CI in addition to p-value.

Chi-square statistic=sum((O−E)²/E), with E=row total×column total/grand total; df=(r−1)(c−1). ANOVA F compares between-group to within-group variation; a significant result does not identify which groups differ. Preplanned contrasts or adjusted post-hoc tests address that question.

```python
from scipy import stats
a = [10,11,9,12,8]
b = [12,13,11,14,10]
stats.ttest_ind(a,b,equal_var=False)
stats.ttest_rel(a,b) # only if observations really are paired
stats.chi2_contingency([[30,70],[45,55]], correction=False)
stats.f_oneway(a,b,[11,12,10,13,9])
```

### Sampling and inference pitfalls
Selection bias is systematic sampling distortion; larger n reduces random uncertainty but not systematic bias. Bootstrap resamples observations with replacement and recomputes the statistic to estimate uncertainty; preserve clusters for dependent data. A permutation test shuffles exchangeable labels under H0; randomization design matters. One-sided hypotheses must be specified before seeing outcomes. If testing 20 independent true nulls at .05, chance at least one false rejection is 1−.95²⁰≈64.2%; Bonferroni uses .05/20=.0025 per test.

Reference: [SciPy statistics](https://docs.scipy.org/doc/scipy/reference/stats.html).
