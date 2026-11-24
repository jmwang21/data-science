# 05.6 — Hypothesis Testing, A/B Testing & Power

[← Chapter index](../05-statistics.md) · [Previous](05-inference.md)

> Core review · 中文觀念、英文題目、逐步解析與變化題。

## Hypothesis testing & p-value 

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

## Choosing statistical tests (P0) 

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

## A/B testing — worked example 

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

## Power, MDE & sample size 

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

## Statistical versus commercial decisions

A treatment conversion rate goes from 10% to 11%. Absolute lift=1 percentage point, relative lift=10%. p=.03 at alpha=.05 gives reject H0 under assumptions, not 97% probability treatment works. If operating costs exceed benefit, significance alone is insufficient.

| English wording | Interpretation |
|---|---|
| Fail to reject | Insufficient evidence, not proof H0 true |
| Type I error | P(reject H0 given H0 true) |
| Type II error | P(fail to reject given specified alternative) |
| Power | 1−beta under specified alternative |
| Minimum detectable effect | Planned effect size at target power |

### Worked variation: sample size

All else equal, sample size approximately proportional to 1/MDE². Halving MDE from 2 pp to 1 pp needs about 4× sample size. Higher power or lower alpha needs more observations. Randomization unit matters: repeated transactions within one merchant are not independent observations.

<details><summary>Variation 1: A team checks p&lt;.05 daily and stops at first significance. Is nominal alpha maintained by ordinary fixed-horizon testing?</summary>

Generally no. Repeated opportunities inflate false positives. Use a prespecified sequential method or a fixed horizon.

</details>

<details><summary>Variation 2: 20 independent true null hypotheses are tested at alpha=.05. Probability of at least one false positive?</summary>

1−.95^20≈.642. Bonferroni controls family-wise error by per-test alpha=.05/20; independence not required for the bound.

</details>

<details><summary>Variation 3: The A/B test is not significant. Can we conclude variants are equivalent?</summary>

No. Failure to reject may reflect low power. Equivalence requires a prespecified practical margin and appropriate equivalence test.

</details>

[← Chapter index](../05-statistics.md)
