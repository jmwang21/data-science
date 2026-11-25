# 05.5 — Sampling, CLT & Confidence Intervals

[← Chapter index](../05-statistics.md) · [Previous](04-expectation.md) · [Next →](06-testing-ab.md)

> Core review · 中文觀念、英文題目、逐步解析與變化題。

## Sampling distribution & CLT 

IID、有限 variance 下，sample mean 的期望 μ，variance σ²/n，standard error σ/√n。CLT 說明標準化 sample mean 隨 n 增大趨近常態；不是原始資料變成常態，也沒有普遍適用的 n≥30 保證。

例：σ=20,n=100，SE=2；n=400，SE=1。樣本數 ×4，SE 減半。若同 merchant 多筆觀察或時序相依，IID standard error 可能低估不確定性，需要 cluster-aware 方法。

## Confidence interval 

已知 σ 的常態或大樣本近似：`x̄ ± z(1−α/2) × σ/√n`。未知 σ、常態 IID 資料時，用 t 分位數與 sample SD：`x̄ ± t(df=n−1, quantile=1−α/2) × s/√n`。

例：mean=100, SE=2，95% z CI=[96.08,103.92]。Frequentist 95% CI 意思是重複抽樣並用同一方法建 CI，長期約 95% 區間覆蓋真參數；不是已得到的固定區間有 95% 機率包含固定真值。

比例的大樣本近似：`p̂ ± z × √[p̂(1−p̂)/n]`；小樣本/接近 0 或 1 時可使用 Wilson interval。

## Worked variation — individual versus sample mean

Individual order amount is Normal(mean=100, SD=20). P(X>110)=1−Φ(.5)≈.3085. For the mean of 25 independent orders, SE=20/5=4 and P(mean>110)=1−Φ(2.5)≈.0062. Same threshold, different random variable. The z-score denominator depends on whether the question concerns an individual or sample mean.

### Worked example

Mean=52, population SD=12, n=36: SE=2; approximate 95% z CI=52±1.96×2=[48.08,55.92]. If n=144, same mean, SE=1, CI≈[50.04,53.96]. Unknown sigma: use s and t critical value under appropriate assumptions.

For proportion=.2,n=400, approximate SE=.02, CI=[.1608,.2392]. Near 0/1, prefer Wilson or other appropriate interval.

<details><summary>Variation 1: A 95% confidence interval excludes the null value. Is the corresponding two-sided test significant at 5%?</summary>

Yes, for matching interval/test construction and assumptions. This does not imply a practically important effect.

</details>

<details><summary>Variation 2: Can increasing sample size fix selection bias?</summary>

No. It reduces sampling variance under the model, but may precisely estimate the wrong population quantity.

</details>

[← Chapter index](../05-statistics.md) · [Next →](06-testing-ab.md)
