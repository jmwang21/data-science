# 05.1 — Basic Probability & Independence

[← Chapter index](../05-statistics.md) · [Next →](02-bayes.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## Basic probability & set operations 

| Concept | Formula |
|---|---|
| Complement | $P(A^c)=1-P(A)$ |
| Union | $P(A\cup B)=P(A)+P(B)-P(A\cap B)$ |
| Conditional | $P(A\mid B)=P(A\cap B)/P(B)$ |
| Multiplication | $P(A\cap B)=P(A\mid B)P(B)$ |
| At least one | $1-P(\text{none})$ |

Example: 40% buy A, 30% buy B, 10% buy both. P(A or B)=.6, P(A only)=.3, P(A|B)=1/3. Mutually exclusive means intersection=0；independent means intersection=P(A)P(B)。非零機率的互斥事件不獨立。

## Independence 

Independent events: $P(A\cap B)=P(A)P(B)$；等價於 $P(A\mid B)=P(A)$（分母非零）。Conditional independence: $P(A,B\mid C)=P(A\mid C)P(B\mid C)$，不代表 marginal independence。

Three independent attempts, each success=.2: none=.8³=.512, at least one=.488, exactly one=3(.2)(.8²)=.384。

> ⚠️ Without replacement 通常不獨立。若題目未給獨立，不可直接把 marginal probabilities 相乘。

## Worked variation — union versus conditional

40% buy A, 30% buy B, 10% buy both. P(A or B)=.4+.3−.1=.6; P(A only)=.3; P(neither)=.4; P(A|B)=.1/.3=1/3. Independence would require joint=.12, so not independent. Mutually exclusive would require joint=0.

**English words:** at least one = one or more; exactly one = one but not two; given that = condition on event; mutually exclusive ≠ independent.

<details><summary>Variation: Two events have P(A)=.4, P(B)=.5 and are mutually exclusive. Are they independent?</summary>

No. Joint=0 but product=.2. Nonzero mutually exclusive events cannot be independent.

</details>

<details><summary>Variation: Three independent systems fail with probabilities .1, .2, .3. What is the probability at least one fails?</summary>

1−(.9×.8×.7)=.496. Complement is easier than summing all overlapping failure events.

</details>

[← Chapter index](../05-statistics.md) · [Next →](02-bayes.md)
