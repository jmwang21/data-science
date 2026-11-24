# 01.7 — DateTime Processing

[← pandas 目錄](../01-pandas.md) · [Previous](06-reshape-window.md) · [Next →](08-integrated.md)

## 1. Parse dates

```python
import pandas as pd
df = pd.DataFrame({
    'merchant':['A','B','C','D'],
    'order_date':['2025-01-01','2025-02-28 23:59:00',
                  '2025-03-01','invalid']
})
# Explicit mixed parsing is supported in pandas 2.0+
df['date'] = pd.to_datetime(df.order_date,format='mixed',errors='coerce')
```

invalid → NaT。若格式一致，可指定 format='%Y-%m-%d'；混合格式例子使用 pandas 2.0+ 的 format='mixed'。不要直接用字串月份做跨格式時間比較。

## 2. Date ranges

**English prompt:** Keep orders placed in January or February 2025.

```python
out = df.loc[(df.date>=pd.Timestamp('2025-01-01')) &
             (df.date<pd.Timestamp('2025-03-01'))]
```

保留 A、B。使用 inclusive start / exclusive end（左閉右開）可以涵蓋 2 月 28 日所有時刻，不需要寫 23:59:59。

```python
df.date.between('2025-01-01','2025-03-01',inclusive='left')
```

## 3. Date features and tenure

```python
df['month'] = df.date.dt.to_period('M')
df['weekday'] = df.date.dt.dayofweek # Monday=0
df['days_since'] = (pd.Timestamp('2025-03-01')-df.date).dt.days
```

| merchant | month | weekday | days_since |
|---|---|---:|---:|
| A | 2025-01 | 2 | 59 |
| B | 2025-02 | 4 | 0 |
| C | 2025-03 | 5 | 0 |
| D | NaT | NaN | NaN |

日期相減得到 Timedelta；.dt.days 為完整日數（注意負值的 floor 行為）。若題目問日曆日期差而非經過的完整 24 小時，先 normalize 日期。

商家 tenure 應在商家層級以 join_date 計算，reference date 依題目固定，避免直接使用今天日期造成結果不可重現。

## 4. Monthly aggregation

```python
df['amount']=[100,200,300,400]
monthly = df.groupby('month',as_index=False).amount.sum()
```

輸出 Jan=100、Feb=200、Mar=300；NaT group 預設不包含。resample 可用於 DatetimeIndex 的固定頻率彙總，但一般月彙總用 to_period + groupby 已足夠。多時區 timestamp 先統一 UTC，再轉 business timezone 後取日期（重要延伸）。

## Variation — time cutoff and calendar duration

**English:** Compute customer tenure as of the prediction cutoff rather than today's date.

```python
join_date = pd.Series(pd.to_datetime(['2025-01-01','2025-02-15']))
tenure = (pd.Timestamp('2025-03-01')-join_date).dt.days
# [59,14]
```

For timestamps across time zones, parse with utc=True and convert to the business zone before extracting day/month. A UTC timestamp on Feb 28 at 18:00 is already Mar 1 at 02:00 in UTC+8; the monthly bucket differs.

<details><summary>Variation: A timestamp is Feb 28 at 23:59. Does <= '2025-02-28' include it?</summary>

Usually no: the right-hand side parses to midnight at the start of Feb 28. Use a half-open interval ending at March 1.

</details>

<details><summary>Variation: Why might .dt.days return 0 for timestamps that fall on different calendar dates?</summary>

It counts whole elapsed days, not calendar boundaries. Normalize both dates first if the requirement is calendar-day difference.

</details>

## Concept check

**Why is date < '2025-03-01' preferable to date <= '2025-02-28' for filtering February timestamps?**

<details><summary>Answer</summary>

後者的日期字串通常代表 2 月 28 日午夜，會漏掉當天晚些時候的事件。前者保留整個二月，排除三月。

</details>
