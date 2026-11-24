# 08.1 — Python & NumPy Core

[← Chapter index](../08-numpy-data.md)

## Python / NumPy 核心複習（P0）

| Concept | 常用語法 | 重點 |
|---|---|---|
| List / tuple | [1,2], (1,2) | tuple 不可變；list 可變 |
| Dict / set | {k:v for k,v in pairs}, set(values) | mapping、去重、membership |
| Sorting | sorted(rows,key=lambda r:r[1],reverse=True) | key 是排序依據 |
| Iteration | zip(ids,values), enumerate(values) | zip 預設截到最短 |
| Function | def f(x,threshold=0): return x>threshold | 避免可變 default arguments |
| Array creation | np.arange(6).reshape(3,2) | reshape 元素總數不變 |
| Indexing | a[:,0], a[[0,2]], a[a>0] | slice 通常 view；advanced indexing 是 copy |
| Aggregation | a.sum(axis=0), a.mean(axis=1) | axis=0 沿列消去；axis=1 沿欄消去 |
| Ranking | np.argsort(a), np.argmax(a) | index 而非 value |
| Matrix | X @ w, X * w | matrix product vs elementwise |
| Random | np.random.default_rng(42) | 可重現的 generator |

```python
import numpy as np
a=np.arange(6).reshape(3,2)
# [[0,1],[2,3],[4,5]]
a.sum(axis=0)  # [6,9]
a.sum(axis=1)  # [1,5,9]
a - a.mean(axis=0)  # column centering
a - a.mean(axis=1,keepdims=True)  # row centering
a @ np.array([2,3])  # [3,13,23]
np.unique(a,return_counts=True)
np.quantile(a,[.25,.5,.75])
```



## NumPy essentials

| Operation | Syntax | Notes |
|---|---|---|
| Array | `np.array([1,2,3])` | homogeneous dtype |
| Shape | `a.shape`, `a.ndim` | inspect dimensions |
| Select column | `X[:, 1]` | all rows, second column |
| Boolean mask | `a[a>0]` | array selection |
| Conditional | `np.where(a>0,1,0)` | elementwise |
| Mean | `X.mean(axis=0)` | one mean per column |
| Reshape | `a.reshape(-1,1)` | column vector |
| Combine | `np.concatenate([a,b],axis=0)` | compatible other dimensions |
| Missing | `np.isnan(a)`, `np.nanmean(a)` | floating numeric arrays |
| Infinity | `np.isfinite(a)` | excludes NaN and ±inf |
| Sort index | `np.argsort(a)` | positions, not values |

```python
import numpy as np
X=np.array([[1.,10.],[3.,20.],[5.,30.]])
print(X.mean(axis=0))  # [3.,20.]
print(X.mean(axis=1))  # [5.5,11.5,17.5]
Z=(X-X.mean(axis=0))/X.std(axis=0)
print(Z.shape)  # (3,2)
```

Broadcasting aligns trailing dimensions. `(3,2)-(2,)` works, but `(3,2)-(3,)` fails. For row means use `keepdims=True` to preserve shape `(3,1)`. For ML, fit the scaler only on training data, preferably in a Pipeline. A constant column has zero standard deviation; manual division needs handling.

```python
num=np.array([100.,0.,50.]); den=np.array([0.,0.,10.])
ratio=np.divide(num,den,out=np.full(num.shape,np.nan),where=den!=0)
# [nan, nan, 5.]
```

`np.where(den!=0,num/den,np.nan)` can still evaluate the division and emit a warning; `np.divide(..., where=...)` avoids evaluating those divisions.

**Practice:** `P=np.array([[.8,.2],[.3,.7],[.4,.6]])`. Obtain class 1 probabilities and predicted labels at threshold .65.

<details><summary>Answer</summary>

`p=P[:,1]` gives `[.2,.7,.6]`; `(p>=.65).astype(int)` gives `[0,1,0]`. Assumes columns are ordered class 0 then class 1.

</details>

## Variation — row vs column operations

```python
import numpy as np
X=np.array([[1.,10.],[3.,20.],[5.,30.]])
print(X.mean(axis=0)) # [3.,20.]
print(X.mean(axis=1)) # [5.5,11.5,17.5]
print(X-X.mean(axis=1,keepdims=True))
# [[-4.5,4.5],[-8.5,8.5],[-12.5,12.5]]
```

<details><summary>Variation: P is a 100×2 probability matrix. What is the shape of P[:,1] and P[:,[1]]?</summary>

(100,) and (100,1). The first is a 1-D array suitable for binary scores; the second preserves a column dimension. Confirm class order before selecting the positive class.

</details>

<details><summary>Variation: A feature has zero variance. What happens if you standardize by manually dividing by its SD?</summary>

Division by zero gives invalid values; handle constant features explicitly or use StandardScaler, which handles zero variance. Do not fit scale statistics on validation.

</details>

[Next → Multi-file integration](02-multi-file.md)
