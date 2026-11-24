# 02.3 — Feature Engineering, Selection & Imbalance

[← Chapter index](../02-preprocessing.md) · [Previous](02-scaling-encoding.md) · [Next →](04-pipeline.md)

> Core review · 中文觀念解釋、English terminology、worked examples、variation questions。先理解，再展開答案。

## Feature Engineering 

```python
df['aov'] = df.gmv/df.order_count.replace(0,np.nan)
df['tenure_days'] = (pd.Timestamp('2025-03-01')-pd.to_datetime(df.join_date)).dt.days
df['active'] = (df.order_count>0).astype(int)
```

AOV=GMV/orders，無交易時是否填 0 由定義決定；repeat rate=回購客戶/全部客戶（先定義回購）；比例的總體值通常是分子總和/分母總和，不是各商家比例的簡單平均。所有特徵須符合預測時點，避免未來資訊。

## Practical feature construction (P0)

Numeric/categorical column roles should be decided by meaning rather than dtype alone: postal codes and IDs are often categorical; dates usually need extraction; targets must be excluded. For text, CountVectorizer creates token counts; TfidfVectorizer downweights common terms. Fit vocabulary/IDF only on train. Sparse matrices are efficient for one-hot and text. StandardScaler(with_mean=False) avoids centering sparse matrices.

For historical behavioral features, use an explicit as-of time and aggregate only past events. For a label defined in the next 30 days, no future transaction or post-outcome status belongs in X. A lag feature must be shifted within entity after sorting. Check that transformations produce compatible columns and order for unseen data.

## Feature selection and feature extraction

| Method | Mechanism | Example |
|---|---|---|
| Filter | statistics independent of final model | variance, correlation, mutual information |
| Wrapper | repeatedly evaluate subsets | RFE |
| Embedded | learned during model fitting | L1 regularization, tree importance |
| Extraction | construct new coordinates | PCA, TF-IDF |

Feature selection 若使用 y，必須在每個 CV training fold 執行。即使不使用 y，對全資料 fit scaler/PCA 仍可能洩漏 validation distribution。

### Worked example — supervised feature selection

```python
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
X,y=make_classification(n_samples=200,n_features=10,
                        n_informative=3,random_state=42)
pipe=make_pipeline(StandardScaler(),SelectKBest(f_classif,k=4),
                   LogisticRegression(max_iter=1000))
scores=cross_val_score(pipe,X,y,cv=5,scoring='roc_auc')
```

SelectKBest 以與 y 的統計關係評分，保留 k 個 feature。這段示例每 fold 都重新估計 feature scores。若先 fit 全資料再 CV，validation label 已參與選擇。PCA 不看 y，但仍應在 fold 內 fit。需注意 selection score 不等於特徵的因果效應。

## Imbalanced data

Imbalanced classification：class_weight 改變 loss；oversampling 重複 minority；undersampling 移除 majority；SMOTE 建立插值樣本。重抽樣放進 training folds，validation/test 維持原分布。調 threshold 是改 decision rule，不同於訓練重抽樣。

<details><summary>Variation: A fraud classifier has 99% accuracy in a dataset with 1% fraud. Is the model necessarily useful?</summary>

No. 全部預測 legitimate 即 99% accuracy、fraud recall=0。需看 precision/recall、AP、成本及 review capacity。

</details>

<details><summary>Variation: Can you compute average merchant GMV using transactions recorded after the prediction date?</summary>

No, if those transactions were unavailable at inference. 即使不直接包含 target，仍有 temporal leakage。

</details>

[← Chapter index](../02-preprocessing.md) · [Next →](04-pipeline.md)
