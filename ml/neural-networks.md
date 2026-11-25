# 04.12 — Neural Network Fundamentals (P1)

[← ML index](../04-ml.md) · [Next →](regularization-generalization.md)

## 1. Forward pass

A neuron computes z = wᵀx+b and a = activation(z). Multiple layers compose transformations. Without nonlinear activation, a stack of linear layers is still an affine transformation, so depth alone does not create nonlinear decision boundaries.

Example x=[2,1], w=[.5,−1], b=.5 → z=.5 → ReLU=.5, sigmoid≈.622.

| Activation | Meaning / use |
|---|---|
| ReLU | max(0,z), common hidden layers |
| Sigmoid | maps to (0,1), binary probability output |
| Tanh | maps to (−1,1), can saturate |
| Softmax | multiclass scores → probabilities summing to 1 |

Typical output/loss combinations: continuous regression → linear output + MSE/MAE; binary → sigmoid + BCE; mutually exclusive multiclass → softmax + categorical cross-entropy. Multilabel classification can use independent sigmoids.

## 2. Training

Forward pass produces prediction and loss; backpropagation uses chain rule to compute gradients; optimizer updates weights. Backpropagation computes gradients, not itself the parameter update algorithm.

Vanishing gradients may occur when repeated derivatives are small; exploding gradients when very large. ReLU, suitable initialization, normalization, residual connections can help in appropriate architectures. Dropout randomly disables units during training, typically off at inference. Early stopping monitors validation. L2 weight decay limits complexity. Batch normalization behaves differently during training and inference.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
X,y=make_classification(n_samples=300,n_features=8,random_state=42)
Xtr,Xv,ytr,yv=train_test_split(X,y,stratify=y,test_size=.25,random_state=42)
m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),
    activation='relu',alpha=.001,early_stopping=True,max_iter=300,random_state=42))
m.fit(Xtr,ytr)
print(m.score(Xv,yv))
```

Neural nets can learn complex representations, but small tabular data may favor simpler models or boosted trees. CNN uses local filters, RNN handles sequences recurrently, Transformer uses attention (recognition-level scope only). Focus on fundamentals, not architecture internals.

## Concept variations

<details><summary>1. Why are nonlinear activations necessary in a multilayer network?</summary>

A composition of affine maps is affine. Nonlinear activation enables complex nonlinear functions.

</details>

<details><summary>2. Does backpropagation directly choose the learning rate?</summary>

No. It calculates gradients; the optimizer uses its configured/adaptive step rule.

</details>

<details><summary>3. Is softmax appropriate for a task where several labels may simultaneously be true?</summary>

Generally no. Softmax enforces mutually exclusive probabilities summing to one; independent sigmoid outputs are common for multilabel tasks.

</details>
