# Multilayer Perceptrons

*Volume II · Chapter 5*

---

The multilayer perceptron (MLP) — also called a feedforward network — is the simplest deep architecture. It is a composition of affine maps and elementwise nonlinearities:

$$
f_\theta(\mathbf{x}) = (\mathbf{W}_L \sigma(\cdot) + \mathbf{b}_L) \circ \cdots \circ (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1),
$$

where $\sigma$ is a fixed nonlinearity (ReLU, tanh, GELU, etc.) applied elementwise. Despite its simplicity, the MLP is the right starting point: it is the architecture in which the central questions of deep learning — expressivity, trainability, generalization — can be posed most cleanly, and the answers obtained for MLPs often transfer (with modification) to more complex architectures.

## Universal approximation

The foundational theoretical result is that MLPs with a single hidden layer can approximate any continuous function on a compact set, given enough width.

#### **Theorem (Cybenko–Hornik universal approximation theorem)**

Let $\sigma : \mathbb{R} \to \mathbb{R}$ be a continuous, non-constant, bounded activation function (a "sigmoidal" activation). Let $K \subset \mathbb{R}^d$ be compact. Then for any continuous $f : K \to \mathbb{R}$ and any $\varepsilon > 0$, there exists an integer $n$ and parameters $\{\alpha_i, \mathbf{w}_i, b_i\}_{i=1}^n$ such that the single-hidden-layer network
$$
g(\mathbf{x}) = \sum_{i=1}^n \alpha_i \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)
$$
satisfies $\sup_{\mathbf{x} \in K} |g(\mathbf{x}) - f(\mathbf{x})| < \varepsilon$.

The theorem was proved independently by [cybenko1989approximation] (for sigmoidal activations) and [hornik1989multilayer] (for general non-polynomial activations). The proof is non-constructive: it shows *existence* of suitable parameters, not how to find them by gradient descent. The gap between existence and learnability is the central theoretical puzzle of deep learning.

### Depth vs. width

A striking refinement shows that **depth can be exponentially more efficient than width** for representing certain functions. [telgarsky2016benefits] constructed a family of functions on $[0,1]$ that can be represented by a deep ReLU network of size $O(k^2)$ but require width $\Omega(2^k)$ for any shallow network with the same accuracy. The intuition is that each layer of ReLU composes the function with itself, allowing deep networks to express oscillatory functions exponentially more compactly.

#### **Theorem (Telgarsky's depth separation)**

For each $k$, there exists a function $f_k : [0,1] \to \{0,1\}$ that can be computed by a ReLU network of depth $k$ and $O(k^2)$ parameters, but any ReLU network of depth $k' < k$ computing $f_k$ to accuracy $1/2 - \varepsilon$ requires width $\Omega(2^{k - k'})$.

This is a worst-case separation; whether depth provides similar advantages for the functions that arise in practice is an empirical question. The empirical record — ResNets with hundreds of layers outperform shallow networks on ImageNet, deep transformers outperform shallow ones on language tasks — suggests that it does, for natural data.

## Activation functions

The choice of activation function is consequential. We summarize the principal options.

| Activation | Formula | Properties |
|:----------:|:--------|:-----------|
| ReLU | $\max(0, x)$ | Non-saturating for $x > 0$; sparse activations; "dead neuron" problem for $x < 0$ |
| Leaky ReLU | $\max(\alpha x, x)$, $\alpha \ll 1$ | Addresses dead neurons; sparse activations lost |
| ELU | $x$ if $x > 0$, $\alpha(e^x - 1)$ else | Smooth; zero-mean outputs; saturating for $x \to -\infty$ |
| GELU | $x \Phi(x)$, $\Phi$ = Gaussian CDF | Smooth; used in transformers; non-saturating |
| SwiGLU | $\text{Swish}(xW_1) \otimes (x W_2)$ | Gated; used in modern transformers |
| Tanh | $\tanh(x)$ | Bounded, smooth; saturating; classical |

The Rectified Linear Unit (ReLU) [nair2010rectified] was the dominant activation in the early deep learning era; it addresses the vanishing gradient problem of saturating activations like sigmoid and tanh. GELU [hendrycks2016gaussian] has become the default in transformer architectures since BERT, partly for empirical reasons (better performance) and partly for theoretical reasons (a smoother profile that better matches the statistics of natural data).

## Initialization

The choice of initialization is one of the most consequential design decisions for training deep networks. Poor initialization leads to vanishing or exploding activations and gradients, especially in deep networks.

#### **Definition (Xavier/Glorot initialization)**

For a layer with $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs, initialize weights as $W_{ij} \sim \mathcal{N}(0, 2/(d_{\text{in}} + d_{\text{out}}))$ or uniform on $[-\sqrt{6/(d_{\text{in}}+d_{\text{out}})}, \sqrt{6/(d_{\text{in}}+d_{\text{out}})}]$.

Xavier initialization [glorot2010understanding] is derived by requiring that the variance of activations and gradients is preserved across layers, under the assumption of linear activations. For ReLU activations, which zero out half of the inputs, the appropriate scaling is doubled:

#### **Definition (He/Kaiming initialization)**

For a layer with $d_{\text{in}}$ inputs followed by a ReLU, initialize weights as $W_{ij} \sim \mathcal{N}(0, 2/d_{\text{in}})$.

He initialization [he2015delving] is the standard for ReLU-based networks. For other activations, the appropriate scaling can be derived analytically or tuned empirically.

## Normalization

Even with careful initialization, the statistics of activations can drift during training, especially in deep networks. Normalization layers stabilize training by re-centering and re-scaling activations.

#### **Definition (Batch normalization)**

For a mini-batch $B = \{\mathbf{x}_1, \ldots, \mathbf{x}_m\}$, batch normalization computes
$$
\hat{\mathbf{x}}_i = \gamma \odot \frac{\mathbf{x}_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}} + \beta,
$$
where $\mu_B, \sigma_B^2$ are the batch mean and variance, $\gamma, \beta$ are learned, and $\varepsilon$ is for numerical stability.

Batch normalization [ioffe2015batch] was a key enabler of very deep networks; it allows higher learning rates, reduces sensitivity to initialization, and provides a mild regularizing effect (the batch statistics are noisy). Its dependence on batch size is a drawback for small batches and for distributed training; **layer normalization** [ba2016layer], which normalizes over features rather than batch elements, avoids this and is the default in transformers.

## Dropout

Dropout [srivastava2014dropout]; [hinton2012improving] is a regularization technique that randomly zeroes a fraction $p$ of activations during training, then scales by $1/(1-p)$ at test time. The effect is to approximate training an ensemble of subnetworks (one per dropout mask) and averaging their predictions. Dropout is theoretically motivated as preventing *co-adaptation* of features: each feature must be useful on its own, not in combination with specific other features.

## A complete MLP in PyTorch

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dims, activation=nn.GELU, p_drop=0.1):
        """dims: list like [784, 256, 64, 10]"""
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # no norm/act on final layer
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(activation())
                layers.append(nn.Dropout(p_drop))
        self.net = nn.Sequential(*layers)
        self._init_weights()
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x.flatten(1))

model = MLP([784, 256, 64, 10])
```

## Computational experiments

### Experiment 1: Universal approximation on 1D

Train a single-hidden-layer ReLU MLP to approximate a 1D function (e.g., $\sin(2\pi x) + 0.3 \sin(8\pi x)$) to high accuracy. Plot the learned function and the network's approximation as a function of the number of hidden units.

### Experiment 2: Depth vs. width on a classification task

On MNIST, compare a shallow network (1 hidden layer of width 1024) to a deep network (4 hidden layers of width 256). Both have approximately the same parameter count. Plot training and test accuracy as a function of training epoch.

### Experiment 3: Effect of initialization

Train the MLP from Listing the listing above on MNIST with three initializations: (a) all weights = 0.01, (b) Xavier, (c) He. Plot the loss curves and the activation statistics per layer.

## Historical notes

The perceptron, the single-layer predecessor of the MLP, was introduced by [rosenblatt1958perceptron]. The limitations of single-layer perceptrons (their inability to learn XOR) were emphasized by [minsky2017perceptron], leading to the "AI winter" of the 1970s. The backpropagation algorithm, which made training multi-layer networks tractable, was popularized by [rumelhart1986learning], though it had been described earlier by Werbos (1974) and others.

Universal approximation was proved independently by [cybenko1989approximation] and [hornik1989multilayer]. The depth-separation results are more recent; [telgarsky2016benefits] gave the first clean separation in the ReLU case. ReLU was introduced to deep learning by [nair2010rectified], building on earlier work in computational neuroscience. Xavier initialization is due to [glorot2010understanding]; He initialization to [he2015delving]. Batch normalization is due to [ioffe2015batch]; layer normalization to [ba2016layer]. Dropout is due to [srivastava2014dropout] and [hinton2012improving].

## Exercises

1. **(Easy)** Show that a single-hidden-layer network with ReLU activations is a piecewise-linear function of the input. How many "pieces" can a network with $n$ hidden units have?

2. **(Easy)** Derive the gradient of the loss with respect to $\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2$ for a two-layer MLP with squared loss.

3. **(Medium)** Prove the universal approximation theorem for sigmoidal activations. (Hint: use the density of step functions in $L^1$ and the ability of sigmoids to approximate steps.)

4. **(Medium)** Implement Xavier and He initialization. Train a 10-layer MLP on MNIST with each, plus a "bad" initialization (e.g., all weights = 1). Report the activation and gradient norms per layer.

5. **(Medium)** Implement batch normalization as a custom layer. Compare training curves with and without batch norm on a deep MLP (10+ layers).

6. **(Hard)** Reproduce a depth-separation experiment: construct a function that is easy for a deep network to represent and hard for a shallow one. Train both and compare sample efficiency.

7. **(Hard)** Show that dropout with rate $p$ is equivalent, in expectation, to training an ensemble of $2^n$ subnetworks (where $n$ is the number of dropout units) and averaging. Why is the explicit ensemble infeasible?

8. **(Research)** Investigate the role of activation function choice on the neural tangent kernel. How does the NTK change when the activation is ReLU vs. GELU vs. tanh? What are the implications for training dynamics?

## Bibliography
