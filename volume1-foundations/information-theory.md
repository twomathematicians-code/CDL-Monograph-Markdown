# Information Theory

*Volume I · Chapter 4*

---

Information theory enters deep learning in three ways. As a **modeling tool**: variational autoencoders, the evidence lower bound, and the information bottleneck principle are all explicitly information-theoretic. As a **training objective**: cross-entropy, KL divergence, and mutual-information maximization are standard losses. As a **theoretical lens**: the representations learned by deep networks are increasingly analyzed in terms of their information content — mutual information with the input, with the label, with intermediate representations. This chapter develops the information theory required to read this literature.

## Entropy and mutual information

#### **Definition (Shannon entropy)**

For a discrete random variable $X$ with probability mass function $p(x)$, the **Shannon entropy** is
$$
H(X) = -\sum_x p(x) \log p(x),
$$
with the convention $0 \log 0 = 0$. The unit is bits when $\log$ is base 2, nats when $\log$ is natural.

Entropy measures the *uncertainty* in $X$. A fair coin has $H = 1$ bit; a biased coin with $p = 0.01$ has $H \approx 0.08$ bits. The maximum entropy for a discrete random variable on $k$ values is $\log k$, attained by the uniform distribution.

For continuous random variables, the analog is **differential entropy**, $h(X) = -\int p(x) \log p(x) \, dx$. Differential entropy can be negative (it is for a uniform on $[0, 0.1]$, which has $h = -\log 10 < 0$), and lacks the clean operational meaning of discrete entropy. It is nonetheless useful as a calculational tool.

#### **Definition (KL divergence)**

The **Kullback–Leibler (KL) divergence** from $P$ to $Q$ is
$$
D_{\mathrm{KL}}(P \| Q) = \mathbb{E}_P \left[ \log \frac{p(X)}{q(X)} \right] = \int p(x) \log \frac{p(x)}{q(x)} \, dx.
$$

KL divergence is non-negative (Gibbs' inequality) and zero iff $P = Q$ almost everywhere. It is *not* symmetric — $D_{\mathrm{KL}}(P\|Q) \neq D_{\mathrm{KL}}(Q\|P)$ in general — and does not satisfy the triangle inequality. It is, however, the canonical measure of dissimilarity between distributions in machine learning, both because of its statistical interpretation (it governs the asymptotic rate of distinguishability) and its computational convenience.

#### **Definition (Mutual information)**

The **mutual information** between two random variables $X$ and $Y$ is
$$
I(X; Y) = D_{\mathrm{KL}}(P_{X,Y} \| P_X \otimes P_Y) = \mathbb{E}_{X,Y} \left[ \log \frac{p(X, Y)}{p(X) p(Y)} \right].
$$

Mutual information is symmetric ($I(X; Y) = I(Y; X)$), non-negative, and zero iff $X$ and $Y$ are independent. It measures the *information* $Y$ carries about $X$ (or vice versa). For Gaussian $(X, Y)$ with correlation $\rho$, $I(X; Y) = -\frac{1}{2} \log(1 - \rho^2)$.

## The data processing inequality

#### **Theorem (Data processing inequality)**

If $X \to Y \to Z$ is a Markov chain (i.e., $Z$ is conditionally independent of $X$ given $Y$), then
$$
I(X; Z) \leq I(X; Y).
$$

The data processing inequality (DPI) formalizes the intuition that *post-processing cannot create information*: if $Z$ is a (possibly randomized) function of $Y$ alone, and $Y$ is the only channel through which information about $X$ flows, then $Z$ cannot contain more information about $X$ than $Y$ does.

The DPI has a sharp consequence for deep networks: if we view the network's layers as a Markov chain $X \to h_1 \to h_2 \to \cdots \to h_L \to \hat{Y}$, then $I(X; h_\ell)$ is *non-increasing* in $\ell$. The intermediate representations cannot contain more information about the input than earlier layers do — a constraint that, combined with the goal of preserving information about the label $Y$, gives the information bottleneck principle.

## The information bottleneck principle

[tishby2015deep] and [tishby1999information] proposed that deep learning can be understood as a process of *compressing* the input $X$ while *preserving* information about the label $Y$. The objective is

$$
\min_{p(h | x)} I(X; H) - \beta \, I(H; Y),
$$

where $\beta > 0$ is a trade-off parameter. The first term pushes the representation $H$ to compress $X$; the second pushes it to retain information about $Y$. The Lagrangian reveals that the optimal representation is a *minimal sufficient statistic* for $Y$ from $X$.

The empirical claim — controversial but influential — is that the layers of a trained network traverse a trajectory in the **information plane** $(I(X; h_\ell), I(h_\ell; Y))$, with an early *fitting* phase in which both quantities grow, and a later *compression* phase in which $I(X; h_\ell)$ decreases while $I(h_\ell; Y)$ stays roughly constant. The compression phase has been argued to underlie generalization. The empirical status of this claim is the subject of ongoing debate; see Volume III, Chapter 4.

## Variational inference and the ELBO

Many probabilistic models in deep learning — variational autoencoders [kingma2013auto], deep latent Gaussian models — involve an intractable posterior $p(\mathbf{z} | \mathbf{x})$. Variational inference approximates the posterior by a tractable distribution $q_\phi(\mathbf{z} | \mathbf{x})$ by maximizing the **evidence lower bound (ELBO)**:

$$
\log p(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} [\log p(\mathbf{x} | \mathbf{z})] - D_{\mathrm{KL}}(q_\phi(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z})).
$$

The ELBO is derived by applying Jensen's inequality to $\log p(\mathbf{x}) = \log \int p(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}$. The first term is the *reconstruction* term — it rewards the model for explaining the data — and the second is the *regularization* term — it penalizes the approximate posterior for straying from the prior. The two terms are in tension, controlled by the choice of prior and the capacity of $q_\phi$.

The reparameterization trick [kingma2013auto] allows the ELBO to be optimized by stochastic gradient descent: write $\mathbf{z} = g_\phi(\mathbf{x}, \boldsymbol{\epsilon})$ for a deterministic function $g_\phi$ and a noise variable $\boldsymbol{\epsilon}$ with a fixed distribution, then differentiate through $g_\phi$. This turns an expectation over $q_\phi$ into an expectation over a fixed distribution, which can be estimated by Monte Carlo and differentiated.

## Cross-entropy and softmax

The most common loss for classification is **cross-entropy**:

$$
\ell_{\text{CE}}(\theta; \mathbf{x}, y) = -\log p_\theta(y | \mathbf{x}) = -\log \frac{\exp(f_y(\mathbf{x}; \theta))}{\sum_{y'} \exp(f_{y'}(\mathbf{x}; \theta))},
$$

where $f_y(\mathbf{x}; \theta)$ is the logit for class $y$. The cross-entropy is the negative log-likelihood under a softmax distribution; equivalently, it is the KL divergence (up to a constant) between the true label distribution (a one-hot) and the predicted distribution. The gradient with respect to the logits has the clean form $\partial \ell / \partial f_y = p_\theta(y|\mathbf{x}) - \mathbf{1}[y = y']$.

The **temperature-scaled softmax** $p(y|\mathbf{x}) = \exp(f_y / T) / \sum_{y'} \exp(f_{y'} / T)$ interpolates between a uniform distribution (large $T$) and an argmax (small $T$). The temperature plays a central role in knowledge distillation: a smaller student network is trained to match the temperature-scaled outputs of a larger teacher, transferring not just the hard label but the teacher's *soft* distribution over classes.

## Computational experiments

### Experiment 1: Mutual information estimation

Estimating mutual information from samples is hard in general; the naive plug-in estimator is biased and high-variance. The following code implements the **MINE** estimator (Mutual Information Neural Estimation, [belghazi2018mine]), which uses a neural network to lower-bound $I(X; Y)$ via the Donsker–Varadhan representation.

```python
import torch
import torch.nn as nn

class MINE(nn.Module):
    """Mutual Information Neural Estimator.
    Lower-bounds I(X;Y) via Donsker-Varadhan:
        I(X;Y) >= E_{P_XY}[T(x,y)] - log E_{P_X P_Y}[exp(T(x,y))]
    """
    def __init__(self, dim_x, dim_y, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x + dim_y, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x, y):
        # joint samples
        t_joint = self.net(torch.cat([x, y], dim=1)).mean()
        # marginal samples: shuffle y to break dependence
        y_marg = y[torch.randperm(y.shape[0])]
        t_marg = torch.logsumexp(
            self.net(torch.cat([x, y_marg], dim=1)).squeeze(), dim=0
        ) - torch.log(torch.tensor(float(x.shape[0])))
        return t_joint - t_marg  # lower bound on I(X;Y)
```

### Experiment 2: ELBO and reconstruction

Implement a small VAE on MNIST. Plot the reconstruction term, the KL term, and the ELBO as a function of training epoch. Observe that the KL term grows quickly early in training (the posterior learns to differ from the prior) and then plateaus, while the reconstruction term continues to decrease.

## Historical notes

Information theory was founded by [shannon1948mathematical], who introduced entropy, mutual information, and the basic coding theorems. The data processing inequality appears in its modern form in [cover1999elements]. The Kullback–Leibler divergence is due to [kullback1951information].

The information bottleneck principle was introduced by [tishby1999information] and applied to deep learning in [tishby2015deep]. The empirical study of information flow in deep networks has been controversial; [saxe2018on] showed that the compression phase depends critically on the choice of activation function (saturating vs. non-saturating) and on the mutual information estimator. The current consensus is that the IB principle is a useful *perspective* on representation learning but not a complete *theory*.

Variational inference has a long history in Bayesian statistics; the modern formulation in terms of the ELBO is due to [jordan1999introduction]. The variational autoencoder was introduced in [kingma2013auto] and [rezende2014stochastic]; the reparameterization trick is the key technical innovation that makes the ELBO differentiable. Knowledge distillation is due to [hinton2015distilling].

## Exercises

1. **(Easy)** Show that $D_{\mathrm{KL}}(P\|Q) \geq 0$ with equality iff $P = Q$. (Hint: use Jensen's inequality on $-\log$.)

2. **(Easy)** Compute the entropy of a Bernoulli($p$) random variable. Plot it as a function of $p$ and verify that the maximum is at $p = 1/2$.

3. **(Medium)** Prove the data processing inequality. (Hint: use the chain rule for mutual information and the conditional independence.)

4. **(Medium)** For a bivariate Gaussian $(X, Y)$ with correlation $\rho$, compute $I(X; Y)$ in closed form. Verify numerically that the MINE estimator recovers this value to within estimator error.

5. **(Medium)** Derive the ELBO from Jensen's inequality. Show that the gap $\log p(\mathbf{x}) - \text{ELBO}$ is exactly $D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x}))$.

6. **(Hard)** Implement a small VAE on MNIST. Train it and plot reconstructions and samples from the prior. Vary the dimension of the latent space and observe the trade-off between reconstruction quality and the KL term.

7. **(Hard)** Implement the information bottleneck objective for a small classifier on a synthetic dataset with a known relevant subspace. Verify that the learned representation concentrates information about the label and discards information about the rest of the input.

8. **(Research)** Reproduce the [tishby2015deep] experiment on the information plane trajectory of a small network. Use the MINE estimator (or a comparable one) and plot $I(X; h_\ell)$ and $I(h_\ell; Y)$ as a function of training epoch. Does compression occur? Does it depend on the activation function?

## Bibliography
