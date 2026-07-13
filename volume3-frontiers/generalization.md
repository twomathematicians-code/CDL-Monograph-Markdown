# Generalization Theory

*Volume III · Chapter 13*

---

The central theoretical puzzle of deep learning is generalization: why do networks trained to interpolate their training data — networks with sufficient capacity to memorize random labels [zhang2021understanding] — generalize to unseen test data? Classical learning theory, based on uniform convergence and capacity measures (VC dimension, Rademacher complexity), is *vacuous* for overparameterized networks: it predicts a generalization gap of order $O(\sqrt{P/N})$, which is $\gg 1$ for modern networks. This chapter surveys the candidate explanations that have emerged since 2017, treating each with its strengths and limitations.

## The classical theory is vacuous

As reviewed in Chapter 3, classical uniform convergence gives bounds of the form

$$
\mathcal{L}(\theta) \leq \hat{\mathcal{L}}(\theta) + O\left(\sqrt{\frac{P \log N}{N}}\right),
$$

where $P$ is the parameter count and $N$ is the dataset size. For a modern network with $P = 10^9$ parameters trained on $N = 10^9$ examples, this bound is $\gg 1$ — useless.

Worse, [belkin2019reconciling] and [belkin2018understanding] showed that the classical bias-variance tradeoff (small models underfit, large models overfit) breaks down in the overparameterized regime: test error *decreases* beyond the interpolation threshold. This **double descent** phenomenon is inconsistent with the classical theory, which predicts monotone increase of test error with capacity in the overfitting regime.

## Implicit bias of SGD

The first modern candidate explanation is **implicit bias**: among all the parameter settings that fit the training data, SGD preferentially finds solutions with low complexity. The complexity measure depends on the architecture and optimizer:

- For **linear regression** in the overparameterized regime, gradient descent on the squared loss converges to the **minimum-norm** interpolating solution [zhang2021understanding]. This is the maximum-margin solution under appropriate conditions, and it generalizes well.
- For **linear classifiers** trained with logistic loss on separable data, gradient descent converges to the **maximum-margin** (L2-SVM) solution [soudry2018implicit].
- For **deep networks**, the situation is more complex: the implicit bias depends on the architecture, initialization, optimizer, and training duration. For wide networks, the bias is toward the **NTK regime** solution (Chapter 4); for narrower networks, the bias is toward flatter minima, sparser activations, or lower-rank weight matrices, depending on the setup.

The implicit bias program has been the most productive vein of modern generalization theory. It is partial — the bias is well-characterized only in restricted regimes — but it provides a clean conceptual framework.

## Flat minima and PAC-Bayes

An older line of work connects generalization to the **flatness** of the minimum: a minimum in a flat region of the loss surface generalizes better than a minimum in a sharp region, because small perturbations to the parameters do not change the loss. The intuition was formalized by [hochreiter1997flat] and given a PAC-Bayesian foundation by [dziugaite2017computing].

The **PAC-Bayes bound** states that, with high probability over the sample,

$$
\mathcal{L}(\theta) \leq \hat{\mathcal{L}}_Q(\theta) + \sqrt{\frac{D_{\mathrm{KL}}(Q \| P) + \log(2\sqrt{N}/\delta)}{N}},
$$

where $Q$ is a "posterior" over parameters (typically a Gaussian centered at the trained weights), $P$ is a "prior", and $\hat{\mathcal{L}}_Q$ is the empirical loss under $Q$. The bound is non-vacuous when the posterior is concentrated near the trained weights (small KL) and the empirical loss is low.

PAC-Bayes bounds have been made non-vacuous for small networks ([dziugaite2017computing]), but extending them to modern large networks remains an open problem. The principal obstacle is the KL term: for a network with $10^9$ parameters, even a Gaussian posterior with variance $10^{-6}$ has a KL of order $10^3$, which is too large.

## Compression-based bounds

A separate line of work bounds generalization via **compression**: if the trained network can be compressed to $k$ bits with no loss in training accuracy, then the generalization gap is $O(\sqrt{k/N})$, regardless of the original parameter count.

[arora2018stronger] proved compression bounds for specific architectures; [zhou2018compressing] and others have shown that trained networks can be heavily compressed (often by $10\times$–$100\times$) with minimal loss. The compression bound is appealing because it connects to the empirical practice of pruning and quantization. It is, however, hard to make tight: the achievable compression depends on the network and the training procedure, and the bound is only as good as the compression.

## The neural tangent kernel

For wide networks at initialization, the dynamics of gradient descent simplify dramatically. [jacot2018neural] showed that, in the infinite-width limit, the function $f_\theta(\mathbf{x})$ evolves under gradient descent according to a kernel regression with the **neural tangent kernel** (NTK):

$$
K(\mathbf{x}, \mathbf{y}) = \mathbb{E}_{\theta \sim \text{init}} \left[ \langle \nabla_\theta f_\theta(\mathbf{x}), \nabla_\theta f_\theta(\mathbf{y}) \rangle \right].
$$

The optimization in function space is convex, even though the optimization in parameter space is not. Generalization in the NTK regime can be analyzed with kernel-method theory, giving rates that match the empirical scaling laws in some regimes.

The NTK has substantial explanatory power in the infinite-width limit. Its limitations are equally substantial: real networks are not infinitely wide, and they do not stay in the NTK regime during training (the kernel changes substantially). The NTK is best understood as a *linearization* of the training dynamics around initialization; it captures the early-time dynamics and the wide-network limit, but not the feature-learning dynamics that distinguish deep learning from kernel methods.

## The lottery ticket hypothesis

[frankle2018lottery] proposed the **lottery ticket hypothesis**: a randomly initialized dense network contains a sparse subnetwork ("the winning ticket") that, when trained in isolation from the original initialization, matches the test accuracy of the dense network. The hypothesis has substantial empirical support — winning tickets can be pruned by $90\%$ or more without loss — and provides a natural connection to compression-based generalization bounds.

The lottery ticket hypothesis is an empirical observation, not a theorem. Theoretical follow-ups ([malach2020proving]) have shown that sufficiently overparameterized networks contain winning tickets with high probability, providing a partial theoretical foundation.

## The state of the theory

The honest state of deep learning generalization theory in 2025 is that **no single explanation is fully satisfactory**. The classical theory is vacuous. The implicit bias program is illuminating in restricted regimes but does not extend cleanly to realistic networks. PAC-Bayes is non-vacuous only for small networks. Compression bounds are appealing but hard to make tight. The NTK captures only the linearized dynamics. The lottery ticket hypothesis is empirical.

What the field has converged on is a **constellation** of mechanisms: implicit bias selects simple solutions among the many that fit; flatness and compression are correlated with generalization; wide networks are analyzable via the NTK; sparsity and pruning preserve performance. A unified theory that integrates these mechanisms is the principal open problem of the field (see Chapter 15).

## Computational experiments

### Experiment 1: Double descent

Train a network of varying width on a fixed dataset. Plot training and test error as a function of width. Verify the double descent phenomenon.

### Experiment 2: Random labels

Train a network on (a) true labels and (b) random labels. Compare training and test accuracy. Verify [zhang2021understanding]'s observation.

### Experiment 3: Lottery ticket

Train a network, prune by $90\%$, retrain the pruned subnetwork from (a) the original initialization and (b) a new initialization. Compare test accuracy.

## Historical notes

The classical learning theory is due to [vapnik1971uniform] (VC dimension) and [bartlett2002rademacher] (Rademacher complexity). The breakdown of classical theory in deep learning was emphasized by [zhang2021understanding]. Double descent was systematically studied by [belkin2019reconciling]. Implicit bias of gradient descent in linear models is due to [soudry2018implicit] and [ji2019implicit]; the deep-network case is treated in [neyshabur2014search]. PAC-Bayes bounds are due to [mcallester1999pac]; modern non-vacuous bounds to [dziugaite2017computing]. The NTK is due to [jacot2018neural]. The lottery ticket hypothesis is due to [frankle2018lottery].

## Exercises

1. **(Easy)** Compute the classical uniform-convergence bound for a ResNet-50 ($P \approx 25$M) trained on ImageNet ($N \approx 1.3$M). Verify that it is vacuous.

2. **(Easy)** State the lottery ticket hypothesis. What is the role of the original initialization?

3. **(Medium)** Reproduce the [zhang2021understanding] experiment on a small dataset: train a network on true labels and on random labels. Compare training and test accuracy.

4. **(Medium)** Reproduce the double descent phenomenon: train a network of varying width on a fixed dataset. Plot test error as a function of width.

5. **(Medium)** Reproduce the lottery ticket experiment: train a network, prune by $90\%$, retrain from the original initialization and from a fresh initialization. Compare.

6. **(Hard)** Implement the PAC-Bayes bound of [dziugaite2017computing] for a small network. Make the bound non-vacuous.

7. **(Hard)** Implement the NTK for a small MLP. Compare its predictions to the actual training dynamics.

8. **(Research)** Investigate the relationship between implicit bias and architecture. How does the implicit bias of a CNN differ from that of an MLP? From that of a transformer?

## Bibliography
