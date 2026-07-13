# Probability and Concentration

*Volume I · Chapter 3*

---

Probability enters deep learning in three distinct ways. As a **model**: a neural network's outputs are interpreted as parameters of a probability distribution over labels, and the loss is a (negative) log-likelihood. As a **tool**: stochastic gradient descent, mini-batching, dropout, and data augmentation are all probabilistic algorithms. As a **theoretical lens**: generalization is a statement about the gap between the empirical risk (a sample average) and the population risk (an expectation), and the size of this gap is controlled by concentration inequalities.

This chapter develops the probability required to read the modern generalization literature. The first three sections treat the basic inequalities (Markov, Chebyshev, Chernoff), sub-Gaussian and sub-exponential random variables, and McDiarmid's bounded-differences inequality. The fourth section treats uniform convergence — VC dimension and Rademacher complexity — which is the classical foundation of generalization theory. The fifth section explains why classical uniform convergence is *vacuous* for overparameterized deep networks, motivating the modern theory developed in Volume III.

## Basic inequalities

The most basic concentration inequality is **Markov's**, which requires only non-negativity.

#### **Theorem (Markov's inequality)**

For any non-negative random variable $X$ and $t > 0$,
$$
\Pr[X \geq t] \leq \frac{\mathbb{E}[X]}{t}.
$$

Markov is sharp in general but loose for specific distributions. The power of Markov is that it can be applied to powers or exponentials of $X$ to obtain tighter bounds — this is the route to Chernoff bounds.

#### **Theorem (Chebyshev's inequality)**

For any random variable $X$ with finite mean $\mu$ and variance $\sigma^2$,
$$
\Pr[|X - \mu| \geq t] \leq \frac{\sigma^2}{t^2}.
$$

Chebyshev is Markov applied to $(X-\mu)^2$. It already exhibits the key feature of concentration: deviations scale as $1/\sqrt{t}$ in probability, not as $1/t$. For sums of $n$ independent random variables, the variance grows linearly in $n$ while the mean grows linearly in $n$, so deviations are of order $\sqrt{n}$ — the central limit theorem regime.

## Sub-Gaussian and sub-exponential random variables

The right framework for sharp concentration is the **moment generating function**.

#### **Definition (Sub-Gaussian random variable)**

A random variable $X$ with mean $\mu = \mathbb{E}[X]$ is **sub-Gaussian** with parameter $\sigma$ if, for all $\lambda \in \mathbb{R}$,
$$
\mathbb{E}[\exp(\lambda (X - \mu))] \leq \exp(\sigma^2 \lambda^2 / 2).
$$

Gaussian random variables with variance $\sigma^2$ are sub-Gaussian with parameter $\sigma$; bounded random variables (e.g., Rademacher $\pm 1$) are sub-Gaussian with parameter depending on their range. The class of sub-Gaussian random variables is closed under summation and scaling, which makes it the right class for analyzing sums of independent random variables.

#### **Theorem (Hoeffding's inequality)**

Let $X_1, \ldots, X_n$ be independent, mean-zero, sub-Gaussian random variables with parameter $\sigma$. Then for any $t > 0$,
$$
\Pr\left[\left| \sum_{i=1}^n X_i \right| \geq t \right] \leq 2 \exp\left(-\frac{t^2}{2 n \sigma^2}\right).
$$

Hoeffding's inequality is the workhorse concentration result: the sum of $n$ independent sub-Gaussian random variables is sub-Gaussian with parameter $\sigma\sqrt{n}$. Equivalently, the *average* $\frac{1}{n}\sum X_i$ is sub-Gaussian with parameter $\sigma/\sqrt{n}$, giving the canonical $1/\sqrt{n}$ scaling of statistical fluctuations.

For random variables with heavier tails — products of sub-Gaussians, for example — the right class is **sub-exponential**.

#### **Definition (Sub-exponential random variable)**

A mean-zero random variable $X$ is **sub-exponential** with parameters $(\nu, b)$ if, for all $|\lambda| < 1/b$,
$$
\mathbb{E}[\exp(\lambda X)] \leq \exp(\nu^2 \lambda^2 / 2).
$$

The sum of $n$ independent sub-exponential random variables is sub-exponential with parameters $(\nu \sqrt{n}, b)$, and the corresponding concentration inequality (Bernstein's) has two regimes: Gaussian-like $\exp(-t^2 / (2 n \nu^2))$ for small $t$, and exponential $\exp(-t / (2b))$ for large $t$.

## McDiarmid's inequality

Many functions of interest — the loss of a trained network as a function of the training data, for example — are not sums of independent random variables. McDiarmid's inequality handles the more general case of **bounded-differences** functions.

#### **Theorem (McDiarmid's inequality)**

Let $X_1, \ldots, X_n$ be independent random variables taking values in a set $\mathcal{X}$, and let $f : \mathcal{X}^n \to \mathbb{R}$ satisfy the **bounded differences** condition: there exist $c_1, \ldots, c_n$ such that for all $i$ and all $x_1, \ldots, x_n, x_i'$,
$$
|f(x_1, \ldots, x_i, \ldots, x_n) - f(x_1, \ldots, x_i', \ldots, x_n)| \leq c_i.
$$
Then for any $t > 0$,
$$
\Pr\left[ |f(X_1, \ldots, X_n) - \mathbb{E}[f]| \geq t \right] \leq 2 \exp\left(-\frac{2 t^2}{\sum_i c_i^2}\right).
$$

McDiarmid's inequality is the master tool for proving concentration of functions of independent random variables. Hoeffding's inequality is the special case $f(x_1, \ldots, x_n) = \sum_i x_i$. The empirical process inequalities of the next section are derived by applying McDiarmid to suprema of empirical processes.

## Uniform convergence and classical learning theory

The generalization problem is to bound the gap between empirical risk $\hat{\mathcal{L}}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(\theta; \mathbf{z}_i)$ and population risk $\mathcal{L}(\theta) = \mathbb{E}[\ell(\theta; \mathbf{Z})]$. For a *fixed* $\theta$, this gap is controlled by Hoeffding: $|\hat{\mathcal{L}}(\theta) - \mathcal{L}(\theta)| = O(\sqrt{1/N})$ with high probability. The challenge is that $\theta$ is *not* fixed — it is chosen by minimizing $\hat{\mathcal{L}}$ — and the bound must hold uniformly over the function class $\mathcal{F} = \{f_\theta : \theta \in \Theta\}$.

### VC dimension and Rademacher complexity

Two complexity measures of function classes dominate the classical theory.

#### **Definition (VC dimension)**

The **Vapnik–Chervonenkis (VC) dimension** of a function class $\mathcal{F}$ of binary classifiers is the largest $d$ such that there exists a set of $d$ points $\{\mathbf{x}_1, \ldots, \mathbf{x}_d\}$ that $\mathcal{F}$ can shatter — i.e., realize every labeling.

#### **Definition (Rademacher complexity)**

The **empirical Rademacher complexity** of a function class $\mathcal{F}$ on a sample $S = (\mathbf{z}_1, \ldots, \mathbf{z}_N)$ is
$$
\hat{\mathfrak{R}}_S(\mathcal{F}) = \mathbb{E}_\sigma \left[ \sup_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \sigma_i f(\mathbf{z}_i) \right],
$$
where $\sigma_i$ are independent Rademacher ($\pm 1$) random variables. The (population) Rademacher complexity is $\mathfrak{R}_N(\mathcal{F}) = \mathbb{E}_S[\hat{\mathfrak{R}}_S(\mathcal{F})]$.

#### **Theorem (Rademacher generalization bound)**

With probability $1 - \delta$ over the sample $S$ of size $N$, for all $f \in \mathcal{F}$,
$$
\mathcal{L}(f) \leq \hat{\mathcal{L}}(f) + 2 \mathfrak{R}_N(\mathcal{F}) + \sqrt{\frac{2 \log(2/\delta)}{N}}.
$$

The Rademacher complexity is a data-dependent complexity measure — it depends on the distribution of $\mathbf{Z}$, not just on the function class. For neural networks with $P$ parameters and Lipschitz activations, the Rademacher complexity is $O(\sqrt{P/N})$ in the worst case, leading to the classical wisdom that one needs $N \gg P$ samples to generalize.

### Why classical bounds are vacuous for deep networks

Modern deep networks have $P \sim 10^9$ parameters and are trained on $N \sim 10^9$ examples; classical bounds based on parameter count give a generalization gap of order $O(1)$, which is useless. Worse, [zhang2021understanding] showed that networks can fit *randomly labeled* data perfectly — they have enough capacity to memorize the training set — yet generalize well on real data. The function class is the same in both cases; only the data differs. Classical uniform convergence cannot distinguish the two.

This is the entry point for the modern theory of generalization, which proceeds by exploiting the *implicit bias* of SGD (the algorithm selects specific minima from the many available), the *structure* of real data (which makes memorization unnecessary), and the *compressibility* of trained networks (which gives a smaller effective complexity than the parameter count suggests). We develop these ideas in Volume III, Chapter 4.

## Computational experiments

### Experiment 1: Hoeffding vs. Chebyshev

Sample $n$ Bernoulli(1/2) random variables for various $n$. Plot the empirical tail probability $\Pr[|\bar{X}_n - 1/2| \geq t]$ against the bounds given by Chebyshev and Hoeffding. Verify that Hoeffding is exponentially tighter.

### Experiment 2: Rademacher complexity of a linear class

Compute the empirical Rademacher complexity of the class $f_w(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ with $\|\mathbf{w}\| \leq B$ on a synthetic Gaussian dataset. Verify the theoretical bound $\hat{\mathfrak{R}}_S \leq B \|\bar{\mathbf{x}}\|/\sqrt{N}$.

## Historical notes

The basic concentration inequalities (Markov, Chebyshev, Chernoff) are 19th- and mid-20th-century results, with Chernoff's bound appearing in [chernoff1952measure]. Hoeffding's inequality is due to [hoeffding1963probability]; the extension to sub-Gaussian random variables is developed systematically in [boucheron2013concentration]. McDiarmid's inequality is due to [mcdiarmid1989method].

VC dimension was introduced by [vapnik1971uniform]. Rademacher complexity has a longer history in empirical process theory; its introduction to learning theory is due to [koltchinskii2001rademacher] and [bartlett2002rademacher]. The recognition that classical uniform convergence is vacuous for overparameterized networks is due to [zhang2021understanding], which launched the modern theory of generalization in deep learning.

## Exercises

1. **(Easy)** Prove that a bounded random variable $X \in [a, b]$ is sub-Gaussian with parameter $(b-a)/2$.

2. **(Easy)** Show that a sum of independent sub-Gaussian random variables is sub-Gaussian, with parameter equal to the $\ell_2$ norm of the individual parameters.

3. **(Medium)** Derive Hoeffding's inequality from the sub-Gaussian definition. (Hint: apply Markov to $\exp(\lambda \sum X_i)$ and optimize over $\lambda$.)

4. **(Medium)** Compute the VC dimension of half-spaces in $\mathbb{R}^2$. Compute the VC dimension of the class of axis-aligned rectangles in $\mathbb{R}^d$.

5. **(Medium)** For a fixed function $f$ with $\ell(f; \mathbf{z}) \in [0, 1]$, derive a high-probability bound on $|\hat{\mathcal{L}}(f) - \mathcal{L}(f)|$ as a function of $N$ and the confidence $\delta$.

6. **(Hard)** Prove the Rademacher generalization bound. (Hint: symmetrization, then McDiarmid.)

7. **(Hard)** Show that the Rademacher complexity of a class of neural networks with $P$ parameters, $L$ layers, and Lipschitz activations grows at most polynomially in $P$ under suitable normalization. Compare this to the naive $O(\sqrt{P/N})$ bound.

8. **(Research)** Reproduce a variant of the [zhang2021understanding] experiment: train a small CNN on (a) CIFAR-10 with true labels and (b) CIFAR-10 with random labels. Report training and test accuracy in both cases. Discuss what this implies about the relevance of uniform convergence bounds to deep learning.

## Bibliography
