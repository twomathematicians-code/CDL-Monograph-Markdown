# Computational Deep Learning

---

# Welcome
This monograph is a self-contained, three-volume treatment of **computational deep learning** — the mathematical, algorithmic, and empirical study of neural networks trained by gradient-based optimization on large datasets. The aim is to provide a researcher-grade reference that moves fluidly between rigorous theory, implementable algorithms, reproducible computational experiments, and a candid map of the open problems that define the field's frontier.

The work is organized around three convictions that have shaped the field since [lecun2015deep]:

1. **Deep learning is a computational discipline.** Theorems matter, but so does the cost of matrix multiplications, the structure of GPU memory hierarchies, the curvature of the loss surface, and the empirical scaling behavior of training loss. Theory and computation are inseparable.
2. **Architectures are best understood as constraints on the function class, not as recipes.** Convolution, attention, recurrence, and residual connections are each a different inductive bias. A monograph should clarify what each buys and what each costs.
3. **The frontier moves quickly, but the questions are stable.** Specific architectures, optimizers, and benchmarks rise and fall; the underlying questions about generalization, optimization, representation, and interpretability persist. A monograph should orient the reader toward the questions.

## What you will find

The three volumes proceed from foundations to frontier:

- **Volume I — Mathematical Foundations**. Linear algebra for neural computation, convex and non-convex optimization, probability and concentration, and information theory. Each chapter ends with explicit notes on how the material surfaces in deep learning.
- **Volume II — Algorithms and Architectures**. Multilayer perceptrons, convolutional networks, recurrent networks, the transformer family, generative models (VAEs, GANs, diffusion), and the practice of training (initialization, normalization, optimization, regularization).
- **Volume III — Frontiers and Open Problems**. Representation learning and the geometry of feature spaces, scaling laws and compute-optimal training, the theory of generalization in overparameterized regimes, interpretability and mechanistic understanding, and a chapter devoted to explicit open problems.

Every chapter contains: formal definitions and theorems with proofs where the proof is instructive; pseudocode algorithm boxes; runnable Python listings (companion notebooks in the `notebooks/` directory); computational experiments with explicit numerical results; exercises graded by difficulty; historical notes; and a bibliography.

## How to read this monograph

The intended reader has the mathematical maturity of a beginning PhD student in computer science, statistics, physics, or applied mathematics, and is comfortable with the basics of probability, linear algebra, and multivariate calculus at the level of [strang2016introduction] and [bishop2006pattern]. Familiarity with Python is assumed for the computational sections; familiarity with PyTorch is helpful but not required.

Volumes are designed to be read in any order, although Volume I is a prerequisite for the proofs in Volumes II and III. The computational notebooks can be read alongside the chapters or run independently; each is self-contained.

## Conventions

We use the following notation throughout:

| Symbol | Meaning |
|:------:|:--------|
| $\mathbb{R}^d$ | The $d$-dimensional real vector space |
| $\mathbf{x} \in \mathbb{R}^d$ | A column vector; components $x_i$ |
| $\mathbf{X} \in \mathbb{R}^{m \times n}$ | A real matrix; entries $X_{ij}$ |
| $\mathbf{A}^\top, \mathbf{A}^{-1}$ | Transpose and inverse |
| $\|\mathbf{x}\|_p$ | The $\ell_p$ norm; $\|\mathbf{x}\|$ defaults to $\ell_2$ |
| $\nabla f, \nabla^2 f$ | Gradient and Hessian of $f$ |
| $\mathbb{E}[X], \mathrm{Var}(X)$ | Expectation and variance of a random variable |
| $\mathcal{N}(\mu, \Sigma)$ | Gaussian distribution with mean $\mu$, covariance $\Sigma$ |
| $D_{\mathrm{KL}}(P \| Q)$ | Kullback–Leibler divergence |
| $H(X)$ | Shannon entropy |
| $f_\theta$ | A parametric function with parameters $\theta$ |
| $\mathcal{L}(\theta)$ | A loss function; $\hat{\mathcal{L}}$ is the empirical loss |

Throughout, theorems, lemmas, propositions, and corollaries are numbered consecutively within each chapter. Algorithms, definitions, examples, and remarks share the same counter. Exercises are numbered separately at the end of each chapter.

## Acknowledgments

This monograph draws on a vast body of work by the deep learning community. References to the primary literature are given at the end of each chapter and in the consolidated bibliography. We are grateful to the many colleagues whose questions, corrections, and discussions have shaped the presentation.

— **scitamehtam research group** · Mahesh Solanki, 2025
