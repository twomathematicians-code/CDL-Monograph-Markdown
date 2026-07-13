# Volume I — Mathematical Foundations

*Volume I*

---

# Introduction to Volume I
Volume I develops the four pillars of mathematics on which the rest of the monograph rests: **linear algebra**, **optimization**, **probability**, and **information theory**. Each is presented at the depth required to read current research in deep learning theory, with explicit attention to the points where the mathematics surfaces in the algorithms and proofs of Volumes II and III.

## Why these four

A deep learning system is, mathematically, four things at once. It is a **parametric function class** — a family of maps $f_\theta : \mathcal{X} \to \mathcal{Y}$ indexed by parameters $\theta \in \mathbb{R}^P$. The geometry of this function class, and of the parameter space that indexes it, is linear-algebraic and differential-geometric. It is an **optimization problem** — find the parameter $\theta^\star$ that minimizes an empirical loss. The theory of how this optimization proceeds, why it converges, and what it converges *to* is the theory of stochastic gradient descent and its variants. It is a **probabilistic model** — the data are drawn from an unknown distribution, the network's predictions are interpreted as probabilities, and generalization is a statement about the gap between empirical and population risk. Finally, it is an **information-processing system** — the network transforms inputs through a sequence of representations, and the information-theoretic properties of these representations (mutual information with the input, with the label, with each other) are increasingly central to the theory of representation learning.

The four chapters of this volume treat these four aspects in turn. The treatment is *not* a survey: it is a development of the specific results, tools, and proof techniques that recur in the deep learning literature.

## Chapter map

**Chapter 1 — Linear Algebra for Neural Computation** treats the spectral theory of matrices, the SVD and low-rank approximation, randomized numerical linear algebra, the geometry of high-dimensional spaces (concentration of measure, near-orthogonality, the Johnson–Lindenstrauss lemma), tensors and tensor decompositions, and the structured linear operators (convolution, attention) that arise in neural network architectures.

**Chapter 2 — Optimization** treats convex optimization (gradient descent, accelerated methods, the convergence theory of [nesterov2018lectures]) and non-convex optimization (SGD, the Polyak–Łojasiewicz condition, the role of saddle points, the loss surface geometry of [choromanska2015loss]). The chapter closes with a treatment of the adaptive optimizers (AdaGrad, RMSProp, Adam, AdamW) that dominate modern practice, and a careful statement of the convergence results that are known — and the gaps that remain.

**Chapter 3 — Probability and Concentration** treats the basic inequalities (Markov, Chebyshev, Chernoff), sub-Gaussian and sub-exponential random variables, the Bernstein–Hoeffding bound, McDiarmid's inequality, and the uniform convergence bounds (VC dimension, Rademacher complexity) that underlie classical learning theory. The chapter pays particular attention to the regimes where classical bounds are *vacuous* for deep networks — the entry point for the modern generalization theory of Volume III.

**Chapter 4 — Information Theory** treats entropy, mutual information, the data processing inequality, the chain rules, the asymptotic equipartition property, and the information bottleneck principle. The chapter closes with the role of information-theoretic quantities in variational inference (ELBO, the evidence lower bound) and in the information bottleneck view of deep learning due to [tishby2015deep].

## How to read

The four chapters are largely independent and can be read in any order, with two caveats. Chapter 2 (Optimization) assumes the spectral theory of Chapter 1, specifically the operator norm and the SVD. Chapter 4 (Information Theory) assumes the basic probability of Chapter 3, specifically concentration inequalities. A reader interested only in the optimization theory of deep learning can read Chapters 1 and 2 and skip to Volume II. A reader interested only in the generalization theory should read Chapters 1, 3, and 4 before Volume III.

The computational experiments at the end of each chapter are designed to be runnable on a laptop in under ten minutes. The exercises range from routine (verifying definitions) to research-level (open questions suitable for a thesis). Difficulty is marked as Easy / Medium / Hard / Research.
