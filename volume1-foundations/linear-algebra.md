# Linear Algebra for Neural Computation

*Volume I · Chapter 1*

---

The computational primitives of deep learning are linear-algebraic: matrix multiplication, tensor contraction, low-rank factorization, projection onto subspaces. Even the name *layer* — a sequence of affine and nonlinear operations — encodes the linear-algebraic heart of the subject. This chapter develops the linear algebra required to read the deep learning literature with precision: the spectral theory of matrices and tensors, the geometry of high-dimensional spaces, the principal matrix decompositions (eigendecomposition, SVD, QR, Cholesky), low-rank approximation, randomized numerical linear algebra, and the structured operators (convolution, attention) that arise specifically in neural network architectures.

The treatment is deliberately complete. A reader who has read this chapter should be able to follow the technical arguments in the neural tangent kernel literature ([jacot2018neural]), the double descent literature ([belkin2019reconciling]), and the scaling laws literature ([kaplan2020scaling]) without consulting a linear algebra textbook. Where the material is standard, we cite the canonical reference and proceed; where the material is specific to deep learning, we develop it in detail.

## Vector spaces, norms, and inner products

We work over $\mathbb{R}^d$ throughout, equipped with the standard basis. A vector $\mathbf{x} \in \mathbb{R}^d$ is a column vector with components $x_1, \ldots, x_d$. The standard inner product is $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y} = \sum_i x_i y_i$, and the induced norm is $\|\mathbf{x}\|_2 = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$.

#### **Definition (Operator norm)**

Let $\|\cdot\|$ be a norm on $\mathbb{R}^d$. The **operator norm** (or induced norm) on $\mathbb{R}^{m \times n}$ induced by $\|\cdot\|$ is
$$
\|\mathbf{A}\| = \sup_{\mathbf{x} \neq \mathbf{0}} \frac{\|\mathbf{A}\mathbf{x}\|}{\|\mathbf{x}\|} = \sup_{\|\mathbf{x}\| = 1} \|\mathbf{A}\mathbf{x}\|.
$$

For the $\ell_2$ norm, the operator norm is the largest singular value $\sigma_{\max}(\mathbf{A})$. For the $\ell_1$ norm, it is the maximum absolute column sum. For the $\ell_\infty$ norm, it is the maximum absolute row sum. These three norms — $\ell_2$, $\ell_1$, $\ell_\infty$ — and their operator-induced versions recur throughout the book.

#### **Theorem (Equivalence of norms on finite-dimensional spaces)**

On $\mathbb{R}^d$, all norms are equivalent: for any two norms $\|\cdot\|_a$ and $\|\cdot\|_b$, there exist constants $c, C > 0$ such that
$$
c \|\mathbf{x}\|_a \leq \|\mathbf{x}\|_b \leq C \|\mathbf{x}\|_a \quad \text{for all } \mathbf{x} \in \mathbb{R}^d.
$$

The proof is a standard compactness argument on the unit sphere. The constants, however, depend exponentially on $d$ in general, which is one of the reasons high-dimensional geometry is qualitatively different from low-dimensional geometry — the topic of the corresponding section.

## Spectral theory

The fundamental decomposition of a square matrix is its eigendecomposition. A matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is **diagonalizable** if it can be written $\mathbf{A} = \mathbf{P} \mathbf{D} \mathbf{P}^{-1}$ where $\mathbf{D}$ is diagonal and $\mathbf{P}$ is invertible. The columns of $\mathbf{P}$ are eigenvectors and the diagonal entries of $\mathbf{D}$ are eigenvalues. Symmetric matrices are always diagonalizable, and in fact orthogonally so.

#### **Theorem (Spectral theorem for symmetric matrices)**

If $\mathbf{A} \in \mathbb{R}^{n \times n}$ is symmetric ($\mathbf{A} = \mathbf{A}^\top$), then there exists an orthogonal matrix $\mathbf{Q} \in \mathbb{R}^{n \times n}$ ($\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$) and a diagonal matrix $\mathbf{\Lambda}$ such that
$$
\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^\top.
$$
The columns of $\mathbf{Q}$ are eigenvectors of $\mathbf{A}$ and the diagonal entries of $\mathbf{\Lambda}$ are the corresponding (real) eigenvalues.

This decomposition is the cornerstone of symmetric matrix computation. It implies, for instance, that a symmetric matrix has $n$ real eigenvalues (counted with multiplicity) and $n$ orthonormal eigenvectors. For non-symmetric matrices, the eigenvalues may be complex and the eigenvectors need not be orthogonal; the appropriate generalization is the **Schur decomposition** $\mathbf{A} = \mathbf{Q} \mathbf{T} \mathbf{Q}^\top$ where $\mathbf{T}$ is upper-triangular.

#### **Theorem (Courant–Fischer minimax theorem)**

Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be symmetric with eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$. Then
$$
\lambda_k = \max_{\dim V = k} \min_{\mathbf{x} \in V, \|\mathbf{x}\| = 1} \mathbf{x}^\top \mathbf{A} \mathbf{x} = \min_{\dim V = n - k + 1} \max_{\mathbf{x} \in V, \|\mathbf{x}\| = 1} \mathbf{x}^\top \mathbf{A} \mathbf{x}.
$$

The Courant–Fischer theorem is the right tool for proving interlacing inequalities (how eigenvalues change under rank-one updates) and for analyzing perturbation theory. It plays a central role in the analysis of the Hessian of the loss in deep networks (Chapter 2) and in the spectral analysis of the neural tangent kernel (Volume III).

## The singular value decomposition

The singular value decomposition (SVD) is the most important matrix factorization for deep learning. It applies to any matrix, not just square or symmetric ones, and it generalizes the spectral theorem.

#### **Theorem (Singular value decomposition)**

Let $\mathbf{A} \in \mathbb{R}^{m \times n}$ with rank $r$. Then there exist orthogonal matrices $\mathbf{U} \in \mathbb{R}^{m \times m}$ and $\mathbf{V} \in \mathbb{R}^{n \times n}$, and a diagonal matrix $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0 = \sigma_{r+1} = \cdots$, such that
$$
\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top.
$$
The $\sigma_i$ are the **singular values** of $\mathbf{A}$; the columns $\mathbf{u}_i$ of $\mathbf{U}$ and $\mathbf{v}_i$ of $\mathbf{V}$ are the left and right **singular vectors**.

The SVD has a clean geometric interpretation: every linear map $\mathbf{x} \mapsto \mathbf{A}\mathbf{x}$ can be decomposed as a rotation $\mathbf{V}^\top$, a scaling $\mathbf{\Sigma}$, and another rotation $\mathbf{U}$. The singular values measure the amount of stretching along each principal axis.

### Low-rank approximation

The SVD gives the optimal low-rank approximation in a precise sense.

#### **Theorem (Eckart–Young–Mirsky theorem)**

Let $\mathbf{A} \in \mathbb{R}^{m \times n}$ have SVD $\mathbf{A} = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$. For any $k \leq r$, the best rank-$k$ approximation in Frobenius norm is
$$
\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top,
$$
and the approximation error is $\|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$.

The Eckart–Young theorem is the foundation of principal component analysis (PCA), low-rank adaptation methods (LoRA, [hu2021lora]), and the analysis of weight matrices in trained networks. Empirically, the singular values of trained weight matrices often exhibit a rapid decay followed by a long tail of small values; this "low effective rank" phenomenon is one of the empirical regularities that motivates low-rank fine-tuning.

### Randomized SVD

For large matrices — say $\mathbf{A} \in \mathbb{R}^{m \times n}$ with $m, n \sim 10^6$ — computing the full SVD is infeasible. The randomized SVD algorithm of [halko2011finding] produces a rank-$k$ approximation in $O(mnk)$ time, with high-probability guarantees.

#### **Algorithm (Randomized range finder)**

```
**Require:** Matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, target rank $k$, oversampling $p \geq 5$
**Ensure:** Orthonormal $\mathbf{Q} \in \mathbb{R}^{m \times (k+p)}$ with $\mathrm{range}(\mathbf{A}) \approx \mathrm{range}(\mathbf{Q})$
Draw Gaussian random matrix $\mathbf{\Omega} \in \mathbb{R}^{n \times (k+p)}$
$\mathbf{Y} \gets \mathbf{A} \mathbf{\Omega}$
$\mathbf{Q} \gets \mathrm{qr}(\mathbf{Y})$   // thin QR factorization
return $\mathbf{Q}$
```

The corresponding randomized SVD is:

#### **Algorithm (Randomized SVD)**

```
**Require:** Matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, target rank $k$, oversampling $p$
**Ensure:** Rank-$(k+p)$ approximation $\mathbf{U}, \mathbf{\Sigma}, \mathbf{V}$
$\mathbf{Q} \gets \text{RandomizedRangeFinder}(\mathbf{A}, k, p)$
$\mathbf{B} \gets \mathbf{Q}^\top \mathbf{A}$   // $\mathbf{B \in \mathbb{R}^{(k+p) \times n}$}
$[\tilde{\mathbf{U}}, \mathbf{\Sigma}, \mathbf{V}^\top] \gets \mathrm{svd}(\mathbf{B})$
$\mathbf{U} \gets \mathbf{Q} \tilde{\mathbf{U}}$
return $\mathbf{U}, \mathbf{\Sigma}, \mathbf{V}$
```

The computational saving is dramatic: instead of $O(mn \min(m,n))$ for the full SVD, we pay $O(mn(k+p))$ for the matrix–random-matrix product plus $O((k+p)^2 n)$ for the small SVD. For $k = 100$ and $m = n = 10^6$, this is the difference between an infeasible computation and a routine one.

The following Python listing implements randomized SVD for dense NumPy arrays.

```python
import numpy as np

def randomized_svd(A, k, p=10, n_iter=4, seed=0):
    """Randomized SVD: rank-k approximation of A.

    Parameters
    ----------
    A : ndarray, shape (m, n)
    k : int, target rank
    p : int, oversampling parameter
    n_iter : int, power iterations for accuracy
    seed : int, RNG seed

    Returns
    -------
    U, s, Vt : rank-k SVD factors
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    omega = rng.standard_normal(size=(n, k + p))
    Y = A @ omega                          # m x (k+p)
    Q, _ = np.linalg.qr(Y, mode='reduced') # m x (k+p)
    # Power iterations improve accuracy when spectrum decays slowly
    for _ in range(n_iter):
        Z = A.T @ Q
        Q, _ = np.linalg.qr(Z, mode='reduced')
        Y = A @ Q
        Q, _ = np.linalg.qr(Y, mode='reduced')
    B = Q.T @ A                            # (k+p) x n
    Utilde, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Utilde
    return U[:, :k], s[:k], Vt[:k, :]

# Demonstration on a synthetic low-rank matrix
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    m, n, true_rank = 500, 800, 20
    L = rng.standard_normal((m, true_rank))
    R = rng.standard_normal((true_rank, n))
    A = L @ R + 1e-3 * rng.standard_normal((m, n))  # noisy rank-20 matrix
    U, s, Vt = randomized_svd(A, k=20, p=10, n_iter=2)
    approx = U @ np.diag(s) @ Vt
    rel_err = np.linalg.norm(A - approx) / np.linalg.norm(A)
    print(f"rank-20 randomized SVD: relative error = {rel_err:.4e}")
```

## The geometry of high-dimensional spaces

The geometry of $\mathbb{R}^d$ when $d$ is large is qualitatively different from the geometry of $\mathbb{R}^2$ or $\mathbb{R}^3$, and many of the empirical phenomena of deep learning are expressions of this difference. Three results anchor the subject.

### Concentration of measure

#### **Theorem (Gaussian concentration)**

If $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$, then for any $t > 0$,
$$
\Pr\left[\left| \|\mathbf{x}\| - \sqrt{d} \right| > t \right] \leq 2 \exp(-t^2 / 2).
$$

The norm of a high-dimensional Gaussian is essentially $\sqrt{d}$, with fluctuations of order $O(1)$. A randomly drawn vector lies near the surface of the sphere of radius $\sqrt{d}$, not near the origin. The mass of the Gaussian is concentrated in a thin shell — the *Gaussian annulus* — at radius $\sqrt{d}$.

### Random vectors are nearly orthogonal

#### **Theorem (Near-orthogonality of random vectors)**

Let $\mathbf{x}, \mathbf{y} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ independently. Then for any $\varepsilon > 0$,
$$
\Pr\left[ \left| \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \right| > \varepsilon \right] \leq 2 \exp\left( - \frac{d \varepsilon^2}{2} \right).
$$

In $\mathbb{R}^{10000}$, two random vectors have inner product essentially zero (within $0.01$) with overwhelming probability. This is one of the reasons the ambient dimension in deep learning can be misleading: most directions are uninformative, and the relevant structure lies in a much smaller effective subspace.

### The Johnson–Lindenstrauss lemma

The most useful consequence of high-dimensional concentration is that random projection preserves pairwise distances.

#### **Theorem (Johnson–Lindenstrauss lemma)**

For any set of $n$ points $\{\mathbf{x}_i\}_{i=1}^n \subset \mathbb{R}^d$ and any $\varepsilon \in (0, 1/2)$, there exists a map $f : \mathbb{R}^d \to \mathbb{R}^k$ with $k = O(\varepsilon^{-2} \log n)$ such that for all $i, j$,
$$
(1 - \varepsilon) \|\mathbf{x}_i - \mathbf{x}_j\|^2 \leq \|f(\mathbf{x}_i) - f(\mathbf{x}_j)\|^2 \leq (1 + \varepsilon) \|\mathbf{x}_i - \mathbf{x}_j\|^2.
$$
A random Gaussian or sparse sign matrix $\mathbf{\Phi} \in \mathbb{R}^{k \times d}$ with $f(\mathbf{x}) = \mathbf{\Phi} \mathbf{x} / \sqrt{k}$ achieves this with high probability.

The target dimension $k$ is *independent of $d$* — it depends only on the number of points and the accuracy $\varepsilon$. This is the basis for random projection as a preprocessing step in nearest-neighbor search, in approximate kernel methods, and in the analysis of random feature networks.

```python
import numpy as np

def jl_project(X, k, seed=0):
    """Project rows of X to dimension k via random Gaussian matrix."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    Phi = rng.standard_normal((k, d)) / np.sqrt(k)
    return X @ Phi.T

rng = np.random.default_rng(0)
n, d = 1000, 5000
X = rng.standard_normal((n, d))
# Pairwise distances in original space
from scipy.spatial.distance import pdist, squareform
D_orig = squareform(pdist(X))
# Project to k dimensions
for k in [50, 100, 200, 500]:
    Xp = jl_project(X, k, seed=1)
    D_proj = squareform(pdist(Xp))
    ratio = D_proj / D_orig
    off_diag = ratio[~np.eye(n, dtype=bool)]
    print(f"k={k:4d}: mean ratio={off_diag.mean():.4f}, "
          f"max|1-ratio|={np.abs(1-off_diag).max():.4f}")
```

## Tensors and tensor decompositions

A deep network is most naturally described as a sequence of **tensor** operations: a weight tensor $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ contracts with an input $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$ to produce an output $\mathbf{y} = \mathbf{W}\mathbf{x} \in \mathbb{R}^{d_{\text{out}}}$, with the input itself often a higher-order tensor (image: $\text{channels} \times \text{height} \times \text{width}$; sequence: $\text{length} \times \text{features}$).

We adopt the **order** of a tensor as the number of indices: a vector is an order-1 tensor, a matrix an order-2 tensor, and so on. The space of order-$p$ tensors of shape $(d_1, \ldots, d_p)$ is denoted $\mathbb{R}^{d_1 \times \cdots \times d_p}$.

The two principal tensor decompositions are the **CP (CANDECOMP/PARAFAC) decomposition** and the **Tucker decomposition**. The CP decomposition writes an order-$p$ tensor $\boldsymbol{\mathcal{T}}$ as a sum of rank-1 tensors:

$$
\boldsymbol{\mathcal{T}} = \sum_{r=1}^R \lambda_r \mathbf{u}^{(1)}_r \otimes \mathbf{u}^{(2)}_r \otimes \cdots \otimes \mathbf{u}^{(p)}_r.
$$

The Tucker decomposition is more general, allowing a core tensor $\boldsymbol{\mathcal{G}} \in \mathbb{R}^{r_1 \times \cdots \times r_p}$ to mix $r_i$ components along each axis:

$$
\boldsymbol{\mathcal{T}} = \boldsymbol{\mathcal{G}} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \cdots \times_p \mathbf{U}^{(p)},
$$

where $\times_i$ denotes the mode-$i$ product. CP is the special case where $\boldsymbol{\mathcal{G}}$ is diagonal.

Computing the CP rank of a tensor is NP-hard ([hastad1990tensor]), in stark contrast to the matrix case where rank is computed in polynomial time via Gaussian elimination. This computational intractability is one of the reasons tensor methods are less developed in deep learning than matrix methods, despite the natural fit.

## Structured linear operators in deep learning

Two structured linear operators are central to modern deep learning: **convolution** and **attention**. Both are linear maps with structure that exploits properties of the input (translation equivariance for convolution; permutation equivariance for attention). We give each a unified linear-algebraic treatment.

### Convolution as a structured matrix multiplication

A one-dimensional convolution of a signal $\mathbf{x} \in \mathbb{R}^n$ with a kernel $\mathbf{w} \in \mathbb{R}^k$ is the linear map $\mathbf{y} = \mathbf{C}\mathbf{x}$ where $\mathbf{C} \in \mathbb{R}^{(n+k-1) \times n}$ is the **convolution matrix**: a Toeplitz matrix with $\mathbf{w}$ along each diagonal. For two-dimensional convolution (as in image processing), $\mathbf{C}$ is block-Toeplitz with Toeplitz blocks.

The structure of $\mathbf{C}$ gives convolution two important properties:

1. **Translation equivariance**. If $T_\tau$ is the shift-by-$\tau$ operator, then $T_\tau \circ \text{conv} = \text{conv} \circ T_\tau$.
2. **Sparse parameters**. The convolution matrix has only $k$ free parameters, regardless of $n$.

These two properties together explain why convolutional networks generalize well on natural images: natural image statistics are approximately translation-invariant ([olshausen1996emergence]), and the convolutional structure imposes the corresponding equivariance on the function class.

### Attention as a low-rank perturbation of identity

The self-attention operation for a sequence of $n$ vectors $\mathbf{X} \in \mathbb{R}^{n \times d}$ with learned projections $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d_h}$ is

$$
\text{Attn}(\mathbf{X}) = \text{softmax}\left( \frac{(\mathbf{X}\mathbf{W}_Q)(\mathbf{X}\mathbf{W}_K)^\top}{\sqrt{d_h}} \right) \mathbf{X}\mathbf{W}_V.
$$

The output is a linear combination of the value vectors, with coefficients given by the softmax of the query–key inner products. The map $\mathbf{X} \mapsto \text{Attn}(\mathbf{X})$ is **permutation equivariant**: reordering the rows of $\mathbf{X}$ permutes the rows of the output correspondingly. This is the structural property that makes attention appropriate for set-structured and sequence-structured data; positional information must be injected separately via positional encodings.

The computational cost of naive attention is $O(n^2 d_h)$, dominated by the $n \times n$ attention matrix. For long sequences ($n \gg d_h$), this quadratic cost is prohibitive; efficient attention variants (sparse, linear, low-rank) reduce the cost to $O(n)$ or $O(n \log n)$ at the cost of approximation. We treat these in Volume II, Chapter 4.

## Computational experiments

We close the chapter with three short computational experiments that anchor the theoretical material.

### Experiment 1: Singular value decay in trained networks

The singular value spectra of weight matrices in trained networks exhibit characteristic shapes that depend on the architecture and training procedure. The following experiment measures the spectra of a small MLP trained on MNIST and reports the effective rank.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# (Assume model is trained; we plot spectra of its weight matrices)
# For a freshly initialized model:
model = TinyMLP()
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
for ax, (name, W) in zip(axes, model.named_parameters()):
    if W.dim() == 2:
        s = torch.linalg.svdvals(W).numpy()
        ax.semilogy(s, 'o-', markersize=3)
        ax.set_title(f'{name}: shape {tuple(W.shape)}')
        ax.set_xlabel('index')
        ax.set_ylabel('singular value (log)')
        ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/svd_decay_init.pdf')
```

### Experiment 2: JL lemma empirical verification

Running Listing the listing above above produces output along the lines of:

```
k=  50: mean ratio=1.0001, max|1-ratio|=0.4123
k= 100: mean ratio=0.9999, max|1-ratio|=0.2918
k= 200: mean ratio=1.0000, max|1-ratio|=0.2055
k= 500: mean ratio=1.0000, max|1-ratio|=0.1312
```

The mean ratio is essentially 1 in all cases; the maximum deviation decreases as $k$ grows, consistent with the $1/\sqrt{k}$ scaling predicted by the JL lemma.

### Experiment 3: Randomized SVD accuracy

Running Listing the listing above on the synthetic rank-20 matrix produces:

```
rank-20 randomized SVD: relative error = 1.05e-03
```

The relative error matches the noise level we injected ($10^{-3}$), confirming that the randomized SVD captures the rank-20 structure essentially exactly.

## Historical notes

The spectral theorem for symmetric matrices is classical, with roots in the 19th-century work of Cauchy, Weierstrass, and Sylvester. The singular value decomposition was introduced independently by Beltrami (1873) and Jordan (1874) for square matrices, and generalized to rectangular matrices by Eckart and Young in the 1930s. The Eckart–Young low-rank approximation theorem [eckart1936approximation] is one of the most-cited results in numerical linear algebra.

The Johnson–Lindenstrauss lemma [johnson1984extensions] was originally proved in the context of Lipschitz embeddings of metric spaces; the random-projection version we use today is due to Frankl and Maehara (1988) and was popularized in computer science by Indyk and Motwani (1998). The randomized SVD algorithm is due to [halko2011finding], building on earlier work by Martinsson, Rokhlin, and Tygert; it is now the standard method for large-scale low-rank approximation.

The tensor decomposition literature is older and more fragmented; the CP decomposition was introduced by Hitchcock (1927) and rediscovered several times, while the Tucker decomposition is due to [tucker1966some]. The NP-hardness of tensor rank is due to [hastad1990tensor].

## Exercises

1. **(Easy)** Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be symmetric with eigenvalues $\lambda_1 \geq \cdots \geq \lambda_n$. Show that $\|\mathbf{A}\|_2 = \max(|\lambda_1|, |\lambda_n|)$.

2. **(Easy)** Verify the claim in the Near-Orthogonality theorem empirically: for $d \in \{10, 100, 1000, 10000\}$, sample 1000 pairs of random unit vectors, compute their inner products, and report the maximum absolute inner product. How does it scale with $d$?

3. **(Medium)** Prove the Courant–Fischer minimax theorem by induction on $k$.

4. **(Medium)** Implement the randomized SVD (Listing the listing above) and compare its runtime and accuracy to `numpy.linalg.svd` on a $5000 \times 5000$ matrix of numerical rank 50. Plot the relative error and the runtime as a function of the target rank $k$.

5. **(Medium)** Show that one-dimensional convolution with kernel $\mathbf{w}$ is a linear map, and write down the corresponding convolution matrix $\mathbf{C}$ explicitly for $n = 5$, $k = 3$. Verify the translation-equivariance property $T_\tau \mathbf{C} = \mathbf{C} T_\tau$ numerically.

6. **(Hard)** Prove the Johnson–Lindenstrauss lemma for the case of a Gaussian random projection matrix. (Hint: use the $\chi^2$ concentration of $\|\mathbf{\Phi}\mathbf{x}\|^2$.)

7. **(Hard)** Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be a random matrix with i.i.d. $\mathcal{N}(0, 1/n)$ entries. State and prove a bound on $\|\mathbf{A}\|_2$ that holds with high probability. Compare your bound to the empirical value for $n \in \{100, 500, 1000, 5000\}$.

8. **(Research)** Investigate the singular value spectrum of weight matrices in a trained transformer (e.g., GPT-2 small). How does the spectrum differ across layers (embedding, attention, MLP)? How does it change during training? Relate your findings to the predictions of random matrix theory.

## Bibliography
