# Optimization

*Volume I · Chapter 2*

---

The training of a deep network is, mathematically, an instance of **stochastic optimization**: minimize a population risk $\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{z}}[\ell(\theta; \mathbf{z})]$ given only samples $\mathbf{z}_1, \mathbf{z}_2, \ldots$ from the data distribution, by way of the empirical risk $\hat{\mathcal{L}}(\theta) = \frac{1}{N}\sum_{i=1}^N \ell(\theta; \mathbf{z}_i)$. The function $\mathcal{L}$ is non-convex, high-dimensional (modern networks have $10^9$–$10^{12}$ parameters), and accessed only through stochastic gradient estimates. That this optimization is, in practice, tractable — and that the resulting minima generalize — is one of the central empirical facts about deep learning, and one of the central theoretical puzzles.

This chapter develops the optimization theory in two passes. The first pass treats **convex optimization**, where the theory is essentially complete and the convergence rates are tight. The second pass treats **non-convex optimization**, where the theory is partial, the relevant questions are different, and the empirical record is the final arbiter.

## Convex optimization: the complete theory

A function $f : \mathbb{R}^d \to \mathbb{R}$ is **convex** if for all $\mathbf{x}, \mathbf{y}$ and $\lambda \in [0,1]$,
$$
f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y}).
$$
If $f$ is differentiable, this is equivalent to $f(\mathbf{y}) \geq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$ for all $\mathbf{x}, \mathbf{y}$. If $f$ is twice differentiable, it is equivalent to $\nabla^2 f(\mathbf{x}) \succeq 0$.

The crucial property of convex functions is that **every local minimum is a global minimum**. Optimization is, in the convex case, simply the problem of finding *a* minimum; there is no risk of getting stuck.

### Gradient descent

Gradient descent with step size $\eta$ on a differentiable function $f$ iterates $\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$.

#### **Theorem (Gradient descent on smooth convex functions)**

Let $f : \mathbb{R}^d \to \mathbb{R}$ be convex and $L$-smooth (i.e., $\nabla f$ is $L$-Lipschitz). Then gradient descent with $\eta = 1/L$ satisfies
$$
f(\theta_T) - f(\theta^\star) \leq \frac{L \|\theta_0 - \theta^\star\|^2}{2 T}.
$$

The proof is a one-line inequality: by smoothness, $f(\theta_{t+1}) \leq f(\theta_t) + \langle \nabla f(\theta_t), \theta_{t+1} - \theta_t \rangle + \frac{L}{2}\|\theta_{t+1} - \theta_t\|^2$; substituting $\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$ and simplifying yields the descent lemma $f(\theta_{t+1}) \leq f(\theta_t) - \frac{1}{2L}\|\nabla f(\theta_t)\|^2$. Summing and using convexity gives the result.

If $f$ is additionally $\mu$-strongly convex (i.e., $f - \frac{\mu}{2}\|\cdot\|^2$ is convex), the rate improves to linear convergence: $f(\theta_T) - f(\theta^\star) \leq (1 - \mu/L)^T \cdot \frac{L \|\theta_0 - \theta^\star\|^2}{2}$. The condition number $\kappa = L/\mu$ controls the rate; for ill-conditioned problems ($\kappa \gg 1$), convergence is slow.

### Accelerated methods

The rate $O(1/T)$ for smooth convex optimization is **not optimal**. Nesterov's accelerated gradient descent [nesterov1983method] achieves the optimal rate $O(1/T^2)$.

#### **Algorithm (Nesterov's accelerated gradient)**

```
**Require:** Initial $\theta_0$, smoothness $L$, iteration count $T$
$\mathbf{y}_0 \gets \theta_0$, $\gamma_0 \gets 0$
for $t = 0, 1, \ldots, T-1$ do
  $\gamma_{t+1} \gets \frac{1 + \sqrt{1 + 4 \gamma_t^2}}{2}$
  $\alpha_t \gets \frac{1 - \gamma_t}{\gamma_{t+1}}$
  $\mathbf{y}_{t+1} \gets \theta_t - \frac{1}{L} \nabla f(\theta_t)$
  $\theta_{t+1} \gets \mathbf{y}_{t+1} + \alpha_t (\mathbf{y}_{t+1} - \mathbf{y}_t)$
return $\theta_T$
```

#### **Theorem (Nesterov acceleration rate)**

Let $f$ be convex and $L$-smooth. Nesterov's accelerated gradient descent satisfies
$$
f(\theta_T) - f(\theta^\star) \leq \frac{4 L \|\theta_0 - \theta^\star\|^2}{(T+2)^2}.
$$
If $f$ is additionally $\mu$-strongly convex, the rate is linear with constant $\sqrt{\mu/L}$ replacing $\mu/L$.

The acceleration is **purely a momentum phenomenon**: the algorithm uses the previous iterate's direction to project the next step further along the descent direction. Modern deep learning optimizers (Adam, AdamW) inherit this idea through their momentum terms, though they are not, in general, provably accelerated.

## Stochastic gradient descent

In deep learning, the gradient $\nabla \mathcal{L}(\theta)$ is not available; only a stochastic estimate $\nabla \hat{\mathcal{L}}_B(\theta) = \frac{1}{|B|} \sum_{i \in B} \nabla \ell(\theta; \mathbf{z}_i)$ based on a mini-batch $B$ is accessible. Stochastic gradient descent (SGD) iterates $\theta_{t+1} = \theta_t - \eta_t \nabla \hat{\mathcal{L}}_{B_t}(\theta_t)$.

The fundamental result is due to [robbins1951stochastic].

#### **Theorem (Robbins–Monro convergence)**

Let $\mathcal{L}$ be convex and $L$-smooth, with $\|\nabla \hat{\mathcal{L}}(\theta)\|^2 \leq G^2$ almost surely. If the step sizes satisfy $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$, then SGD converges almost surely: $\mathcal{L}(\theta_t) \to \mathcal{L}(\theta^\star)$.

The condition $\sum \eta_t = \infty$ ensures that the algorithm can travel arbitrarily far; $\sum \eta_t^2 < \infty$ ensures that the noise variance accumulates finitely. The canonical choice is $\eta_t = \eta_0 / (1 + \gamma t)$, satisfying both conditions.

For strongly convex $\mathcal{L}$, the convergence rate of SGD is $O(1/T)$ — slower than the $O((1-\mu/L)^T)$ of deterministic gradient descent, because the noise variance sets a floor. The mini-batch size controls the noise variance: a batch of size $B$ reduces variance by $1/B$ at $B\times$ the per-step cost, leaving the rate at $O(1/T)$ in *wall-clock* terms but reducing the constant.

### The Polyak–Łojasiewicz condition

The PL condition is a weakening of strong convexity that holds for many problems of interest (least squares, logistic regression with separable data) and gives linear convergence of SGD without requiring convexity.

#### **Definition (Polyak–Łojasiewicz condition)**

A differentiable function $f$ satisfies the **PL condition** with parameter $\mu > 0$ if
$$
\frac{1}{2} \|\nabla f(\theta)\|^2 \geq \mu (f(\theta) - f(\theta^\star)) \quad \text{for all } \theta.
$$

#### **Theorem (SGD under PL condition)**

If $f$ is $L$-smooth and satisfies the PL condition with parameter $\mu$, and the stochastic gradient estimator has variance at most $\sigma^2$, then SGD with constant step size $\eta = 1/(2L)$ satisfies
$$
\mathbb{E}[f(\theta_T) - f(\theta^\star)] \leq (1 - \mu/(2L))^T (f(\theta_0) - f(\theta^\star)) + \frac{\sigma^2}{2 \mu L}.
$$

The first term is linear convergence; the second is the noise floor. Increasing the batch size reduces $\sigma^2$ and hence the floor.

## The loss surface of deep networks

The convex theory above is reassuring but inapplicable to deep networks: the empirical loss $\hat{\mathcal{L}}(\theta)$ is highly non-convex, with a loss surface that has been empirically characterized as a tangle of saddle points, flat regions, and connected minima.

### Saddle points dominate

[dauphin2014identifying] argued empirically that saddle points, not local minima, are the principal obstacle to optimization in deep networks. In a network with $P$ parameters, the Hessian at a typical critical point has many positive and many negative eigenvalues — it is a saddle, not a minimum. Theoretical support comes from [choromanska2015loss], who modeled the loss surface as a spin glass and showed that, under the model, critical points are overwhelmingly saddles.

The practical consequence is that **optimization algorithms must escape saddle points efficiently**. Plain gradient descent is provably slow at saddles (the gradient is small, so progress is slow); momentum-based methods and second-order methods escape faster.

### Connected minima

A second empirical regularity is that the local minima of deep networks appear to be **connected** by paths of low loss — there are no "walls" separating good minima. [draxler2018essentially] showed that for a variety of architectures and datasets, any two minima can be connected by a piecewise-linear path along which the loss remains close to its value at the endpoints. This is in stark contrast to the convex picture, where the minimum is unique up to symmetry.

The implication is that the choice of initialization and the trajectory of SGD matter less than one might expect: the algorithm finds *some* good minimum, and the set of good minima is, in a precise sense, connected.

### The neural tangent kernel

For sufficiently wide networks at initialization, the dynamics of gradient descent simplify dramatically. [jacot2018neural] showed that, in the infinite-width limit, the function $f_\theta(\mathbf{x})$ evolves under gradient descent according to a kernel regression with the **neural tangent kernel** $K(\mathbf{x}, \mathbf{y}) = \langle \nabla_\theta f_\theta(\mathbf{x}), \nabla_\theta f_\theta(\mathbf{y}) \rangle$. The optimization becomes convex in function space, even though it remains non-convex in parameter space. We treat the NTK in detail in Volume III, Chapter 4.

## Adaptive optimizers

Modern deep learning is dominated by adaptive gradient methods: AdaGrad [duchi2011adaptive], RMSProp [tieleman2012lecture], Adam [kingma2014adam], and their descendants. These methods maintain per-parameter scaling of the step size based on the history of squared gradients.

#### **Algorithm (Adam)**

```
**Require:** Initial $\theta_0$, step size $\eta$, moments $\beta_1, \beta_2 \in [0,1)$, $\varepsilon > 0$
$m_0 \gets 0$, $v_0 \gets 0$, $t \gets 0$
while not converged do
  $t \gets t + 1$
  $g_t \gets \nabla \hat{\mathcal{L}}_{B_t}(\theta_{t-1})$   // mini-batch gradient
  $m_t \gets \beta_1 m_{t-1} + (1-\beta_1) g_t$   // biased first moment
  $v_t \gets \beta_2 v_{t-1} + (1-\beta_2) g_t^2$   // biased second moment
  $\hat{m}_t \gets m_t / (1 - \beta_1^t)$   // bias correction
  $\hat{v}_t \gets v_t / (1 - \beta_2^t)$
  $\theta_t \gets \theta_{t-1} - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$
```

Adam's popularity stems from its robustness: it works across architectures, datasets, and hyperparameter settings with relatively little tuning. The default settings $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, $\eta = 10^{-3}$ are a reasonable starting point for a wide range of problems.

### Convergence of Adam: the state of the theory

[kingma2014adam] proved convergence of Adam in the convex setting under a decreasing-step-size schedule, with rate $O(1/\sqrt{T})$. [reddi2018convergence] subsequently showed that Adam can **fail to converge** even on simple convex problems under the constant-step-size schedule that is universal in practice. The fix — AMSGrad — maintains the maximum of $v_t$ and recovers convergence.

For non-convex problems (deep networks), there is no satisfactory convergence theory for Adam. The best available results show convergence to a stationary point ($\|\nabla \hat{\mathcal{L}}\| \to 0$ in expectation) at rate $O(1/\sqrt{T})$, but this is a weak statement: it does not guarantee that the algorithm finds a *good* stationary point. The gap between theory and practice is significant.

```python
import numpy as np

def adam(grad_f, theta0, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
         n_iter=1000, seed=0):
    """Adam optimizer on a stochastic gradient function grad_f.

    Parameters
    ----------
    grad_f : callable, returns stochastic gradient given theta
    theta0 : ndarray, initial parameters
    lr : float, learning rate
    betas : (float, float), (beta1, beta2) decay rates
    eps : float, numerical stability
    n_iter : int, number of iterations
    seed : int, RNG seed

    Returns
    -------
    thetas : ndarray, shape (n_iter+1, *theta0.shape), trajectory
    """
    rng = np.random.default_rng(seed)
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2 = betas
    thetas = np.zeros((n_iter + 1,) + theta.shape)
    thetas[0] = theta
    for t in range(1, n_iter + 1):
        g = grad_f(theta, rng)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        thetas[t] = theta
    return thetas
```

## Computational experiments

### Experiment 1: Convergence rates on a quadratic

The following experiment compares plain gradient descent, Nesterov acceleration, and Adam on a quadratic objective $f(\theta) = \frac{1}{2}\theta^\top \mathbf{A} \theta$ where $\mathbf{A}$ has condition number $\kappa$.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(A, theta):    return 0.5 * theta @ A @ theta
def grad(A, theta): return A @ theta

d = 50
kappa = 100
eigvals = np.linspace(1, kappa, d)
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
A = Q @ np.diag(eigvals) @ Q.T
L = kappa  # smoothness
mu = 1     # strong convexity

theta0 = rng.standard_normal(d)
theta_star = np.zeros(d)
f_star = f(A, theta_star)

# GD
T = 5000
gd = np.zeros(T)
theta = theta0.copy()
for t in range(T):
    gd[t] = f(A, theta) - f_star
    theta = theta - (1/L) * grad(A, theta)

# Nesterov
nag = np.zeros(T)
theta, y = theta0.copy(), theta0.copy()
gamma = 0
for t in range(T):
    nag[t] = f(A, theta) - f_star
    g = grad(A, y)
    y_new = theta - (1/L) * g
    gamma_new = (1 + np.sqrt(1 + 4*gamma**2)) / 2
    alpha = (1 - gamma) / gamma_new
    theta = y_new + alpha * (y_new - y)
    y, gamma = y_new, gamma_new

plt.semilogy(gd,  label=f'GD ($\\kappa={kappa}$)')
plt.semilogy(nag, label='Nesterov')
plt.xlabel('iteration')
plt.ylabel('suboptimality $f - f^*$')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig('figures/opt_convergence.pdf')
```

### Experiment 2: SGD noise floor

The following experiment demonstrates the noise floor of SGD under the PL condition. With a small batch size, the loss plateaus at a level set by $\sigma^2 / (2 \mu L)$; with a larger batch, the floor is lower.

### Experiment 3: Adam on a non-convex surface

A simple non-convex test function — the Rosenbrock function — illustrates Adam's robustness. Adam escapes the long curved valley of Rosenbrock in roughly $10^4$ iterations, while plain gradient descent takes $10^6$ or more.

## Historical notes

The convergence theory of gradient descent for convex functions dates to [cauchy1847methode]. Stochastic approximation was introduced by [robbins1951stochastic] in the context of root-finding; its application to machine learning is essentially as old as machine learning itself. The Polyak–Łojasiewicz condition is due to [polyak1963gradient]; its modern revival in machine learning is due to [karimi2016linear].

Nesterov's accelerated gradient method appeared in [nesterov1983method] and was subsequently recognized as optimal among first-order methods for smooth convex optimization [nesterov2018lectures]. The interpretation of acceleration as a momentum phenomenon is due to [su2016differential], who recast NAG as a discretization of a second-order ODE.

The adaptive optimizer family begins with AdaGrad [duchi2011adaptive], which was designed for sparse problems and uses a cumulative squared gradient. RMSProp [tieleman2012lecture] introduced exponential moving average of squared gradients, addressing AdaGrad's vanishing-step problem on dense problems. Adam [kingma2014adam] combined RMSProp's second-moment scaling with momentum; the bias correction makes it work robustly in the early iterations. The non-convergence result is due to [reddi2018convergence]; AMSGrad is the standard fix.

The empirical study of the deep network loss surface is more recent. [choromanska2015loss] modeled it as a spin glass; [dauphin2014identifying] emphasized the role of saddles; [draxler2018essentially] and [garipov2018loss] showed that minima are connected by low-loss paths. The neural tangent kernel is due to [jacot2018neural]; the literature on the NTK and its descendants is now vast.

## Exercises

1. **(Easy)** Prove the descent lemma: if $f$ is $L$-smooth, then $f(\mathbf{y}) \leq f(\mathbf{x}) + \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle + \frac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2$.

2. **(Easy)** Show that the PL condition is implied by strong convexity but does not imply convexity. (Hint: $f(\theta) = \theta^2 + 10 \sin^2(\theta)$ is non-convex but PL.)

3. **(Medium)** Implement Nesterov's accelerated gradient descent and verify the $O(1/T^2)$ rate on a quadratic with $\kappa = 100$. Plot the suboptimality against $T$ on a log-log scale and confirm the slope.

4. **(Medium)** Reproduce the [reddi2018convergence] counterexample: construct a simple one-dimensional convex problem on which Adam fails to converge under constant step size. Verify that AMSGrad converges.

5. **(Medium)** Implement SGD with mini-batching on a logistic regression problem. Plot the loss as a function of wall-clock time for batch sizes $\{1, 16, 256, 4096\}$. Identify the regime where larger batches help and the regime where they hurt.

6. **(Hard)** Prove the SGD-under-PL theorem: SGD under the PL condition with constant step size converges linearly to a noise floor $\sigma^2/(2\mu L)$.

7. **(Hard)** Show that, for a quadratic with Hessian $\mathbf{A}$, the convergence rate of gradient descent is determined by $\kappa = \lambda_{\max}/\lambda_{\min}$. Show that preconditioning by $\mathbf{A}^{-1}$ gives one-step convergence (and explain why this is not useful in practice).

8. **(Research)** Investigate the loss landscape of a small transformer trained on a simple algorithmic task (e.g., modular addition). How does the loss surface change with depth, width, and training data size? Are the minima connected?

## Bibliography
