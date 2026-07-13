# Scaling Laws

*Volume III · Chapter 12*

---

One of the most striking empirical regularities of modern deep learning is that **training loss decreases as a power law in compute, model size, and data**. This regularity — a *scaling law* — has reshaped the field since 2020, transforming model development from an artisanal process of architecture search into an engineering discipline of scale-up. This chapter surveys the empirical laws, the theoretical explanations that have been proposed, and the implications for compute-optimal training.

## The Kaplan scaling law

[kaplan2020scaling] measured the test loss of decoder-only transformers trained on a large text corpus, varying model size $N$ (parameters), dataset size $D$ (tokens), and compute $C$ (FLOPs). They found that, when not bottlenecked by the other factors, the loss scales as a power law:

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C},
$$

with exponents $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.05$. The scaling is smooth over six orders of magnitude in $N$ and $D$, with no detectable departures — a remarkable regularity.

The Kaplan law implies that the path to better models is straightforward: scale up. It also implies a specific *compute-optimal* allocation: given a fixed compute budget $C$, how should one trade off $N$ and $D$? Kaplan et al. concluded that model size should grow faster than data — that very large models trained on relatively modest data were compute-optimal. This conclusion drove the development of models like GPT-3 [brown2020language] ($175$B parameters trained on $300$B tokens).

## The Chinchilla correction

[hoffmann2022training] revisited the Kaplan scaling law with a much larger set of training runs and a refined methodology. Their headline finding was that **Kaplan et al. under-allocated data**: compute-optimal training requires data to scale roughly linearly with model size, not sub-linearly. The corrected scaling law is

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta},
$$

with $\alpha \approx 0.34$, $\beta \approx 0.28$, $E$ an irreducible entropy term, and $A, B$ constants. The compute-optimal allocation is approximately $N \propto C^{0.5}$, $D \propto C^{0.5}$: model size and data should scale in equal proportion.

The practical implication is dramatic: a compute-optimal Chinchilla model ($70$B parameters, $1.4$T tokens) matches or exceeds the performance of GPT-3 ($175$B parameters, $300$B tokens) while using $4\times$ less compute. The Kaplan-GPT-3 allocation was, in retrospect, a substantial over-investment in parameters at the expense of data. Subsequent large models (LLaMA, PaLM, Mistral) have largely adopted the Chinchilla allocation.

## Scaling beyond Chinchilla

Several extensions and refinements of the Chinchilla law have been studied:

**Scaling of downstream task performance**. The Kaplan and Chinchilla laws describe *next-token prediction loss*, not task performance. The mapping from loss to task accuracy is approximately sigmoidal: a given loss reduction corresponds to a larger accuracy gain when the model is near the "threshold" of capability and a smaller gain when the model is far from the threshold. The emergence of new capabilities at scale — a phenomenon documented by [wei2022emergent] — is partly an artifact of this sigmoid mapping and partly a real change in the model's behavior.

**Data-constrained scaling**. The Chinchilla law assumes unlimited data. In practice, high-quality text data is finite and may be exhausted at scales of $10^{13}$–$10^{14}$ tokens. [muennighoff2023scaling] extended the scaling law to data-constrained regimes, where the law breaks down and additional data must be synthesized, translated, or replayed.

**Compute-optimal inference**. At inference time, the cost of generating each token depends on the model size. For a fixed total inference budget, smaller models running longer can outperform larger models running shorter; the optimal model size for inference is much smaller than for training. This is the foundation of the **distillation** paradigm: train a large model, distill to a small one, deploy the small one.

## Theoretical explanations

Several theoretical explanations for the empirical scaling laws have been proposed, none fully satisfactory.

**Statistical mechanics**. [bahri2021statistical] and others have derived scaling laws from a statistical-mechanics analysis of the loss landscape, treating the network as a disordered system and the loss as a free energy. The exponents fall out of the analysis under certain assumptions; the assumptions are restrictive.

**Information-theoretic**. The irreducible loss $E$ is naturally interpreted as the entropy of the data; the reducible loss $(A/N^\alpha + B/D^\beta)$ is the *complexity* of approximating the data distribution with a network of size $N$ trained on $D$ samples. The exponents $\alpha, \beta$ are harder to derive from information-theoretic arguments.

**Random features and kernel methods**. In the infinite-width limit, neural networks become kernel methods (the NTK, Chapter 13), and the scaling laws of kernel methods can be computed explicitly. The result matches the empirical exponents qualitatively but not quantitatively.

The honest state of the field is that **scaling laws are an empirical regularity in search of a theory**. The laws themselves are robust — they have been reproduced across many labs, datasets, and architectures — but the underlying mechanism is not understood.

## Implications for compute-optimal training

The practical implications of scaling laws are:

1. **Scale predictably**. Given a compute budget, the Chinchilla law predicts the achievable loss and the optimal model/data split. This makes large-scale training a planning exercise rather than a research gamble.
2. **Allocate data generously**. The Chinchilla correction showed that data should scale with model size; the field has largely corrected the under-allocation of the GPT-3 era.
3. **Watch for breakdowns**. The laws assume unlimited high-quality data; as data becomes constrained, the laws break down and the marginal return on compute falls.
4. **Plan for inference**. The training compute-optimal model is not the inference compute-optimal model; distillation and quantization are essential parts of the deployment pipeline.

## Computational experiments

### Experiment 1: Kaplan scaling on a small dataset

Train transformers of varying size (e.g., $10$M, $30$M, $100$M, $300$M parameters) on a fixed text dataset. Plot the final training loss as a function of parameter count on a log-log scale. Verify the power-law scaling.

### Experiment 2: Compute-optimal allocation

For a fixed compute budget, train models of varying sizes with appropriate data. Identify the compute-optimal allocation. Compare to the Chinchilla prediction.

### Experiment 3: Emergence

Train a model on a task with a clear threshold (e.g., modular arithmetic). Plot accuracy as a function of compute. Verify the sigmoid-like emergence.

## Historical notes

The empirical study of scaling in deep learning predates the transformer: [hestness2017deep] reported power-law scaling of test loss with data for a range of models. The transformer-specific scaling law is due to [kaplan2020scaling]. The Chinchilla correction is due to [hoffmann2022training]. The emergence phenomenon was systematically documented by [wei2022emergent]. The statistical-mechanics perspective is developed in [bahri2021statistical]. The data-constrained regime is treated in [muennighoff2023scaling].

## Exercises

1. **(Easy)** State the Chinchilla scaling law. What are the values of the exponents $\alpha, \beta$?

2. **(Easy)** Compute the compute-optimal allocation of model and data for a budget of $10^{22}$ FLOPs, using the Chinchilla exponents.

3. **(Medium)** Reproduce a small-scale scaling experiment: train models of $4$ sizes on a fixed dataset, fit the Kaplan and Chinchilla laws, compare the fits.

4. **(Medium)** Investigate emergence: train a model on a synthetic task with a clear capability threshold (e.g., $n$-digit addition). Plot accuracy as a function of model size and identify the emergence scale.

5. **(Medium)** Implement a data replay strategy: train a model on a fixed dataset for multiple epochs. Does the scaling law continue to hold, or does the law break down?

6. **(Hard)** Derive the compute-optimal allocation from the Chinchilla law. Show that $N \propto C^a$, $D \propto C^b$ with $a + b = 1$, and find $a, b$.

7. **(Hard)** Investigate the breakdown of scaling laws in the data-constrained regime. Train a model on a small dataset for many epochs. How does the loss scale with compute?

8. **(Research)** Investigate the scaling of *downstream* task performance (rather than next-token loss). How does it differ from the scaling of the loss? Are the exponents the same?

## Bibliography
