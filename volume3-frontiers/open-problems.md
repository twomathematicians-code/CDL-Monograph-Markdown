# Open Problems

*Volume III · Chapter 15*

---

This chapter enumerates twenty open problems that we believe will shape the next decade of deep learning research. The list is necessarily partial and reflects our own biases; we have favored problems that are *precisely stateable*, *empirically motivated*, and *not obviously impossible*. For each, we give the current state of knowledge, the principal obstacles, and pointers to the relevant literature. The problems are organized by area, but many overlap.

We do not claim this list is comprehensive. A reader looking for a different emphasis should consult the open-problem lists of [bengio2014iclr], [goodfellow2016deep] (Chapter 5), and the various *Frontiers in AI* reports.

## Optimization

### Problem 1: A satisfactory convergence theory for Adam

The Adam optimizer is universally used and universally under-theorized. The [kingma2014adam] convergence proof requires a decreasing step size, which is not used in practice; the [reddi2018convergence] counterexample shows that constant-step Adam can diverge on simple convex problems; AMSGrad fixes this but is not used in practice. A convergence theory that matches practice is the principal open problem of optimizer theory.

The obstacle is that Adam's behavior depends on the interaction between the momentum terms, the second-moment scaling, and the geometry of the loss surface, and these interactions are difficult to analyze cleanly. A candidate approach is the *biased-gradient* framework of [bottou2018optimization], but extending it to Adam's adaptive scaling is non-trivial.

### Problem 2: Why does SGD find good minima in non-convex landscapes?

The empirical fact that SGD finds good minima in deep networks' highly non-convex loss landscapes is the foundational puzzle of optimization in deep learning. The candidate explanations — implicit bias, landscape connectivity, the spin-glass picture — are partial (Chapter 2). A unified theory is missing.

The obstacle is that the relevant property is not about a single optimization trajectory but about the *distribution* of trajectories that SGD visits, and this distribution depends on the interaction between the geometry of the loss surface, the noise structure of mini-batch gradients, and the discrete-time dynamics of SGD.

### Problem 3: The mechanism of scaling laws

The empirical scaling laws (Chapter 12) are remarkably robust but theoretically unexplained. Why does the loss scale as a power law in compute, with exponents that are remarkably consistent across architectures? The candidate explanations — statistical mechanics, information theory, random features — capture aspects but not the whole.

The obstacle is that the scaling law is a statement about the *learning dynamics* of the network, not just the function class, and the dynamics are hard to analyze.

## Generalization

### Problem 4: A unified theory of generalization in deep learning

The classical theory is vacuous; the modern candidates (implicit bias, PAC-Bayes, compression, NTK, lottery ticket) are partial (Chapter 13). A unified theory that integrates these mechanisms and gives non-vacuous bounds for realistic networks is the central open problem of generalization theory.

The obstacle is that the mechanisms interact: implicit bias selects solutions with low complexity, but the relevant complexity measure depends on the architecture and the optimizer; compression bounds work but require explicit compression; PAC-Bayes is non-vacuous only for small networks. A theory that integrates these mechanisms requires understanding their interaction.

### Problem 5: Double descent beyond the simple regimes

Double descent has been documented in many settings, but the precise conditions under which it occurs — and the precise mechanisms by which it occurs — are not fully understood. When does the second descent begin? Why does it eventually saturate? Does it occur in language models at scale?

The obstacle is that double descent requires tracking both the optimization trajectory and the test loss, and the two are coupled in non-trivial ways.

### Problem 6: Generalization under distribution shift

Real-world deployment involves distribution shift: the test distribution differs from the training distribution. The current theory of distribution shift — domain adaptation, robust optimization, invariance — is partial. A predictive theory of when and how networks generalize under distribution shift is missing.

The obstacle is that distribution shift is heterogeneous: it can be a covariate shift, a label shift, a concept drift, or a more complex structural change, and each requires different theoretical tools.

## Representation

### Problem 7: A theory of feature learning

Deep networks learn features, but the feature-learning process is poorly understood. The NTK captures only the linearized dynamics; feature-learning dynamics require going beyond the NTK. The *mean-field* and *feature-learning* limits of [yang2021tensor] and [roberts2022principles] are partial steps.

The obstacle is that feature learning involves *changes* in the kernel during training, and analyzing these changes requires tools that go beyond the linearization.

### Problem 8: The geometry of learned representations

Why are representations often linear? Why are they sometimes in superposition (Chapter 11)? A predictive theory of the geometry of learned representations is missing.

The obstacle is that the geometry depends on the interaction between the architecture, the data, and the optimizer, and the interaction is hard to analyze.

### Problem 9: Disentanglement

The goal of disentanglement — learning representations in which each semantically meaningful factor corresponds to a single dimension — has been pursued since the early days of representation learning. Despite substantial work, no method reliably achieves disentanglement without supervision. Theoretical results ([locatello2019challenging]) show that unsupervised disentanglement is fundamentally ill-posed without inductive bias.

The obstacle is that the "true" factors of variation are not uniquely defined by the data distribution; they depend on the downstream task. A theory that integrates the task into the disentanglement criterion is missing.

## Architectures

### Problem 10: Beyond the transformer

The transformer has dominated for seven years. Is there a better architecture, and if so, what is its inductive bias? Candidate alternatives — Mamba [gu2023mamba], RWKV [peng2023rwkv], Hyena [poli2023hyena] — are competitive in some regimes but have not displaced the transformer.

The obstacle is that the transformer's success is partly due to its hardware friendliness (parallelism, memory access patterns), and any alternative must compete on hardware as well as on mathematical structure.

### Problem 11: The role of recurrence in long-context models

Transformers have $O(n^2)$ attention cost; recurrent models have $O(n)$ but a more limited function class. The right trade-off for long-context modeling — the architecture that combines the expressivity of attention with the linear scaling of recurrence — is open.

### Problem 12: Modular and compositional architectures

Biological brains are modular; current deep networks are largely monolithic. A theory of *modular composition* — when should the network be split into modules, how should the modules communicate, how should the modules be specialized — is missing. Mixture-of-experts [shazeer2017outrageously] is a partial step.

## Interpretability

### Problem 13: Scalable mechanistic interpretability

Current interpretability methods work on small models. Scaling them to models with $10^{11}$–$10^{13}$ parameters is open (Chapter 14). The obstacle is that the methods are labor-intensive: each circuit must be identified and validated by hand.

### Problem 14: Automated circuit discovery

Manual circuit discovery does not scale. Automated methods — sparse autoencoders, dictionary learning, attribution — are promising but immature. A method that automatically identifies the circuits in a large network is the principal methodological open problem of interpretability.

### Problem 15: The relationship between interpretability and alignment

The intended use of interpretability is alignment: ensuring that networks do what we want, not what they appear to do. The connection between interpretability and alignment is, however, not fully worked out. Does interpretability actually help alignment, or does it just give us a false sense of understanding?

## Training and data

### Problem 16: The data wall

High-quality text data is finite and may be exhausted at scales of $10^{13}$–$10^{14}$ tokens (Chapter 12). What happens then? Candidate solutions — synthetic data, data replay, multimodal data — are partial. A theory of the *data requirements* of large models, beyond the Chinchilla law, is open.

### Problem 17: Curriculum and data ordering

The order in which data is presented affects the final model, but the effect is poorly understood. Optimal curricula are not known; the heuristics in use (easy-to-hard, balanced sampling) have only weak theoretical justification.

### Problem 18: The role of model and data quality

Not all data is equally useful; not all parameters are equally informative. A theory that integrates *quality* into the scaling laws — predicting how the loss scales with data quality, or how the optimal allocation changes with model architecture — is open.

## Theory and practice

### Problem 19: A theory of in-context learning

Large language models learn from context: a few examples in the prompt improve performance on a new task. The mechanism by which in-context learning works — and the conditions under which it works — are only partially understood. The induction-head discovery ([olsson2022incontext]) is a step; a complete theory is open.

### Problem 20: The relationship between deep learning and statistics

Deep learning is, in some sense, a statistical method, but the relationship to classical statistics is murky. Many classical statistical concepts (consistency, efficiency, sufficiency, the Cramer-Rao bound) have analogs in deep learning, but the analogs are partial. A *statistical theory of deep learning* that integrates the classical concepts with the empirical phenomena — a theory that explains why deep networks are surprisingly good statistical estimators despite their huge parameter counts — is the grand open problem of the field.

## Coda

The twenty problems above are not the only open problems in deep learning. They are a partial list, biased by our own interests and by the limits of our knowledge. We offer them in the spirit of [hilbert1900mathematical]: as a guide to the principal work of the next decade, and as an invitation to the reader to take up the problems we have not been able to solve.

Deep learning is, in 2025, a field with enormous empirical success, substantial theoretical progress, and large open theoretical questions. The next decade will, we hope, see the open problems closed, the empirical record systematized, and the theory brought into line with practice. We hope this monograph contributes to that work.

## Bibliography
