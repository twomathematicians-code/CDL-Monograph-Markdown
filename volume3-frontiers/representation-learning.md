# Representation Learning

*Volume III · Chapter 11*

---

What does a deep network *know*? The question has at least three readings. The first asks what *function* the network computes — answered by querying it on inputs. The second asks what *features* the network has learned — answered by inspecting intermediate activations. The third asks what *concepts* those features correspond to — answered, when it can be answered at all, by careful probing and causal intervention. This chapter treats the second and third readings, surveying the empirical and theoretical understanding of the representations learned by deep networks.

## The manifold hypothesis

Natural data — images, audio, text — are widely believed to lie on or near a low-dimensional manifold embedded in the high-dimensional ambient space. The empirical evidence is twofold: (i) local dimensionality estimators applied to natural data give dimensions much smaller than the ambient (e.g., $30$–$100$ for natural image patches, vs. an ambient dimension of $768$ for $16 \times 16$ RGB patches); (ii) generative models with low-dimensional latent spaces (VAEs, GANs, diffusion models) generate convincing samples, suggesting that the data distribution is well-approximated by a low-dimensional projection.

#### **Definition (Manifold hypothesis)**

The support of the data distribution $p_{\text{data}}$ on $\mathbb{R}^D$ is contained, up to a set of measure zero, in a $d$-dimensional smooth submanifold $\mathcal{M} \subset \mathbb{R}^D$ with $d \ll D$.

The manifold hypothesis is an empirical claim, not a theorem. It is contested: [pope2021intrinsic] measured intrinsic dimensions across many datasets and found substantial variation; some data modalities (text in particular) may not satisfy the hypothesis in any simple form. Nonetheless, the hypothesis is the right organizing framework for thinking about representations: a deep network's job is, on this view, to *identify the manifold* — to map from $\mathbb{R}^D$ to a coordinate system on $\mathcal{M}$.

## Linear representations

A striking empirical regularity, documented most clearly in modern large language models, is that **semantic concepts are often linearly encoded** in the activations. The classic example is gender in word embeddings ([mikolov2013distributed]): $\text{vec}(\text{king}) - \text{vec}(\text{man}) \approx \text{vec}(\text{queen}) - \text{vec}(\text{woman})$, with a similar arithmetic for capital-country relations, verb tense, and so on. More recently, the same linear structure has been found in transformer activations for syntactic categories, factual knowledge, and behavioral properties.

The **linear representation hypothesis** [park2024linear] codifies this: concepts correspond to directions in activation space, and concept *strength* corresponds to projection along the direction. The hypothesis has substantial empirical support and is the foundation of much current mechanistic interpretability work.

## Superposition and polysemantic neurons

A central puzzle is the **polysemanticity** of neurons: individual units in trained networks often respond to multiple, apparently unrelated concepts. A single neuron in a vision model might fire for cat faces, car fronts, and the letter "S" — a striking lack of the clean one-neuron-one-concept mapping that early interpretability work hoped for.

[elhage2022superposition] proposed the **superposition hypothesis**: networks represent more features than they have neurons by storing features in *superposition*, with features encoded as nearly-orthogonal directions in a high-dimensional activation space. The geometry is approximately that of the Johnson–Lindenstrauss lemma (Chapter 1): $N$ features can be stored in $d \ll N$ dimensions with low interference, provided the features are *sparse* (only a small fraction are active at any given input).

The superposition hypothesis has substantial explanatory power: it explains polysemanticity, the empirical success of sparse coding, and the emergence of clean monosemantic features under sparse autoencoders. It also poses a clean theoretical question — when does a network prefer sparse superposition over dense one-to-one representation? — that is partially answered by [anthropic2024scaling].

## Sparse autoencoders

A **sparse autoencoder** (SAE) is a tool for recovering the underlying features from a network's activations. Given activations $\mathbf{x} \in \mathbb{R}^d$, an SAE computes

$$
\mathbf{z} = \text{ReLU}(\mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}), \quad \hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}',
$$

with $|\mathbf{z}|_0$ small (sparsity penalty) and the dictionary $\mathbf{W}_{\text{dec}} \in \mathbb{R}^{d \times N}$ having $N \gg d$ columns. The columns of $\mathbf{W}_{\text{dec}}$ are the learned **features**, and the sparse code $\mathbf{z}$ tells which features are present in $\mathbf{x}$.

Empirically, SAEs trained on the activations of large language models recover a substantial number of **monosemantic** features — features that respond to a single, identifiable concept. The recovered features include syntactic roles (subject, object, verb), semantic categories (country, code, math), and behavioral properties (the network's intent to refuse, to flatter, to confabulate). The state of the art is rapidly evolving; see [templeton2024scaling] for a recent large-scale study.

## Probing and causal interventions

A **linear probe** is a simple classifier trained on a network's activations to predict some property of the input. A high probing accuracy indicates that the property is linearly decodable from the activations — that the network "knows" about the property in some operational sense. Probing is widely used but has a key limitation: it tests whether the property is *represented*, not whether the network *uses* the representation. A network might represent a property it does not use, or use a property it does not linearly represent.

**Causal interventions** address this limitation. The simplest is **activation patching**: replace the activations of a network at a specific layer and position with those from a different input, and observe the effect on the output. If the patch changes the output in the expected direction, the patched activations causally mediate the relevant computation. **Ablation** is the special case of patching with zeros.

A more refined intervention is **path patching**, which patches only the activations flowing along specific paths through the network. Path patching has been used to identify the circuits by which transformers implement specific syntactic computations, and is a principal tool of mechanistic interpretability (Chapter 14).

## Computational experiments

### Experiment 1: Word embedding arithmetic

Load a pretrained word embedding (word2vec, GloVe). Verify the king–queen analogy and find three more analogies that work quantitatively.

### Experiment 2: Polysemanticity

Train a small CNN on a multi-task image dataset (e.g., CIFAR-10 + a synthetic attribute). Visualize the top activations of neurons in the penultimate layer; observe that many neurons respond to multiple concepts.

### Experiment 3: Sparse autoencoder

Train a small SAE on the activations of a pretrained model (e.g., ResNet18 on ImageNet). Inspect the learned features; quantify monosemanticity by correlation with known concepts.

## Historical notes

The manifold hypothesis has roots in the early vision literature; [tenenbaum2000global] and [roweis2000nonlinear] introduced the modern dimensionality-reduction framework. Linear representation in word embeddings was emphasized by [mikolov2013distributed]. The superposition hypothesis is due to [elhage2022superposition]; the sparse-autoencoder methodology was developed by [bricken2023monosemanticity] and [templeton2024scaling]. Activation patching as a causal intervention was introduced by [geiger2021causal] and [vig2020investigating]; path patching by [wang2022interpretability].

## Exercises

1. **(Easy)** Implement a linear probe. Train it on the activations of a small CNN to predict the class label. What does the probing accuracy tell you? What does it not tell you?

2. **(Easy)** Implement activation patching. Verify that patching the activations of a transformer at a specific position changes the output in the expected direction.

3. **(Medium)** Train a sparse autoencoder on a small model's activations. Quantify the trade-off between reconstruction error and sparsity.

4. **(Medium)** Estimate the intrinsic dimensionality of a small image dataset (e.g., CIFAR-10) using a nearest-neighbor-based estimator.

5. **(Medium)** Implement a linear concept erasure operation: given a direction in activation space, project it out and observe the effect on the network's behavior.

6. **(Hard)** Reproduce a superposition experiment: train a small ReLU autoencoder on a synthetic sparse dataset with more features than dimensions. Verify that the learned dictionary approximates the original features.

7. **(Hard)** Implement path patching. Use it to identify the circuit by which a small transformer solves a simple algorithmic task (e.g., induction).

8. **(Research)** Investigate the relationship between superposition and feature geometry. How does the angle between learned features relate to their co-occurrence statistics?

## Bibliography
