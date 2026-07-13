# Interpretability

*Volume III · Chapter 14*

---

**Mechanistic interpretability** is the program of reverse-engineering trained neural networks: understanding, in human-meaningful terms, the computations the network performs. The program is motivated by safety (we cannot align systems we do not understand), by science (deep networks are among the most complex artifacts humans have built, and their internal computations are largely opaque), and by engineering (understanding failure modes is the first step to fixing them).

This chapter surveys the principal approaches: **feature visualization** (what does a neuron represent?), **circuits** (how do neurons compose into computations?), **sparse autoencoders** (decomposing polysemantic neurons into monosemantic features), and **causal interventions** (testing hypotheses about computational structure). The chapter treats each method's strengths, limitations, and current frontiers.

## Feature visualization

The simplest interpretability question is: *what does a neuron represent?* **Feature visualization** [olah2017feature] answers this question by optimization: find the input that maximally activates a given unit. The optimization is regularized to produce natural-looking images, since the unconstrained optimum is often an adversarial-like pattern.

Feature visualization is informative for individual neurons in vision models but has limitations: (i) the most-activating natural images often tell a different story than the optimized inputs; (ii) polysemantic neurons (Chapter 11) have no single concept to visualize; (iii) the method is most useful for low-level features and less useful for high-level abstractions.

## Circuits

A **circuit** is a subnetwork of a trained network that implements a specific computation. [olah2020zoom] identified circuits in vision transformers: curve detectors, orientation detectors, color opponency circuits, and others. The methodology combines feature visualization (to identify what each unit represents), weight inspection (to identify which units connect to which), and ablation (to test whether the connections are causally responsible for the computation).

The circuits program has been productive for early layers of vision models, where computations are relatively simple and the units are often monosemantic. For later layers and for language models, the situation is harder: polysemanticity is widespread, the relevant computations are more abstract, and circuits span more of the network.

The current frontier is **mechanistic interpretability of language models**: identifying the circuits by which transformers implement specific syntactic computations (subject-verb agreement, indirect object identification, in-context learning). [wang2022interpretability] identified the circuit for indirect object identification in GPT-2 small; [olsson2022incontext] identified "induction heads" as a mechanism for in-context learning. These case studies are encouraging but cover only a small fraction of the computations performed by the models.

## Sparse autoencoders for interpretability

As introduced in Chapter 11, sparse autoencoders (SAEs) decompose polysemantic neurons into monosemantic features. The methodology has become a principal tool of mechanistic interpretability, allowing researchers to identify the features that the network actually uses (as opposed to the neurons it happens to have).

The state of the art is rapidly evolving. [bricken2023monosemanticity] trained SAEs on a small transformer and recovered interpretable features; [templeton2024scaling] scaled this to Claude 3 Sonnet and recovered features corresponding to a wide range of concepts (countries, code, biases, intentions). The features are causally meaningful: activating or suppressing a specific feature changes the model's behavior in predictable ways.

## Causal interventions

The strongest form of interpretability evidence is **causal**: not just "this unit represents X" but "this unit is *used* by the network to compute X." Causal interventions test the latter by perturbing the unit and observing the effect.

**Activation patching** (Chapter 11) replaces activations from one input with those from another; the effect on the output identifies the causal role. **Path patching** does the same for specific paths through the network. **Ablation** zeros out units and measures the effect. **Activation steering** scales a specific direction up or down, allowing controlled manipulation of the model's behavior.

The combination of SAEs (for identifying features) and causal interventions (for testing their roles) is the current methodological frontier. The principal open question is **scalability**: methods that work on small models (GPT-2 small, 124M parameters) are expensive to apply to large models (GPT-4-class, ~1T parameters). Whether interpretability can scale with model size is one of the central open problems of the field.

## The microscope vs. MRI distinction

[olah2022theoretical] distinguishes two complementary modes of interpretability research:

- **Microscope**: zoom in on a specific computation, identify the circuit, test the mechanism. High-resolution, low-coverage.
- **MRI**: scan the whole model for patterns, identify common motifs, build a topographic map. Low-resolution, high-coverage.

Most current interpretability research is microscope-style: detailed studies of specific circuits in specific models. The MRI mode — systematic surveys of entire models — is methodologically harder and less developed, but is necessary if interpretability is to scale.

## Computational experiments

### Experiment 1: Feature visualization

Implement feature visualization for a pretrained CNN. Optimize inputs to maximally activate specific neurons. Compare the optimized inputs to the top natural images for the same neurons.

### Experiment 2: Circuit identification

Identify a simple circuit (e.g., curve detector) in the early layers of a pretrained vision model. Verify the circuit by ablation: zero out the circuit's units and measure the effect on the relevant behavior.

### Experiment 3: Activation steering

Train a small SAE on a pretrained language model's activations. Find a feature corresponding to a behavioral property (e.g., refusal, sycophancy). Steer the model by scaling the feature's activation up or down; verify the behavioral change.

## Historical notes

Feature visualization has roots in [simonyan2013deep]; the modern form is due to [olah2017feature]. The circuits program was launched by [olah2020zoom]. The sparse-autoencoder methodology for interpretability is due to [bricken2023monosemanticity] and [templeton2024scaling]. Activation patching as an interpretability tool was introduced by [geiger2021causal] and [vig2020investigating]. The induction-head discovery is due to [olsson2022incontext]. The microscope/MRI distinction is due to [olah2022theoretical].

## Exercises

1. **(Easy)** Implement feature visualization for a single neuron in a pretrained CNN. Compare the optimized input to the top natural images for the same neuron.

2. **(Easy)** Implement a simple ablation: zero out a specific unit in a pretrained model and measure the effect on a specific behavior.

3. **(Medium)** Implement activation patching on a small transformer. Use it to identify the positions causally responsible for a specific output token.

4. **(Medium)** Train a small SAE on a pretrained model's activations. Inspect the learned features. Are they monosemantic?

5. **(Medium)** Implement path patching. Use it to identify the circuit by which a small transformer solves a simple algorithmic task.

6. **(Hard)** Implement activation steering. Find a direction in a language model's activations corresponding to a behavioral property, and steer the model by scaling the direction.

7. **(Hard)** Reproduce the induction-head experiment of [olsson2022incontext] in a small transformer.

8. **(Research)** Investigate the scalability of interpretability methods. How do the cost and the yield of SAE-based interpretability scale with model size?

## Bibliography
