# Volume II — Algorithms and Architectures

*Volume II*

---

# Introduction to Volume II
Volume II treats the principal architectures of modern deep learning. Each chapter follows a uniform structure: the architecture is introduced as a function class with a specific inductive bias; the relevant theorems (expressivity, universal approximation, equivariance) are stated and proved or cited; the training algorithm is given explicitly in pseudocode and Python; the empirical performance is reported with references to the primary literature; and the historical notes situate the architecture in the trajectory of the field.

The unifying perspective is that **architectures are constraints on the function class**. Convolution imposes translation equivariance. Recurrence imposes temporal consistency. Attention imposes permutation equivariance. Residual connections impose a bias toward the identity and away from zero. Each constraint makes some functions easy to express and others hard; the choice of architecture is the choice of which constraints match the structure of the problem.

## Chapter map

**Chapter 5 — Multilayer Perceptrons**. The simplest deep architecture: alternating affine maps and elementwise nonlinearities. Universal approximation (Cybenko, Hornik), the depth-vs-width tradeoff, the role of activation functions, and the practical considerations of initialization, normalization, and regularization.

**Chapter 6 — Convolutional Networks**. The architecture that ignited the deep learning era. Translation equivariance, the convolution-as-multiplication view, pooling and invariance, the canonical architectures (LeNet, AlexNet, VGG, ResNet, EfficientNet), and the modern successors (vision transformers, convnext).

**Chapter 7 — Recurrent Networks**. Sequence models before attention. RNNs, LSTMs, GRUs, the vanishing and exploding gradient problem, the connection to dynamical systems, and the practical considerations of sequence length and training stability.

**Chapter 8 — The Transformer**. The architecture that has come to dominate natural language processing and increasingly vision, speech, and other modalities. Self-attention, multi-head attention, positional encodings, the canonical architectures (BERT, GPT, T5), and the scaling properties that have driven the field since 2017.

**Chapter 9 — Generative Models**. The three principal families: variational autoencoders, generative adversarial networks, and diffusion models. Each is treated as a different solution to the problem of fitting an implicit or explicit density model to data.

**Chapter 10 — The Practice of Training**. The engineering of large-scale training: initialization, normalization, regularization, optimization, hyperparameter tuning, distributed training, and the monitoring of training runs. This is the chapter a practitioner needs; the others supply the conceptual context.

## How to read

Chapter 5 is a prerequisite for the others; the remaining chapters can be read in any order. The computational experiments in each chapter are runnable on a single modern GPU; the exercises range from routine to research-level.
