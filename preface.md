# Preface

---

Deep learning has, in less than two decades, moved from a marginal subfield of machine learning — sustained by a small community of persistent researchers — to a dominant paradigm in artificial intelligence, with measurable impact on computer vision, speech recognition, natural language processing, scientific computing, and increasingly on the empirical sciences themselves. The defining methodological fact of this transformation is deceptively simple: very large parametric function classes, trained by stochastic gradient descent on very large datasets, can generalize remarkably well. The defining theoretical puzzle is that they do so in regimes where classical learning theory predicts they should not.

## Why this monograph, and why now

The existing literature on deep learning falls roughly into three categories: (i) course-oriented textbooks such as [goodfellow2016deep] and [bishop2006pattern], which are pedagogical and broad; (ii) survey articles on individual topics, which are deep but narrow; and (iii) the primary research literature, which is current but fragmented across conferences and preprint servers. A monograph that integrates the three — taking the time to develop the mathematics honestly, to implement the algorithms explicitly, and to situate both against the empirical and theoretical frontier — is overdue.

This book is written for the reader who has moved beyond introductions and is ready for the harder material: convergence proofs for Adam, the neural tangent kernel, double descent, mechanistic interpretability, scaling laws. The reader who can already train a transformer but wants to understand *why* a transformer. The reader who has read the original papers but wants them integrated into a single coherent story.

## What "computational" means in the title

We use the word *computational* deliberately. Three senses are intended:

1. **Computational in the algorithmic sense.** Every result in this book is accompanied by an explicit algorithm — pseudocode in the chapter, executable code in the companion notebooks. We are unwilling to leave "the algorithm" as folklore.
2. **Computational in the resource sense.** Modern deep learning is shaped as much by the geometry of GPU memory, the bandwidth of interconnect, and the cost of communication as by any mathematical consideration. Where these constraints matter, we discuss them.
3. **Computational in the experimental sense.** The field progresses by experiments, and theory that ignores the empirical record is of limited use. Each chapter reports key empirical findings — loss curves, scaling plots, ablations — with full provenance.

## The plan of the work

**Volume I** develops the four pillars on which the rest of the book stands. Linear algebra is presented not as a refresher but as the language of neural computation: tensor factorizations, spectral methods, low-rank approximation, randomized numerical linear algebra, and the geometry of high-dimensional spaces. Optimization is developed in two passes: convex optimization (where the theory is essentially complete, following [boyd2004convex] and [nesterov2018lectures]) and non-convex optimization (where the theory is partial and the empirical record matters). Probability and concentration inequalities are developed to the depth required to read the generalization literature. Information theory is developed to the depth required to read both the representation learning literature and the variational inference literature.

**Volume II** treats the principal architectures of modern deep learning in a uniform style: each is introduced as a function class with a specific inductive bias, the relevant theorems (expressivity, universal approximation, equivariance) are proved or cited, the training algorithm is given explicitly, and the empirical performance is reported. The volume closes with a chapter on the practice of training — initialization, normalization, regularization, optimization, and the engineering of large-scale training runs.

**Volume III** turns to the open questions. Why do overparameterized networks generalize? What does the geometry of the loss surface look like, and why does it matter? What do scaling laws tell us about the structure of the learning problem? How should we understand the representations learned by deep networks, and can we read them? The volume closes with an explicit list of open problems — twenty of them — that we believe will shape the next decade.

## A note on rigor

We have chosen rigor where rigor is illuminating and citation where it is not. Proofs are given in full when the proof technique itself is illuminating (the Cybenko–Hornik universal approximation proof, the NTK derivation, the Adam convergence argument under decreasing steps). Proofs are sketched or cited when the full argument is technical but the conclusion is widely used (the Johnson–Lindenstrauss lemma, the manifold tangent approximation, the Bartlett–Mendelson Rademacher bound). We have made every effort to ensure that every cited result is correctly attributed; errors should be reported to the repository's issue tracker.

## A note on the computational material

All code in the monograph is written in Python, using PyTorch as the primary deep learning framework and NumPy / SciPy / Matplotlib for scientific computing and plotting. The companion notebooks in the `notebooks/` directory of the repository are executable: each can be run end-to-end on a single modern GPU in under an hour, and most can be run on a laptop CPU in under ten minutes. The repository includes a `Makefile` and a `pyproject.toml` specifying the exact environment; CI runs all notebooks on every commit.

## How this book was written

This monograph is itself a computational artifact. The source is a Quarto project — markdown files with embedded code, rendered to HTML, PDF (via Tectonic), and EPUB from a single source of truth. The repository is public, version-controlled, and continuously built. Readers are invited to file issues, suggest improvements, and submit pull requests.

— **scitamehtam research group** · Mahesh Solanki

Summer, 2025
