# Computational Deep Learning — Markdown Edition

A three-volume monograph on computational deep learning, covering mathematical foundations, neural-network algorithms, and current research frontiers. Aimed at PhD-level researchers and graduate students.

**Author:** Mahesh Solanki · **Research Group:** scitamehtam research group

This is the **GitHub Markdown edition** — every chapter is a plain `.md` file that renders natively on GitHub. No build tools required.

## Why markdown?

The Quarto/LaTeX PDF edition of this monograph had rendering issues (blank pages, broken theorem environments, missing code listings). Rather than fight LaTeX, this edition strips the markup down to **clean GitHub-flavored Markdown** that:

- Renders perfectly on GitHub, GitLab, Bitbucket, and any markdown viewer
- Reads cleanly as plain text in a terminal
- Diffs cleanly in version control
- Survives any future toolchain change

## Volumes and chapters

### Volume I — Mathematical Foundations

| # | Chapter | Words |
|---|---|---|
| — | [Introduction to Volume I](volume1-foundations/introduction.md) | 662 |
| 1 | [Linear Algebra for Neural Computation](volume1-foundations/linear-algebra.md) | 3,264 |
| 2 | [Optimization](volume1-foundations/optimization.md) | 2,565 |
| 3 | [Probability and Concentration](volume1-foundations/probability.md) | 1,590 |
| 4 | [Information Theory](volume1-foundations/information-theory.md) | 1,699 |

### Volume II — Algorithms and Architectures

| # | Chapter | Words |
|---|---|---|
| — | [Introduction to Volume II](volume2-algorithms/introduction.md) | 424 |
| 5 | [Multilayer Perceptrons](volume2-algorithms/mlp.md) | 1,585 |
| 6 | [Convolutional Networks](volume2-algorithms/cnn.md) | 1,292 |
| 7 | [Recurrent Networks](volume2-algorithms/rnn.md) | 1,013 |
| 8 | [The Transformer](volume2-algorithms/transformer.md) | 1,323 |
| 9 | [Generative Models](volume2-algorithms/generative.md) | 1,100 |
| 10 | [The Practice of Training](volume2-algorithms/training.md) | 1,213 |

### Volume III — Frontiers and Open Problems

| # | Chapter | Words |
|---|---|---|
| — | [Introduction to Volume III](volume3-frontiers/introduction.md) | 364 |
| 11 | [Representation Learning](volume3-frontiers/representation-learning.md) | 1,301 |
| 12 | [Scaling Laws](volume3-frontiers/scaling-laws.md) | 1,323 |
| 13 | [Generalization Theory](volume3-frontiers/generalization.md) | 1,439 |
| 14 | [Interpretability](volume3-frontiers/interpretability.md) | 1,112 |
| 15 | [Open Problems](volume3-frontiers/open-problems.md) | 1,666 |

### Front matter

- [Welcome](index.md)
- [Preface](preface.md)
- [References](references.md) · [Bibliography (BibTeX)](references.bib)

**Total: ~26,200 words across 21 files.**

## How to read

1. **On GitHub**: just click any chapter link above. Math renders via `$...$` and `$$...$$` (GitHub's native math support). Code blocks have syntax highlighting. Tables, blockquotes, and callouts all render natively.

2. **Locally**: open any `.md` file in VS Code, Obsidian, Typora, or any markdown editor with math support.

3. **As a book**: clone the repo, then use `pandoc` to build a single PDF or EPUB:
   ```bash
   pandoc index.md preface.md \
     volume1-foundations/*.md \
     volume2-algorithms/*.md \
     volume3-frontiers/*.md \
     references.md \
     -o cdl-monograph.pdf \
     --toc --toc-depth=2 \
     --pdf-engine=xelatex \
     -V mainfont="Inter" -V monofont="JetBrains Mono" \
     -V author="Mahesh Solanki"
   ```

## What's in this repository

```
cdl-monograph-md/
├── README.md                       # This file
├── LICENSE                         # CC BY-NC-SA 4.0 (text) + MIT (code)
├── .gitignore
├── index.md                        # Welcome page
├── preface.md                      # Preface
├── references.md                   # References page
├── references.bib                  # Full BibTeX bibliography (167 entries)
├── volume1-foundations/
│   ├── introduction.md
│   ├── linear-algebra.md
│   ├── optimization.md
│   ├── probability.md
│   └── information-theory.md
├── volume2-algorithms/
│   ├── introduction.md
│   ├── mlp.md
│   ├── cnn.md
│   ├── rnn.md
│   ├── transformer.md
│   ├── generative.md
│   └── training.md
└── volume3-frontiers/
    ├── introduction.md
    ├── representation-learning.md
    ├── scaling-laws.md
    ├── generalization.md
    ├── interpretability.md
    └── open-problems.md
```

## Conventions

### Math

GitHub renders `$inline math$` and `$$display math$$` natively. Examples:

- The loss is $L(\theta) = \mathbb{E}[\ell(\theta; X)]$.
- Gradient descent iterates
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t).$$

### Theorem-like environments

Theorems, lemmas, definitions, and algorithms are rendered as **bold H4 headers**:

- **Theorem (Name)** — for theorems, lemmas, propositions, corollaries
- **Definition (Name)** — for definitions
- **Algorithm (Name)** — followed by a code block with the pseudocode
- **Example (Name)** — for examples

### Code listings

Code is in fenced blocks with language hints for syntax highlighting:

```python
import torch
x = torch.randn(64, 128)
```

### Citations

Citations appear as `[bibtexkey]` in square brackets. The full bibliography is in `references.bib`. For example, `[kingma2014adam]` refers to:

> Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization*. ICLR.

## License

- **Text content** (all `.md` files): Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
- **Source code** (code listings in `.md` files): MIT License.

## Citation

If you find this monograph useful, please cite:

```bibtex
@book{solanki2025cdl,
  title  = {Computational Deep Learning: Theory, Algorithms, and Frontiers},
  author = {Solanki, Mahesh},
  year   = {2025},
  note   = {scitamehtam research group, \url{https://github.com/scitamehtam/cdl-monograph}}
}
```

## Contributing

Pull requests welcome. File issues for:

- Errata of any kind
- New exercises at any difficulty level
- New computational experiments
- Missing references
- Translations
