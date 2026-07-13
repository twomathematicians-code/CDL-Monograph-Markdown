# The Transformer

*Volume II · Chapter 8*

---

The transformer [vaswani2017attention] is the architecture that has come to dominate natural language processing and, increasingly, computer vision, speech, and other domains. Its central innovation is the **self-attention** mechanism, which replaces the recurrence of RNNs with a single, parallelizable operation that allows each position in a sequence to attend to all other positions. The combination of attention, residual connections, layer normalization, and positional encodings has proved remarkably scalable: transformers have been trained at scales from millions to hundreds of billions of parameters, with consistent improvements in capability.

## Self-attention

Given a sequence of input vectors $\mathbf{X} \in \mathbb{R}^{n \times d}$, self-attention computes three projections — queries $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$, keys $\mathbf{K} = \mathbf{X}\mathbf{W}_K$, values $\mathbf{V} = \mathbf{X}\mathbf{W}_V$, all in $\mathbb{R}^{n \times d_h}$ — and outputs

$$
\text{Attn}(\mathbf{X}) = \text{softmax}\left( \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}} \right) \mathbf{V} \in \mathbb{R}^{n \times d_h}.
$$

The softmax is row-wise: the $i$-th output is a weighted average of the value vectors, with weights proportional to the (scaled) inner product between the $i$-th query and each key. The scaling by $1/\sqrt{d_h}$ counteracts the growth of inner products with dimension, keeping the softmax well-conditioned.

The computational cost is $O(n^2 d_h)$ for the attention matrix and $O(n d^2)$ for the projections. The quadratic dependence on sequence length $n$ is the principal limitation of the transformer; we discuss efficient attention variants below.

### Multi-head attention

In practice, attention is computed in parallel by $H$ "heads," each with its own projections $\mathbf{W}_Q^{(h)}, \mathbf{W}_K^{(h)}, \mathbf{W}_V^{(h)}$ of dimension $d_h = d / H$. The outputs are concatenated and projected back to dimension $d$:

$$
\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \mathbf{W}_O, \quad \text{head}_h = \text{Attn}(\mathbf{X}\mathbf{W}_Q^{(h)}, \mathbf{X}\mathbf{W}_K^{(h)}, \mathbf{X}\mathbf{W}_V^{(h)}).
$$

Multi-head attention allows the model to attend to different aspects of the input simultaneously: different heads can specialize to different relations (syntactic, semantic, positional) without explicit supervision. The empirical observation that heads in trained transformers specialize is one of the foundations of mechanistic interpretability (Volume III, Chapter 5).

## Positional encodings

Self-attention is permutation equivariant: it has no notion of order. To incorporate positional information, the input is augmented with **positional encodings**. The original transformer [vaswani2017attention] used sinusoidal encodings of different frequencies:

$$
\text{PE}_{i, 2k} = \sin(i / 10000^{2k/d}), \quad \text{PE}_{i, 2k+1} = \cos(i / 10000^{2k/d}),
$$

where $i$ is the position and $k$ is the dimension. Sinusoidal encodings have the property that $\text{PE}_{i + \delta}$ is a linear function of $\text{PE}_i$, allowing the model to learn to attend by relative position.

**Learned absolute positional embeddings** (used in BERT, GPT) replace the sinusoidal encoding with a learned vector per position; this is simpler but does not generalize to sequences longer than those seen in training. **Relative positional encodings** [shaw2018self] and **rotary positional encodings (RoPE)** [su2021roformer] encode position through the relative position of queries and keys, and have become the default in modern large language models.

## The transformer block

A transformer block consists of multi-head self-attention followed by a position-wise feedforward network (FFN), each with residual connections and layer normalization:

$$
\begin{aligned}
\mathbf{X}' &= \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X})) \\
\mathbf{X}'' &= \text{LayerNorm}(\mathbf{X}' + \text{FFN}(\mathbf{X}')),
\end{aligned}
$$

where $\text{FFN}(\mathbf{X}) = \text{ReLU}(\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$ is a two-layer MLP applied independently to each position. Modern variants (LLaMA, GPT-4) use SwiGLU or GeGLU activations and pre-norm rather than post-norm.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, p_drop=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(p_drop)
    def forward(self, x, mask=None):
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) / (self.d_head ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.norm1(x + self.drop(self.o_proj(out)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x
```

## Encoder, decoder, encoder-decoder

The original transformer [vaswani2017attention] was an **encoder-decoder** architecture for sequence-to-sequence tasks: the encoder processed the source sequence, the decoder generated the target sequence auto-regressively, with **cross-attention** allowing the decoder to attend to the encoder's output.

Three architectural lineages descend from this:

**Encoder-only** (BERT [devlin2019bert]). Bidirectional attention over the input. Trained with masked language modeling. Best for tasks requiring understanding of the full input (classification, span extraction).

**Decoder-only** (GPT [radford2018improving]; [brown2020language]). Causal (masked) attention: each position can attend only to earlier positions. Trained with next-token prediction. Best for generation; the architecture of modern large language models.

**Encoder-decoder** (T5 [raffel2020exploring]; BART [lewis2019bart]). The original architecture, retained for tasks where explicit source-target structure is useful (translation, summarization).

## Efficient attention

The $O(n^2)$ cost of self-attention is the principal computational bottleneck of transformers, especially for long sequences. Three families of approximations have been developed.

**Sparse attention**. Restrict attention to a subset of position pairs (local window, strided, or learned). [child2019generating] showed that $O(n \sqrt{n})$ sparse attention is sufficient for image generation.

**Linear attention**. Replace the softmax with a kernel feature map $\phi$ such that $\text{Attn} = (\phi(\mathbf{Q}) \phi(\mathbf{K})^\top) \mathbf{V} = \phi(\mathbf{Q}) (\phi(\mathbf{K})^\top \mathbf{V})$, reducing the cost from $O(n^2 d_h)$ to $O(n d_h^2)$. [katharopoulos2020linear] is a representative example.

**Low-rank attention**. Approximate the attention matrix by a low-rank factorization. [wang2020linformer] projects the key and value matrices to a fixed dimension.

A different line of work avoids approximating attention and instead computes it exactly with $O(n)$ memory via **flash attention** [dao2022flashattention], which exploits the structure of GPU memory hierarchies to compute exact attention with minimal memory traffic.

## Computational experiments

### Experiment 1: Attention patterns

Train a small transformer on a simple algorithmic task (e.g., sorting a sequence of numbers). Visualize the attention weights of each head. Are the patterns interpretable?

### Experiment 2: Positional encoding comparison

Train a small transformer with sinusoidal, learned absolute, and rotary positional encodings on a sequence classification task. Compare generalization to longer sequences than seen in training.

### Experiment 3: Scaling

Train transformers of varying size (depth, width, parameter count) on a language modeling task. Plot training loss as a function of parameter count. Verify the scaling laws of Chapter 9 (Volume III).

## Historical notes

Self-attention was introduced to neural machine translation by [bahdanau2015neural] as an additive attention mechanism over encoder hidden states. The fully attention-based transformer is due to [vaswani2017attention], who showed that attention alone (without recurrence) is sufficient for state-of-the-art translation. BERT [devlin2019bert] demonstrated the power of bidirectional pretraining. GPT-2 [radford2019language] and GPT-3 [brown2020language] demonstrated the scaling behavior of decoder-only language models. The scaling laws of [kaplan2020scaling] and [hoffmann2022training] (Volume III, Chapter 4) provided the empirical foundation for the subsequent scale-up. Flash attention [dao2022flashattention] made exact long-sequence attention practical.

## Exercises

1. **(Easy)** Show that self-attention is permutation equivariant. Why does this necessitate positional encodings?

2. **(Easy)** Compute the parameter count and FLOPs of a transformer block with $d_{\text{model}} = 512$, $H = 8$ heads, $d_{\text{ff}} = 2048$, sequence length $n = 1024$.

3. **(Medium)** Implement sinusoidal, learned, and rotary positional encodings. Train a small transformer with each on a sequence classification task. Compare.

4. **(Medium)** Implement multi-head attention without using `nn.MultiheadAttention`. Verify that your implementation produces the same output as PyTorch's.

5. **(Medium)** Implement masked (causal) attention for autoregressive language modeling. Train a small GPT-like model on a text dataset.

6. **(Hard)** Implement linear attention. Compare its accuracy and runtime to standard softmax attention on a long-sequence task.

7. **(Hard)** Implement flash attention. Verify that it produces the same output as standard attention, with lower memory usage.

8. **(Research)** Investigate the attention patterns of a pretrained model (e.g., GPT-2 small) on natural text. Are specific heads specialized to specific syntactic or semantic phenomena?

## Bibliography
