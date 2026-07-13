# The Practice of Training

*Volume II · Chapter 10*

---

This chapter treats the engineering of large-scale training: the design decisions that, while less theoretically clean than the material of preceding chapters, are essential for actually training modern deep networks. The chapter is organized around the training loop: initialization, optimization, normalization, regularization, hyperparameter selection, distributed training, and the monitoring of training runs. Each section states the canonical practice, the empirical justification, and the failure modes.

## Initialization revisited

The initialization schemes of Chapter 5 (Xavier, He) were derived for MLPs under simplifying assumptions. Modern architectures require modifications:

- **Transformer initialization** typically uses Xavier with a small additional scaling factor on the deeper layers (e.g., $\sqrt{2/L}$ scaling), or initializes the output layer of each residual block to zero, so that the network is initially the identity. The latter was introduced by [gpt2paper] and is now standard.
- **Embedding initialization** is typically $\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})$, with optional truncation.
- **Output projection** is often initialized to zero or to a small constant, to ensure the initial loss matches the uniform-distribution baseline.

## Optimizer selection

AdamW [loshchilov2019decoupled] — Adam with decoupled weight decay — is the dominant optimizer for transformer training. The decoupling separates the gradient update from the weight decay, which empirically improves generalization. The canonical hyperparameters are $\beta_1 = 0.9$, $\beta_2 = 0.95$–$0.999$ (lower for large models), $\varepsilon = 10^{-8}$, weight decay $0.1$, and a learning rate that is warmup-then-cosine-decay.

The **learning rate schedule** is consequential. The standard schedule is a linear warmup over $N_{\text{warmup}}$ steps (typically $0.1$–$1\%$ of total steps), followed by a cosine decay to $10$–$0\%$ of the peak learning rate. The peak learning rate is set by scaling laws: for a model of size $N$ trained on a dataset of size $D$ with batch size $B$, the optimal peak learning rate scales roughly as $\min(N, D)^{-0.5}$.

```python
import torch

def get_lr(step, warmup, n_total, lr_max):
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, n_total - warmup)
    return 0.1 * lr_max + 0.45 * lr_max * (1 + math.cos(math.pi * progress))
```

## Normalization revisited

Layer normalization (Chapter 5) is the default in transformers. The choice of **pre-norm vs. post-norm** — whether the normalization is applied before or after the residual branch — has implications for stability. Post-norm (the original transformer) is harder to train at scale; pre-norm is now standard. **RMSNorm** [zhang2019rootmean], which drops the mean-subtraction and only rescales by the RMS, is a common simplification used in LLaMA and other modern transformers.

## Regularization

In addition to weight decay, modern training uses several regularizers:

- **Dropout** remains useful in non-attention layers, though it has been displaced by data augmentation and scale in many transformer setups.
- **Label smoothing** [szegedy2016rethinking] replaces the one-hot target distribution with a smoothed distribution ($y_i = (1-\alpha) + \alpha/K$ for the correct class, $y_i = \alpha/K$ for others). It acts as a calibrator and is universal in modern training.
- **Stochastic depth** [huang2016deep] randomly drops entire residual blocks during training, providing both regularization and computational savings.
- **Data augmentation** is the single most impactful regularizer for vision. For text, augmentation is harder; paraphrasing and back-translation are common.

## Distributed training

Training at the scale of modern large language models (LLaMA-3 405B, GPT-4) requires thousands of GPUs. The principal strategies are:

**Data parallelism (DP)**. Each GPU holds a complete copy of the model; each processes a different mini-batch; gradients are averaged across GPUs. Limited by the requirement that the model fits in a single GPU's memory.

**Tensor parallelism (TP)**. The model's weight matrices are split across GPUs; each GPU computes a slice of each matrix multiplication; intermediate results are synchronized by all-reduce. Limited by communication overhead.

**Pipeline parallelism (PP)**. The model is split layer-wise across GPUs; activations are passed from one GPU to the next as in a pipeline. Limited by "bubbles" where GPUs wait for the next batch to arrive.

**Fully sharded data parallelism (FSDP)** [zero]. Each GPU holds a shard of the model's parameters, gradients, and optimizer states; before each forward/backward pass, the relevant shard is all-gathered, used, and discarded. Combines the memory efficiency of model parallelism with the simplicity of data parallelism.

In practice, modern large-scale training uses a 3D combination: data parallelism across nodes, tensor parallelism within nodes, and pipeline parallelism across model depth. The exact recipe is a function of the model size, the cluster topology, and the desired throughput.

## Monitoring training

The principal indicators of healthy training are:

- **Loss curve**: smooth decay, no divergence, no plateau (unless the model has converged).
- **Gradient norm**: stable; sudden spikes indicate instability.
- **Activation statistics**: mean near 0, variance near 1 after each normalization layer.
- **Attention entropies**: distribution of attention weights across heads; should not collapse.
- **Validation loss**: should track training loss; divergence indicates overfitting.
- **Learning rate**: actual vs. scheduled, to verify the schedule is correctly applied.

The most common pathologies are: loss divergence (learning rate too high, gradient clipping too lax), loss plateau (learning rate too low, optimizer stuck), gradient explosion (gradient clipping too lax, numerical instability), and overfitting (training loss continues to decrease while validation loss increases). Each has standard fixes that should be applied systematically.

## Computational experiments

### Experiment 1: Learning rate sweep

Train a small transformer on a language modeling task with a sweep of peak learning rates ($10^{-4}$, $10^{-3}$, $10^{-2}$). Plot training loss curves. Identify the stable range.

### Experiment 2: Weight decay ablation

Train the same model with weight decay $\in \{0, 0.01, 0.1, 1.0\}$. Plot training and validation loss. Identify the regularization effect.

### Experiment 3: Distributed training

Use PyTorch's `DistributedDataParallel` to train a small model on multiple GPUs (or simulate on a single GPU). Measure the throughput scaling as a function of GPU count.

## Historical notes

The canonical practice of training large transformers was developed in the GPT [radford2018improving], [radford2019language], [brown2020language] line of work, refined by LLaMA [touvron2023llama] and successors. AdamW is due to [loshchilov2019decoupled]. RMSNorm to [zhang2019rootmean]. Stochastic depth to [huang2016deep]. The FSDP strategy is due to [zero]. The 3D parallelism recipe is documented in the training infrastructure papers of Megatron-LM [shoeybi2019megatron] and GPT-3 [brown2020language]. Flash attention [dao2022flashattention] is essential for efficient long-context training.

## Exercises

1. **(Easy)** Derive the cosine learning rate schedule. What is the gradient at the start and end of the schedule?

2. **(Easy)** Show that label smoothing with $\alpha$ corresponds to a KL regularization toward the uniform distribution with weight $\alpha$.

3. **(Medium)** Implement AdamW from scratch. Verify that it matches PyTorch's implementation on a synthetic problem.

4. **(Medium)** Implement RMSNorm. Compare its training dynamics to LayerNorm on a small transformer.

5. **(Medium)** Implement stochastic depth. Train a ResNet with and without it; compare test accuracy.

6. **(Hard)** Implement data parallelism from scratch (using `torch.distributed`). Verify that gradients are correctly averaged.

7. **(Hard)** Implement tensor parallelism for a transformer block. Measure the communication overhead as a function of model size.

8. **(Research)** Investigate the training stability of large models. What goes wrong, and how can it be mitigated? (See [mosbach2021stability], [chowdhery2022palm] for starting points.)

## Bibliography
