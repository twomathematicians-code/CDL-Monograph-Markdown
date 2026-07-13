# Recurrent Networks

*Volume II · Chapter 7*

---

A recurrent neural network (RNN) processes a sequence $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$ by maintaining a hidden state $\mathbf{h}_t$ that is updated at each step:

$$
\mathbf{h}_t = \sigma(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}).
$$

The same weights $\mathbf{W}_h, \mathbf{W}_x$ are used at every timestep — the network is **weight-shared across time** in the same way that a CNN is weight-shared across space. This weight sharing gives the RNN two key properties: (i) it can process sequences of arbitrary length, and (ii) the number of parameters is independent of the sequence length.

## The vanishing and exploding gradient problem

The fundamental difficulty of training RNNs is the **vanishing and exploding gradient problem**. The gradient of the loss with respect to $\mathbf{h}_t$ involves a product of Jacobians:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{k=t+1}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}}.
$$

If the largest eigenvalue of each Jacobian is less than 1, the product vanishes exponentially in $T - t$; if it is greater than 1, the product explodes. The Jacobian's eigenvalues are bounded by the operator norm of $\mathbf{W}_h$ times the Lipschitz constant of $\sigma$; for sigmoid or tanh activations, this is at most $\|\mathbf{W}_h\|$, so initialization must be carefully tuned to keep $\|\mathbf{W}_h\|$ near 1.

#### **Theorem (RNN gradient bounds)**

Consider a linear RNN $\mathbf{h}_t = \mathbf{W} \mathbf{h}_{t-1} + \mathbf{U}\mathbf{x}_t$ with $\|\mathbf{W}\| = \rho$. Then $\|\partial \mathcal{L}/\partial \mathbf{h}_t\| \leq \rho^{T-t} \|\partial \mathcal{L}/\partial \mathbf{h}_T\|$. The gradient vanishes if $\rho < 1$ and explodes if $\rho > 1$.

## LSTM and GRU

The **Long Short-Term Memory** (LSTM) network [hochreiter1997long] addresses the vanishing gradient problem by introducing a **cell state** $\mathbf{c}_t$ that is updated additively rather than multiplicatively, and three **gates** (input, forget, output) that control the flow of information:

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) & \text{(forget gate)} \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) & \text{(input gate)} \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) & \text{(candidate)} \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t & \text{(cell update)} \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) & \text{(output gate)} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) & \text{(hidden state)}
\end{aligned}
$$

The forget gate $\mathbf{f}_t$ controls what is retained from the previous cell state; the input gate $\mathbf{i}_t$ controls what is written; the output gate $\mathbf{o}_t$ controls what is read out. The additive cell update is the key: the gradient flows through the cell state without multiplication by a Jacobian, addressing the vanishing gradient problem (at least in the limit of forget gate = 1).

The **Gated Recurrent Unit** (GRU) [cho2014learning] is a simplified variant with two gates (update, reset) and no separate cell state:

$$
\begin{aligned}
\mathbf{z}_t &= \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t]) & \text{(update gate)} \\
\mathbf{r}_t &= \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t]) & \text{(reset gate)} \\
\tilde{\mathbf{h}}_t &= \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t]) \\
\mathbf{h}_t &= (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t.
\end{aligned}
$$

GRU has fewer parameters than LSTM and is often competitive in practice.

## Connection to dynamical systems

A continuous-time RNN is a dynamical system $\dot{\mathbf{h}}(t) = -\mathbf{h}(t) + \sigma(\mathbf{W}\mathbf{h}(t) + \mathbf{U}\mathbf{x}(t))$, discretized by an Euler scheme with step 1. This perspective connects RNNs to the rich theory of dynamical systems: fixed points, limit cycles, bifurcations, and chaos all have analogs in the behavior of trained RNNs. [sussillo2013opening] exploited this perspective to stabilize RNN training by initializing near the "edge of chaos."

The **echo state property** [jaeger2001echo] characterizes when an untrained random RNN (with only the readout trained) can approximate any sequence-to-sequence map: roughly, the network must be contracting but not too contracting, with spectral radius of $\mathbf{W}_h$ near 1.

## Computational experiments

### Experiment 1: Vanishing gradient demonstration

Train a vanilla RNN and an LSTM on a synthetic task requiring long-range dependencies (e.g., "copy the first token of a length-50 sequence to the end"). Plot the gradient norm at the first timestep as a function of sequence length. The vanilla RNN's gradient should vanish; the LSTM's should not.

### Experiment 2: LSTM on character-level language modeling

Train an LSTM on a small text dataset (e.g., Shakespeare). Plot training and validation loss as a function of epoch and sequence length. Generate samples from the trained model.

### Experiment 3: GRU vs. LSTM

On a sequence classification task (e.g., MNIST as a 784-step sequence), compare LSTM and GRU in terms of accuracy, training time, and parameter count.

## Historical notes

The RNN has its origins in the work of [rumelhart1986learning] on backpropagation through time. The vanishing gradient problem was identified by [hochreiter1991untersuchungen], leading directly to the LSTM [hochreiter1997long]. The GRU was introduced by [cho2014learning] as a simplified alternative. The connection to dynamical systems has been explored by [sussillo2013opening] and [poole2016exponential], among others.

The dominance of recurrent networks in sequence modeling ended with the introduction of the transformer [vaswani2017attention] in 2017, which we treat in the next chapter. RNNs remain useful in low-resource settings, in real-time streaming applications, and as a baseline against which newer architectures are compared.

## Exercises

1. **(Easy)** Derive backpropagation through time (BPTT) for a vanilla RNN with one hidden layer.

2. **(Easy)** Show that the LSTM's cell state gradient does not suffer from the vanishing gradient problem in the limit of forget gate = 1.

3. **(Medium)** Implement a vanilla RNN and an LSTM from scratch (no PyTorch RNN modules). Compare their behavior on a long-range dependency task.

4. **(Medium)** Implement gradient clipping. Show empirically that it stabilizes RNN training.

5. **(Medium)** Implement bidirectional RNNs and apply to a sequence labeling task (e.g., part-of-speech tagging).

6. **(Hard)** Reproduce the vanishing gradient experiment of [pascanu2013difficulty]: train RNNs of various spectral radii and measure the gradient norm at the first timestep.

7. **(Hard)** Implement an echo state network (random RNN, trained readout). What tasks can it solve? What are its limitations?

8. **(Research)** Investigate modern recurrent architectures (e.g., Mamba, RWKV) that aim to combine the benefits of RNNs (constant memory) with the performance of transformers. What inductive biases do they impose?

## Bibliography
