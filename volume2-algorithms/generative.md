# Generative Models

*Volume II · Chapter 9*

---

A generative model is one that learns to produce samples from a data distribution $p_{\text{data}}$. The three principal families — variational autoencoders (VAEs), generative adversarial networks (GANs), and diffusion models — represent three different solutions to the same fundamental problem: how to learn a probability distribution from samples, when the distribution is supported on a high-dimensional space (e.g., $\mathbb{R}^{3 \times 256 \times 256}$ for images) and is accessible only through samples.

The three families differ in what they explicitly represent. VAEs learn an explicit **encoder** and **decoder** that map between data and a latent space; the density model is implicit but the latent representation is explicit. GANs learn a **generator** that maps noise to data; the density model is implicit and no encoder is learned. Diffusion models learn the **reverse of a noise-adding process**; the density model is explicit (up to a normalizing constant) and a score function is learned.

## Variational autoencoders

The VAE [kingma2013auto]; [rezende2014stochastic] combines an **encoder** $q_\phi(\mathbf{z} | \mathbf{x})$ mapping data to a latent distribution, a **decoder** $p_\theta(\mathbf{x} | \mathbf{z})$ mapping latents to data, and a **prior** $p(\mathbf{z})$ over the latent space (typically $\mathcal{N}(\mathbf{0}, \mathbf{I})$). Training maximizes the **evidence lower bound** (ELBO; see Chapter 4):

$$
\mathcal{L}_{\text{VAE}}(\theta, \phi) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})).
$$

The first term rewards the decoder for reconstructing the data from the latent; the second term regularizes the encoder's posterior toward the prior. The reparameterization trick makes the objective differentiable with respect to both $\theta$ and $\phi$.

VAEs are attractive for the explicit, structured latent space they learn: latents can be interpolated, disentangled, and conditioned on. Their principal drawback is **posterior collapse**: the encoder may ignore the input and match the prior, in which case the decoder learns a marginal model and samples lack diversity. Various mitigations (KL annealing, free bits, stronger decoders) have been proposed.

## Generative adversarial networks

The GAN [NIPS2014_5dce8a06] formulates generation as a two-player game between a **generator** $G_\theta$ (mapping noise $\mathbf{z} \sim p(\mathbf{z})$ to data) and a **discriminator** $D_\psi$ (classifying real vs. fake). The generator tries to fool the discriminator; the discriminator tries to detect fakes. The minimax objective is

$$
\min_\theta \max_\psi \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D_\psi(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1 - D_\psi(G_\theta(\mathbf{z})))].
$$

At the Nash equilibrium, the generator produces samples indistinguishable from the data and the discriminator outputs $1/2$ everywhere. In practice, GANs produce sharper samples than VAEs but are notoriously hard to train: the minimax game is unstable, and the equilibrium is rarely reached cleanly. Mode collapse (the generator produces only a subset of the data modes) is the principal failure mode.

The empirical state of GAN training was substantially improved by **Wasserstein GAN** [arjovsky2017wasserstein], which replaces the Jensen-Shannon divergence with the Wasserstein-1 distance, and by **style-based architectures** [karras2019style], which substantially improved sample quality.

## Diffusion models

A diffusion model [sohl2015deep]; [ho2020denoising] defines a **forward process** that gradually adds Gaussian noise to the data over $T$ steps:

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}),
$$

where $\beta_1, \ldots, \beta_T$ is a variance schedule. The forward process has a closed form:

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I}), \quad \bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s).
$$

A neural network $\epsilon_\theta(\mathbf{x}_t, t)$ is trained to predict the noise added at step $t$, by minimizing

$$
\mathcal{L}_{\text{diffusion}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t) \|^2 \right],
$$

where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Sampling reverses the forward process via a Markov chain that starts from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and iteratively denoises:

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{1 - \beta_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z},
$$

with $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. The score-based perspective [song2021scorebased] unifies diffusion models with score matching, showing that the network learns the **score** $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ of the noise-perturbed data distribution.

Diffusion models have become the dominant paradigm for image generation since 2020, owing to their stable training (a simple squared-error objective) and high sample quality. Their principal drawback is the slow sampling: $T \sim 1000$ denoising steps are required for high-quality samples, though distillation and fast samplers reduce this to $4$–$10$ steps with modest quality loss.

```python
import torch

[torch].no_grad()
def ddpm_sample(model, shape, betas, alphas_cumprod, device='cpu'):
    """Reverse diffusion process: sample x_0 from noise x_T."""
    T = len(betas)
    x = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        beta_t = betas[t]
        alpha_bar_t = alphas_cumprod[t]
        eps_pred = model(x, torch.full((shape[0],), t, device=device))
        mean = (1 / (1 - beta_t).sqrt()) * (x - (beta_t / (1 - alpha_bar_t).sqrt()) * eps_pred)
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + betas[t].sqrt() * noise
        else:
            x = mean
    return x
```

## Computational experiments

### Experiment 1: VAE on MNIST

Train a VAE on MNIST. Plot reconstructions and samples. Vary the KL weight and observe the reconstruction-vs-regularization trade-off.

### Experiment 2: GAN on a small dataset

Train a DCGAN on a small image dataset (e.g., CIFAR-10). Plot samples and discriminator/generator loss curves. Observe mode collapse if it occurs.

### Experiment 3: Diffusion on a 2D toy

Train a small diffusion model on a 2D toy distribution (e.g., a mixture of Gaussians). Visualize the forward process and the reverse sampling trajectory.

## Historical notes

The VAE was introduced by [kingma2013auto] and [rezende2014stochastic] in 2013–2014. The GAN is due to [NIPS2014_5dce8a06]. Diffusion models were introduced by [sohl2015deep] and brought to maturity by [ho2020denoising]. The score-based unification is due to [song2021scorebased]. The combination of diffusion models with classifier-free guidance [ho2022classifier] launched the era of high-quality text-to-image generation.

## Exercises

1. **(Easy)** Derive the VAE objective (ELBO) from the marginal log-likelihood.

2. **(Easy)** Show that the GAN objective is the Jensen-Shannon divergence at the optimal discriminator.

3. **(Medium)** Implement a VAE on MNIST. Investigate posterior collapse: under what conditions does the encoder ignore the input?

4. **(Medium)** Implement a DCGAN. Train on CIFAR-10. Investigate the effect of the discriminator-to-generator update ratio.

5. **(Medium)** Implement a small DDPM. Train on MNIST. Investigate the effect of the number of timesteps $T$.

6. **(Hard)** Implement Wasserstein GAN with gradient penalty. Compare to vanilla GAN in terms of training stability and sample quality.

7. **(Hard)** Implement classifier-free guidance for a class-conditional diffusion model. Quantify the effect of the guidance scale on sample quality.

8. **(Research)** Investigate the score-based perspective on diffusion models. What does the learned score function look like for a simple 1D toy distribution?

## Bibliography
