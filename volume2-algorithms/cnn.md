# Convolutional Networks

*Volume II · Chapter 6*

---

The convolutional neural network (CNN) is the architecture that ignited the modern deep learning era. By imposing **translation equivariance** through weight sharing and local connectivity, CNNs achieve dramatically better sample efficiency than MLPs on data with spatial structure — images, audio spectrograms, and (with modification) time series. The success of AlexNet [krizhevsky2012imagenet] on ImageNet in 2012 is conventionally dated as the start of the deep learning era; that success was, in retrospect, the convergence of convolutional architectures (known since [lecun1998gradient]'s LeNet), GPU computation, and large labeled datasets.

## Translation equivariance

The defining structural property of a CNN is **translation equivariance**: shifting the input and applying the convolution gives the same result as applying the convolution and then shifting. Formally, if $T_\tau$ is the shift-by-$\tau$ operator, then $T_\tau \circ \text{conv}_\mathbf{w} = \text{conv}_\mathbf{w} \circ T_\tau$.

This equivariance is a property of the convolutional structure, not of the parameters. By using the same kernel $\mathbf{w}$ at every spatial location, the network learns features that detect the same pattern wherever it appears in the input. The savings in parameters, compared to a fully connected layer, is dramatic: a $5 \times 5$ kernel on a $224 \times 224$ image has $25$ parameters per output channel, versus $224^2 \approx 50000$ for a fully connected layer.

For tasks where the *location* of a pattern is irrelevant (image classification), translation equivariance is combined with **pooling** to obtain translation *invariance*. For tasks where location matters (object detection, segmentation), the equivariant representation is preserved through the network.

## The convolution operation

A two-dimensional convolution (technically, a cross-correlation; the machine learning convention differs from the signal processing convention by a sign flip) of input $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ with kernel $\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k \times k}$ produces output $\mathbf{Y} \in \mathbb{R}^{C_{\text{out}} \times H' \times W'}$ with

$$
Y_{c', i, j} = \sum_{c=1}^{C_{\text{in}}} \sum_{u=1}^{k} \sum_{v=1}^{k} W_{c', c, u, v} \, X_{c, i + u - \lfloor k/2 \rfloor, j + v - \lfloor k/2 \rfloor} + b_{c'}.
$$

The hyperparameters are: kernel size $k$, stride $s$ (step size of the kernel), padding $p$ (number of zeros added at the boundary), and dilation $d$ (spacing between kernel elements). The output spatial dimensions are $H' = \lfloor (H + 2p - d(k-1) - 1) / s \rfloor + 1$.

The computational cost of a convolutional layer is $O(C_{\text{out}} C_{\text{in}} k^2 H' W')$. The dominant cost in modern CNNs is usually the $C_{\text{out}} C_{\text{in}}$ factor; the spatial dimensions $H' W'$ shrink through the network via pooling and strided convolutions.

## Canonical architectures

The evolution of CNN architectures tracks the evolution of the field.

**LeNet** [lecun1998gradient] (1998). The original CNN for digit recognition: two convolutional layers followed by fully connected layers. Total parameters: $\sim 60{,}000$.

**AlexNet** [krizhevsky2012imagenet] (2012). The first large-scale CNN trained on GPUs: 5 convolutional layers, 3 fully connected, ReLU activations, dropout, $\sim 60$M parameters. Won the ImageNet 2012 competition by a wide margin and triggered the deep learning era.

**VGG** [simonyan2014very] (2014). Stacked $3 \times 3$ convolutions, demonstrating that depth alone (with small kernels) improves performance. $\sim 138$M parameters; very heavy.

**ResNet** [he2015deep] (2015). Introduced **residual connections** $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$, which bypass the layer's transformation. This addressed the degradation problem (deeper networks were harder to train, paradoxically) and enabled networks of hundreds or thousands of layers. ResNet-50 ($\sim 25$M parameters) remains a workhorse backbone.

**EfficientNet** [tan2019efficientnet] (2019). Systematic compound scaling of width, depth, and resolution. Reached state-of-the-art accuracy with an order of magnitude fewer FLOPs than previous architectures.

**ConvNeXt** [liu2022convnet] (2022). Modernized ResNet design with lessons from vision transformers: large kernels ($7 \times 7$), inverted bottlenecks, layer normalization, GELU activations. Demonstrated that pure convolutional architectures remain competitive with attention-based ones.

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.act   = nn.GELU()
        self.downsample = nn.Identity() if stride == 1 else nn.Sequential(
            nn.Conv2d(channels, channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        identity = self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)
```

## Pooling and invariance

**Pooling** reduces the spatial resolution of the feature map, providing a degree of translation invariance and reducing the computational cost of subsequent layers. Max pooling takes the maximum over a local window; average pooling takes the average. **Global average pooling** averages over the entire spatial extent, producing a single value per channel; it is the standard way to transition from convolutional features to a fixed-length representation for classification.

Modern architectures often replace explicit pooling with **strided convolutions**, which learn the downsampling rather than fixing it. ConvNeXt, for instance, uses strided convolutions throughout.

## Receptive field

The **receptive field** of a unit in a CNN is the region of the input that can affect its value. For a stack of $L$ convolutional layers with kernel size $k$ and stride 1, the receptive field is $1 + L(k-1)$. With stride $s > 1$, the receptive field grows multiplicatively rather than additively.

The receptive field has implications for what the network can compute: a unit with a $5 \times 5$ receptive field cannot detect patterns larger than $5 \times 5$ pixels. Architectures for high-resolution images (semantic segmentation, object detection) typically include dilated convolutions or attention layers to expand the receptive field without increasing parameter count.

## Computational experiments

### Experiment 1: Equivariance check

Construct a small CNN. Feed an image and a shifted version of the same image. Verify numerically that the output shifts correspondingly.

### Experiment 2: Receptive field growth

For ResNet-18, compute the theoretical receptive field of each layer's output. Compare to the empirical receptive field (measured by the gradient of a unit's activation with respect to the input).

### Experiment 3: Modern vs. classical

On CIFAR-10, compare LeNet, VGG-11, ResNet-18, and ConvNeXt-tiny. Plot accuracy, parameter count, and FLOPs. Discuss the trade-offs.

## Historical notes

The convolutional architecture was inspired by the architecture of the visual cortex, as described by [hubel1962receptive]. The Neocognitron of [fukushima1982neocognitron] was an early biologically-inspired model with convolution-like weight sharing. LeNet [lecun1998gradient] was the first modern CNN, used commercially for digit recognition. AlexNet [krizhevsky2012imagenet] launched the deep learning era by demonstrating dramatic improvements on ImageNet. ResNet [he2015deep] enabled very deep networks through residual connections. Vision transformers [dosovitskiy2020image] challenged the dominance of CNNs on vision tasks in 2020; ConvNeXt [liu2022convnet] demonstrated that pure convolutional architectures remain competitive when modernized.

## Exercises

1. **(Easy)** Show that a $3 \times 3$ convolution followed by another $3 \times 3$ convolution has the same receptive field as a single $5 \times 5$ convolution. Compare parameter counts.

2. **(Easy)** Compute the receptive field of a unit in the final layer of a 4-layer CNN with $3 \times 3$ kernels and stride 1.

3. **(Medium)** Implement a dilated convolution. Show empirically that dilated convolutions can expand the receptive field without increasing parameter count.

4. **(Medium)** Implement and train a small ResNet on CIFAR-10. Compare training curves with and without residual connections.

5. **(Medium)** Implement depthwise-separable convolutions (used in MobileNet). Compare parameter count and accuracy to a standard convolution.

6. **(Hard)** Prove that a convolutional layer is translation equivariant. Under what conditions is it translation invariant?

7. **(Hard)** Implement a vision transformer (ViT) and compare its performance to a ResNet on CIFAR-10. How do they scale with dataset size?

8. **(Research)** Investigate the role of the kernel size in modern CNNs. Why did ConvNeXt choose $7 \times 7$ kernels? Are larger kernels (e.g., $31 \times 31$) ever beneficial?

## Bibliography
