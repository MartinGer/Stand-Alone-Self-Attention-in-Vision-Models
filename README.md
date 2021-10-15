# Stand-Alone-Self-Attention-in-Vision-Models
This is a pytorch implementation of the paper [Stand-Alone Self-Attention in Vision Models](https://arxiv.org/abs/1906.05909 "Stand-Alone Self-Attention in Vision Models")  by Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya and Jonathon Shlens.

## Method
The paper implements the attention mechanism into different ResNet architectures. The used formula for Single-headed attention to compute the output for one pixel is

![formula sasa](https://user-images.githubusercontent.com/19909320/119891117-ec644480-bf38-11eb-84c0-6e65b4104573.png)

Global Self-Attention on images is subject to the problem, that it can only be applied after a significant
spatial downsampling of the input. Every pixels relation is calculated to every other pixel so learning gets computationally very expensive, which prevents its usage across all layers in a fully attentional model.

To migitate this issue the authors introduce a local attention mechanism, that applies attention only on a first extracted local region of pixels with spatial extent k centered around the current pixel.

![SASA](https://user-images.githubusercontent.com/19909320/119891131-f128f880-bf38-11eb-95e7-8c66a705f8e1.png)

![SASA](https://user-images.githubusercontent.com/19909320/137499552-3bdf3189-7f57-4f95-a85e-8d5dd2ef6fd0.png)

## Implementation details
I only tested the implementation with ResNet50 for now. The used ResNet V1.5 architectures are adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

#### Additional Parameters:
- attention: ResNet stages in which you would like to apply the attention layers
- num_heads: Number of attention heads
- kernel_size: Size of the receptive field of the local attention algorithm
- inference: Allows to inspect the attention weights of a trained model

## Example
See the jupyter notebook or the example training script

## Requirements
- pytorch
- I use [fast.ai](https://www.fast.ai/) and the [imagenette](https://github.com/fastai/imagenette) dataset for the examples
