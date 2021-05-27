# Stand-Alone-Self-Attention-in-Vision-Models
This is a pytorch implementation of the paper [Stand-Alone Self-Attention in Vision Models](https://arxiv.org/abs/1906.05909 "Stand-Alone Self-Attention in Vision Models")  by Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya and Jonathon Shlens.

## Method
The paper implements the attention mechanism into different ResNet architectures. The used formula for Single-headed attention to compute the output for one pixel is

![formula sasa](https://user-images.githubusercontent.com/19909320/119891117-ec644480-bf38-11eb-84c0-6e65b4104573.png)

Global Self-Attention on images is subject to the problem, that it can only be applied after a significant
spatial downsampling of the input. Every pixels relation is calculated to every other pixel so learning gets computationally very expensive, which prevents its usage across all layers in a fully attentional model.

To migitate this issue the authors introduce a local attention mechanism, that applies attention only on a first extracted local region of pixels with spatial extent k centered around the current pixel.

![SASA](https://user-images.githubusercontent.com/19909320/119891131-f128f880-bf38-11eb-95e7-8c66a705f8e1.png)


## Example
```python 

```
## Requirements
- pytorch
