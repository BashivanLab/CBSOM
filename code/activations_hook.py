"""
Library for extracting interesting quantites from autograd, see README.md

Not thread-safe because of module-level variables

Notation:
o: number of output classes (exact Hessian), number of Hessian samples (sampled Hessian)
n: batch-size
do: output dimension (output channels for convolution)
di: input dimension (input channels for convolution)
Hi: per-example Hessian of matmul, shaped as matrix of [dim, dim], indices have been row-vectorized
Hi_bias: per-example Hessian of bias
Oh, Ow: output height, output width (convolution)
Kh, Kw: kernel height, kernel width (convolution)

Jb: batch output Jacobian of matmul, output sensitivity for example,class pair, [o, n, ....]
Jb_bias: as above, but for bias

A, activations: inputs into current layer
B, backprops: backprop values (aka Lop aka Jacobian-vector product) observed at current layer

"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

_supported_layers = ['Conv2d']  # Supported layer class types
_hooks_disabled: bool = False           # work-around for https://github.com/pytorch/pytorch/issues/25723
_enforce_fresh_backprop: bool = False   # global switch to catch double backprop errors on Hessian computation


def add_hooks(model: nn.Module) -> None:
    """
    Adds hooks to model to save activations and backprop values.

    The hooks will
    1. save activations into param.activations during forward pass
    2. append backprops to params.backprops_list during backward pass.

    Call "remove_hooks(model)" to disable this.

    Args:
        model:
    """

    global _hooks_disabled
    _hooks_disabled = False
    
    handles = []
    for layer in model.modules():
        if _layer_type(layer) in _supported_layers:
            handles.append(layer.register_forward_hook(_capture_activations))

    model.__dict__.setdefault('autograd_hacks_hooks', []).extend(handles)


def is_supported(layer: nn.Module) -> bool:
    """Check if this layer is supported"""

    return _layer_type(layer) in _supported_layers


def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__


def _capture_activations(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""
    if _hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, "Hook installed on unsupported layer, this shouldn't happen"
    
    #setattr(layer, "post_activations", output[0])
    setattr(layer, "post_activations", output.mean(0))

    
