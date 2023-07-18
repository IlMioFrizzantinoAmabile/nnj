from typing import List

import torch
import torch.nn as nn

import nnj
from nnj.sequential import Sequential


def convert_to_nnj(sequential: Sequential) -> Sequential:
    model = []
    for layer in sequential:
        if layer.__class__.__name__ == "Linear":
            nnj_layer = getattr(nnj, layer.__class__.__name__)(
                in_features=layer.in_features,
                out_features=layer.out_features,
            )
            nnj_layer.weight.data = layer.weight.data
            nnj_layer.bias.data = layer.bias.data
        elif layer.__class__.__name__ == "Conv2d":
            nnj_layer = getattr(nnj, layer.__class__.__name__)(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
            )
            nnj_layer.weight.data = layer.weight.data
            nnj_layer.bias.data = layer.bias.data
        elif "Pool" in layer.__class__.__name__:
            nnj_layer = getattr(nnj, layer.__class__.__name__)(
                kernel_size=layer.kernel_size,
            )
        else:
            nnj_layer = getattr(nnj, layer.__class__.__name__)()

        model.append(nnj_layer)

    model = nnj.Sequential(*model, add_hooks=True)

    return model


def invert_block_diagonal(matrix):
    if isinstance(matrix, list):
        return [invert_block_diagonal(m) for m in matrix]
    return torch.cholesky_inverse(matrix)
