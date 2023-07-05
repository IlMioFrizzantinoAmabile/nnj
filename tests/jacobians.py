"""
Test jacobians of each layer wrt input and weights
"""

import numpy as np
import torch
import torch.nn as nn
from flax.core import freeze
from flax.linen import Dense
from jax import numpy as jnp
from jax import vjp, jvp

import nnj


def test_jacobian_wrt_input(layer):
    return True


def test_jacobian_wrt_weight(layer):
    return True


if __name__ == "__main__":
    # get all the layers

    for layer in layers:
        test_jacobian_wrt_input(layer)
        test_jacobian_wrt_weight(layer)
