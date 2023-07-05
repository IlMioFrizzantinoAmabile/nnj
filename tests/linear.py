"""
Test linear
"""

import numpy as np
import torch
import torch.nn as nn
from flax.core import freeze
from flax.linen import Dense
from jax import numpy as jnp
from jax import vjp, jvp

import nnj


def test_vjp_wrt_weights():
    batch_size = 8
    input_size = 7
    output_size = 3

    # define input
    x = torch.randn(batch_size, input_size)
    jax_x = jnp.array(x.numpy())

    vector = torch.randn(batch_size, output_size)
    jax_vector = jnp.array(vector.numpy())

    # define nnj layer
    nnj_layer = nnj.Linear(input_size, output_size)

    # define flax layer
    params = freeze(
        {
            "params": {
                "bias": jnp.array(nnj_layer.bias.detach().numpy()),
                "kernel": jnp.array(nnj_layer.weight.detach().numpy().T),
            }
        }
    )

    flax_layer = Dense(output_size)
    _, vjp_fun = vjp(lambda params: flax_layer.apply(params, jax_x), params)

    # compare outputs
    jax_vj = vjp_fun(jax_vector)
    jax_out = np.concatenate(
        [jax_vj[0]["params"]["kernel"].T.flatten(), jax_vj[0]["params"]["bias"].flatten()]
    )
    nnj_vj = nnj_layer._vjp(x, None, vector, wrt="weight")

    diff = np.array(jax_out) - nnj_vj.sum(0).detach().numpy()
    assert np.max(abs(diff)) < 1e-4


def test_vjp_wrt_input():
    batch_size = 3
    input_size = 5
    output_size = 6

    # define input
    x = torch.randn(batch_size, input_size)
    jax_x = jnp.array(x.numpy())

    vector = torch.randn(batch_size, output_size)
    jax_vector = jnp.array(vector.numpy())

    # define nnj layer
    nnj_layer = nnj.Linear(input_size, output_size)

    # define flax layer
    params = freeze(
        {
            "params": {
                "bias": jnp.array(nnj_layer.bias.detach().numpy()),
                "kernel": jnp.array(nnj_layer.weight.detach().numpy().T),
            }
        }
    )

    flax_layer = Dense(output_size)
    _, vjp_fun = vjp(lambda data: flax_layer.apply(params, data), jax_x)

    # compare outputs
    jax_vj = vjp_fun(jax_vector)[0]
    nnj_vj = nnj_layer._vjp(x, None, vector, wrt="input")

    diff = np.array(jax_vj) - nnj_vj.detach().numpy()
    assert np.max(abs(diff)) < 1e-4


def test_jvp_wrt_weights():
    batch_size = 8
    input_size = 7
    output_size = 3

    # define input
    x = torch.randn(batch_size, input_size)
    jax_x = jnp.array(x.numpy())

    vector = torch.randn(batch_size, output_size)
    jax_vector = jnp.array(vector.numpy())

    # define nnj layer
    nnj_layer = nnj.Linear(input_size, output_size)

    # define flax layer
    params = freeze(
        {
            "params": {
                "bias": jnp.array(nnj_layer.bias.detach().numpy()),
                "kernel": jnp.array(nnj_layer.weight.detach().numpy().T),
            }
        }
    )

    flax_layer = Dense(output_size)
    _, jvp_fun = jvp(lambda params: flax_layer.apply(params, jax_x), params)

    # compare outputs
    jax_jv = jvp_fun(jax_vector)
    jax_out = np.concatenate(
        [jax_jv[0]["params"]["kernel"].T.flatten(), jax_jv[0]["params"]["bias"].flatten()]
    )
    nnj_jv = nnj_layer._jvp(x, None, vector, wrt="weight")

    diff = np.array(jax_out) - nnj_jv.sum(0).detach().numpy()
    assert np.max(abs(diff)) < 1e-4


def test_jvp_wrt_input():
    batch_size = 3
    input_size = 5
    output_size = 6

    # define input
    x = torch.randn(batch_size, input_size)
    jax_x = jnp.array(x.numpy())

    vector = torch.randn(batch_size, output_size)
    jax_vector = jnp.array(vector.numpy())

    # define nnj layer
    nnj_layer = nnj.Linear(input_size, output_size)

    # define flax layer
    params = freeze(
        {
            "params": {
                "bias": jnp.array(nnj_layer.bias.detach().numpy()),
                "kernel": jnp.array(nnj_layer.weight.detach().numpy().T),
            }
        }
    )

    flax_layer = Dense(output_size)
    _, jvp_fun = jvp(lambda data: flax_layer.apply(params, data), jax_x)

    # compare outputs
    jax_jv = jvp_fun(jax_vector)[0]
    nnj_jv = nnj_layer._jvp(x, None, vector, wrt="input")

    diff = np.array(jax_jv) - nnj_jv.detach().numpy()
    assert np.max(abs(diff)) < 1e-4


def test_linear():
    # test vjp
    test_vjp_wrt_weights()
    test_vjp_wrt_input()

    # test jvp
    test_jvp_wrt_weights()
    test_jvp_wrt_input()


if __name__ == "__main__":
    test_linear()
