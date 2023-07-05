"""
Test jacobian products wrt input and weights
"""

import torch

import nnj


def test_jvp_wrt_input(layer, input_shape):

    x = torch.randn(*input_shape)
    tangent_vector_input = torch.randn(*input_shape)

    jacobian = layer._jacobian(x, None, wrt="input")
    jvp_slow = torch.einsum("bij, bj -> bi", jacobian, tangent_vector_input)
    jvp_fast = layer._jvp(x, None, tangent_vector_input, wrt="input")

    assert jvp_fast.shape == jvp_slow.shape
    assert torch.isclose(jvp_fast, jvp_slow).all()


def test_jvp_wrt_weight(layer, input_shape):

    x = torch.randn(*input_shape)
    batch_size = input_shape[0]
    tangent_vector_params = torch.randn((batch_size, layer._n_params))

    jvp_fast = layer._jvp(x, None, tangent_vector_params, wrt="weight")
    if layer._n_params == 0:
        assert jvp_fast is None
    else:
        jacobian = layer._jacobian(x, None, wrt="weight")
        jvp_slow = torch.einsum("bij, bj -> bi", jacobian, tangent_vector_params)

        assert jvp_fast.shape == jvp_slow.shape
        assert torch.isclose(jvp_fast, jvp_slow).all()


def test_vjp_wrt_input(layer, input_shape):

    x = torch.randn(*input_shape)
    output_shape = layer.forward(x).shape
    tangent_vector_output = torch.randn(*output_shape)

    jacobian = layer._jacobian(x, None, wrt="input")
    vjp_slow = torch.einsum("bi, bij -> bj", tangent_vector_output, jacobian)
    vjp_fast = layer._vjp(x, None, tangent_vector_output, wrt="input")

    assert vjp_fast.shape == vjp_slow.shape
    assert torch.isclose(vjp_fast, vjp_slow).all()


def test_vjp_wrt_weight(layer, input_shape):

    x = torch.randn(*input_shape)
    output_shape = layer.forward(x).shape
    tangent_vector_output = torch.randn(*output_shape)

    vjp_fast = layer._vjp(x, None, tangent_vector_output, wrt="weight")
    if layer._n_params == 0:
        assert vjp_fast is None
    else:
        jacobian = layer._jacobian(x, None, wrt="weight")
        vjp_slow = torch.einsum("bi, bij -> bj", tangent_vector_output, jacobian)

        assert vjp_fast.shape == vjp_slow.shape
        assert torch.isclose(vjp_fast, vjp_slow).all()


def test_jmp_wrt_input(layer):
    return True


def test_jmp_wrt_weight(layer):
    return True


def test_mjp_wrt_input(layer):
    return True


def test_mjp_wrt_weight(layer):
    return True


def test_jmjTp_wrt_input(layer):
    return True


def test_jmjTp_wrt_weight(layer):
    return True


def test_jTmjp_wrt_input(layer):
    return True


def test_jTmjp_wrt_weight(layer):
    return True


if __name__ == "__main__":
    # get all the layers
    for layer, input_shape in [(nnj.Linear(3, 5), (7, 3)), (nnj.Tanh(), (7, 3))]:
        # jacobian vector products
        test_jvp_wrt_input(layer, input_shape)
        test_jvp_wrt_weight(layer, input_shape)

        test_vjp_wrt_input(layer, input_shape)
        test_vjp_wrt_weight(layer, input_shape)
        """
        # jacobian matrix products
        test_jmp_wrt_input(layer)
        test_jmp_wrt_weight(layer)

        test_mjp_wrt_input(layer)
        test_mjp_wrt_weight(layer)

        # jacobian sandwich products
        test_jmjTp_wrt_input(layer)
        test_jmjTp_wrt_weight(layer)

        test_jTmjp_wrt_input(layer)
        test_jTmjp_wrt_weight(layer) """
