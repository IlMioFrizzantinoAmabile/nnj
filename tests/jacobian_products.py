"""
Test jacobian products wrt input and weights
"""

import nnj


def test_jvp_wrt_input(layer):
    return True


def test_jvp_wrt_weight(layer):
    return True


def test_vjp_wrt_input(layer):
    return True


def test_vjp_wrt_weight(layer):
    return True


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
    for layer in layers:
        # jacobian vector products
        test_jvp_wrt_input(layer)
        test_jvp_wrt_weight(layer)

        test_vjp_wrt_input(layer)
        test_vjp_wrt_weight(layer)

        # jacobian matrix products
        test_jmp_wrt_input(layer)
        test_jmp_wrt_weight(layer)

        test_mjp_wrt_input(layer)
        test_mjp_wrt_weight(layer)

        # jacobian sandwich products
        test_jmjTp_wrt_input(layer)
        test_jmjTp_wrt_weight(layer)

        test_jTmjp_wrt_input(layer)
        test_jTmjp_wrt_weight(layer)
