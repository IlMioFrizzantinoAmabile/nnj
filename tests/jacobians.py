"""
Test jacobians of each layer wrt input and weights
"""


def test_jacobian_wrt_input(layer):
    return True


def test_jacobian_wrt_weight(layer):
    return True


if __name__ == "__main__":
    # get all the layers

    # for layer in layers:
    test_jacobian_wrt_input("layer")
    test_jacobian_wrt_weight("layer")
