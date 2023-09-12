"""
Test that the methods 
.jvp() .vjp() .jmp() .mjp() .jmjTp() .jTmjp()
return the correct tensor

NOTE: this tests assumes that each layer has the correct implementation of .jacobian()
"""

import torch

import nnj

# define some input data
xs = [
    torch.randn(7, 3),
    torch.ones(7, 3),
    torch.randn(7, 3) + torch.ones(7, 3),
    10 * torch.rand(7, 3),
]

# get all the layers
to_test_easy = [
    nnj.Linear(3, 5, bias=False),
    nnj.Linear(3, 5),
    nnj.Tanh(),
    nnj.ReLU(),
    nnj.Sigmoid(),
    nnj.Sinusoidal(),
    nnj.TruncExp(),
    nnj.Softplus(),
]
to_test_advanced = [
    nnj.Sequential(nnj.Linear(3, 5), nnj.Tanh(), nnj.Linear(5, 13)),
    nnj.Sequential(nnj.Linear(3, 5), nnj.ReLU(), nnj.Linear(5, 13)),
    nnj.Sequential(
        nnj.Linear(3, 5),
        nnj.Tanh(),
        nnj.Linear(5, 2),
        nnj.Tanh(),
        nnj.Sinusoidal(),
        nnj.Linear(2, 12),
        nnj.Reshape(3, 4),
        nnj.Tanh(),
        nnj.Reshape(12),
        nnj.Softplus(),
        nnj.Tanh(),
    ),
    nnj.Sequential(
        nnj.Linear(3, 5),
        nnj.Tanh(),
        nnj.Sequential(
            nnj.Linear(5, 5),
            nnj.Tanh(),
            nnj.TruncExp(),
            nnj.Linear(5, 2),
        ),
        nnj.ReLU(),
        nnj.Linear(2, 13),
    ),
    nnj.Sequential(
        nnj.Linear(3, 5),
        nnj.Tanh(),
        nnj.Sequential(
            nnj.Sequential(
                nnj.Linear(5, 5),
                nnj.Tanh(),
                nnj.Linear(5, 3),
            ),
            nnj.Tanh(),
            nnj.Linear(3, 2),
        ),
        nnj.ReLU(),
        nnj.Sequential(
            nnj.Linear(2, 5),
            nnj.Tanh(),
            nnj.Linear(5, 2),
            nnj.TruncExp(),
            nnj.Softplus(),
        ),
        nnj.Linear(2, 13),
    ),
]


###################
# vector products #
###################


def test_jvp_wrt_input():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            tangent_vector_input = torch.randn(*x.shape)

            jacobian = layer.jacobian(x, None, wrt="input")
            jvp_slow = torch.einsum("bij, bj -> bi", jacobian, tangent_vector_input)
            jvp_fast = layer.jvp(x, None, tangent_vector_input, wrt="input")

            assert jvp_fast.shape == jvp_slow.shape
            assert torch.isclose(jvp_fast, jvp_slow, atol=1e-4).all()


def test_jvp_wrt_weight():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            batch_size = x.shape[0]
            tangent_vector_params = torch.randn((batch_size, layer._n_params))

            jvp_fast = layer.jvp(x, None, tangent_vector_params, wrt="weight")
            if layer._n_params == 0:
                assert jvp_fast is None
            else:
                jacobian = layer.jacobian(x, None, wrt="weight")
                jvp_slow = torch.einsum("bij, bj -> bi", jacobian, tangent_vector_params)

                assert jvp_fast.shape == jvp_slow.shape
                assert torch.isclose(jvp_fast, jvp_slow, atol=1e-4).all()


def test_vjp_wrt_input():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            output_shape = layer.forward(x).shape
            tangent_vector_output = torch.randn(*output_shape)

            jacobian = layer.jacobian(x, None, wrt="input")
            vjp_slow = torch.einsum("bi, bij -> bj", tangent_vector_output, jacobian)
            vjp_fast = layer.vjp(x, None, tangent_vector_output, wrt="input")

            assert vjp_fast.shape == vjp_slow.shape
            assert torch.isclose(vjp_fast, vjp_slow, atol=1e-4).all()


def test_vjp_wrt_weight():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            output_shape = layer.forward(x).shape
            tangent_vector_output = torch.randn(*output_shape)

            vjp_fast = layer.vjp(x, None, tangent_vector_output, wrt="weight")
            if layer._n_params == 0:
                assert vjp_fast is None
            else:
                jacobian = layer.jacobian(x, None, wrt="weight")
                vjp_slow = torch.einsum("bi, bij -> bj", tangent_vector_output, jacobian)

                assert vjp_fast.shape == vjp_slow.shape
                assert torch.isclose(vjp_fast, vjp_slow, atol=1e-4).all()


###################
# matrix products #
###################


def test_jmp_wrt_input():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            n_columns = 7
            batch_size = x.shape[0]
            tangent_matrix_input = torch.randn(batch_size, x.shape[1:].numel(), n_columns)

            jacobian = layer.jacobian(x, None, wrt="input")
            jmp_slow = torch.einsum("bij, bjk -> bik", jacobian, tangent_matrix_input)
            jmp_fast = layer.jmp(x, None, tangent_matrix_input, wrt="input")

            assert jmp_fast.shape == jmp_slow.shape
            assert torch.isclose(jmp_fast, jmp_slow, atol=1e-4).all()


def test_jmp_wrt_weight():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            n_columns = 7
            batch_size = x.shape[0]
            tangent_matrix_params = torch.randn((batch_size, layer._n_params, n_columns))

            jmp_fast = layer.jmp(x, None, tangent_matrix_params, wrt="weight")
            if layer._n_params == 0:
                assert jmp_fast is None
            else:
                jacobian = layer.jacobian(x, None, wrt="weight")
                jmp_slow = torch.einsum("bij, bjk -> bik", jacobian, tangent_matrix_params)

                assert jmp_fast.shape == jmp_slow.shape
                assert torch.isclose(jmp_fast, jmp_slow, atol=1e-4).all()


def test_mjp_wrt_input():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            n_rows = 7
            batch_size = x.shape[0]
            output_shape = layer.forward(x).shape
            tangent_matrix_output = torch.randn(batch_size, n_rows, output_shape[1:].numel())

            jacobian = layer.jacobian(x, None, wrt="input")
            mjp_slow = torch.einsum("bij, bjk -> bik", tangent_matrix_output, jacobian)
            mjp_fast = layer.mjp(x, None, tangent_matrix_output, wrt="input")

            assert mjp_fast.shape == mjp_slow.shape
            assert torch.isclose(mjp_fast, mjp_slow, atol=1e-4).all()


def test_mjp_wrt_weight():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            n_rows = 7
            batch_size = x.shape[0]
            output_shape = layer.forward(x).shape
            tangent_matrix_output = torch.randn(batch_size, n_rows, output_shape[1:].numel())

            mjp_fast = layer.mjp(x, None, tangent_matrix_output, wrt="weight")
            if layer._n_params == 0:
                assert mjp_fast is None
            else:
                jacobian = layer.jacobian(x, None, wrt="weight")
                mjp_slow = torch.einsum("bij, bjk -> bik", tangent_matrix_output, jacobian)

                assert mjp_fast.shape == mjp_slow.shape
                assert torch.isclose(mjp_fast, mjp_slow, atol=1e-4).all()


#####################
# sandwich products #
#####################


def test_jmjTp_wrt_input():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            for from_diag in [False, True]:
                for to_diag in [False, True]:
                    batch_size = x.shape[0]
                    jacobian = layer.jacobian(x, None, wrt="input")

                    if from_diag is False:
                        tangent_matrix_input = torch.randn(batch_size, x.shape[1:].numel(), x.shape[1:].numel())

                        if to_diag is False:
                            # full -> full
                            jmjTp_slow = torch.einsum("bij, bjk, bqk -> biq", jacobian, tangent_matrix_input, jacobian)
                            jmjTp_fast = layer.jmjTp(
                                x, None, tangent_matrix_input, wrt="input", from_diag=False, to_diag=False
                            )
                        elif to_diag is True:
                            # full -> diag
                            jmjTp_slow = torch.einsum("bij, bjk, bik -> bi", jacobian, tangent_matrix_input, jacobian)
                            jmjTp_fast = layer.jmjTp(
                                x, None, tangent_matrix_input, wrt="input", from_diag=False, to_diag=True
                            )

                    elif from_diag is True:
                        tangent_diagonal_matrix_input = torch.randn(batch_size, x.shape[1:].numel())

                        if to_diag is False:
                            # diag -> full
                            jmjTp_slow = torch.einsum(
                                "bij, bj, bqj -> biq", jacobian, tangent_diagonal_matrix_input, jacobian
                            )
                            jmjTp_fast = layer.jmjTp(
                                x, None, tangent_diagonal_matrix_input, wrt="input", from_diag=True, to_diag=False
                            )
                        elif to_diag is True:
                            # diag -> diag
                            jmjTp_slow = torch.einsum(
                                "bij, bj, bij -> bi", jacobian, tangent_diagonal_matrix_input, jacobian
                            )
                            jmjTp_fast = layer.jmjTp(
                                x, None, tangent_diagonal_matrix_input, wrt="input", from_diag=True, to_diag=True
                            )

                    assert jmjTp_fast.shape == jmjTp_slow.shape
                    assert torch.isclose(jmjTp_fast, jmjTp_slow, atol=1e-4).all()


def test_jmjTp_wrt_weight():
    # utility functions for sequential layers
    def generate_random_matrix_params(batch_size, module):
        if isinstance(module, nnj.Sequential):
            diagonal_blocks = [
                generate_random_matrix_params(batch_size, submodule) for submodule in module._modules.values()
            ]
            return [d for d in diagonal_blocks if d is not None]
        if module._n_params == 0:
            return None
        return torch.randn(batch_size, module._n_params, module._n_params)

    def embed_block_diagonal(matrix):
        if isinstance(matrix, list):
            embedded_submatrices = [embed_block_diagonal(submatrix) for submatrix in matrix]
            batch_size = embedded_submatrices[0].shape[0]
            n_row = sum([submatrix.shape[1] for submatrix in embedded_submatrices])
            n_col = sum([submatrix.shape[2] for submatrix in embedded_submatrices])
            embedded_matrix = torch.zeros(batch_size, n_row, n_col)
            i_row, i_col = 0, 0
            for submatrix in embedded_submatrices:
                embedded_matrix[:, i_row : i_row + submatrix.shape[1], i_col : i_col + submatrix.shape[2]] = submatrix
                i_row += submatrix.shape[1]
                i_col += submatrix.shape[2]
            return embedded_matrix
        return matrix

    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            for from_diag in [False, True]:
                for to_diag in [False, True]:
                    if layer._n_params == 0:
                        return  # TODO: check that .jmjTp returns None in all cases

                    batch_size = x.shape[0]
                    jacobian = layer.jacobian(x, None, wrt="weight")

                    if from_diag is False:
                        tangent_matrix_params = generate_random_matrix_params(batch_size, layer)

                        if to_diag is False:
                            # full -> full
                            jmjTp_slow = torch.einsum(
                                "bij, bjk, bqk -> biq", jacobian, embed_block_diagonal(tangent_matrix_params), jacobian
                            )
                            jmjTp_fast = layer.jmjTp(
                                x, None, tangent_matrix_params, wrt="weight", from_diag=False, to_diag=False
                            )
                        elif to_diag is True:
                            # full -> diag
                            jmjTp_slow = torch.einsum(
                                "bij, bjk, bik -> bi", jacobian, embed_block_diagonal(tangent_matrix_params), jacobian
                            )
                            jmjTp_fast = layer.jmjTp(
                                x, None, tangent_matrix_params, wrt="weight", from_diag=False, to_diag=True
                            )

                    elif from_diag is True:
                        tangent_diagonal_matrix_params = torch.randn(batch_size, layer._n_params)

                        if to_diag is False:
                            # diag -> full
                            jmjTp_slow = torch.einsum(
                                "bij, bj, bqj -> biq", jacobian, tangent_diagonal_matrix_params, jacobian
                            )
                            jmjTp_fast = layer.jmjTp(
                                x, None, tangent_diagonal_matrix_params, wrt="weight", from_diag=True, to_diag=False
                            )
                        elif to_diag is True:
                            # diag -> diag
                            jmjTp_slow = torch.einsum(
                                "bij, bj, bij -> bi", jacobian, tangent_diagonal_matrix_params, jacobian
                            )
                            jmjTp_fast = layer.jmjTp(
                                x, None, tangent_diagonal_matrix_params, wrt="weight", from_diag=True, to_diag=True
                            )

                    assert jmjTp_fast.shape == jmjTp_slow.shape
                    assert torch.isclose(jmjTp_fast, jmjTp_slow, atol=1e-4).all()


def test_jTmjp_wrt_input():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            for from_diag in [False, True]:
                for to_diag in [False, True]:
                    output_shape = layer.forward(x).shape
                    batch_size = x.shape[0]
                    jacobian = layer.jacobian(x, None, wrt="input")

                    if from_diag is False:
                        tangent_matrix_output = torch.randn(
                            batch_size, output_shape[1:].numel(), output_shape[1:].numel()
                        )

                        if to_diag is False:
                            # full -> full
                            jTmjp_slow = torch.einsum("bji, bjk, bkq -> biq", jacobian, tangent_matrix_output, jacobian)
                            jTmjp_fast = layer.jTmjp(
                                x, None, tangent_matrix_output, wrt="input", from_diag=False, to_diag=False
                            )
                        elif to_diag is True:
                            # full -> diag
                            jTmjp_slow = torch.einsum("bji, bjk, bki -> bi", jacobian, tangent_matrix_output, jacobian)
                            jTmjp_fast = layer.jTmjp(
                                x, None, tangent_matrix_output, wrt="input", from_diag=False, to_diag=True
                            )

                    elif from_diag is True:
                        tangent_diagonal_matrix_output = torch.randn(batch_size, output_shape[1:].numel())

                        if to_diag is False:
                            # diag -> full
                            jTmjp_slow = torch.einsum(
                                "bji, bj, bjk -> bik", jacobian, tangent_diagonal_matrix_output, jacobian
                            )
                            jTmjp_fast = layer.jTmjp(
                                x, None, tangent_diagonal_matrix_output, wrt="input", from_diag=True, to_diag=False
                            )
                        elif to_diag is True:
                            # diag -> diag
                            jTmjp_slow = torch.einsum(
                                "bji, bj, bji -> bi", jacobian, tangent_diagonal_matrix_output, jacobian
                            )
                            jTmjp_fast = layer.jTmjp(
                                x, None, tangent_diagonal_matrix_output, wrt="input", from_diag=True, to_diag=True
                            )

                    assert jTmjp_fast.shape == jTmjp_slow.shape
                    assert torch.isclose(jTmjp_fast, jTmjp_slow, atol=1e-4).all()


def test_jTmjp_wrt_weight():
    # utility functions for sequential layers
    def get_shape_of_block_diagonal(matrix):
        if isinstance(matrix, list):
            n_row = sum([get_shape_of_block_diagonal(submatrix)[0] for submatrix in matrix])
            n_col = sum([get_shape_of_block_diagonal(submatrix)[1] for submatrix in matrix])
            return n_row, n_col
        return matrix.shape[1], matrix.shape[2]

    def is_close_block_diagonal(matrix1, matrix2, atol=1e-4):
        if isinstance(matrix1, list):
            is_close = True
            i_row, i_col = 0, 0
            for submatrix1 in matrix1:
                n_row, n_col = get_shape_of_block_diagonal(submatrix1)
                submatrix2 = matrix2[:, i_row : i_row + n_row, i_col : i_col + n_col]
                is_close = is_close and is_close_block_diagonal(submatrix1, submatrix2, atol=atol)
                i_row += n_row
                i_col += n_col
            return is_close
        return torch.isclose(matrix1, matrix2, atol=atol).all()

    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            for from_diag in [False, True]:
                for to_diag in [False, True]:
                    # if to_diag is False:
                    # TODO: test these cases as well
                    #    continue

                    if layer._n_params == 0:
                        return  # TODO: check that .jmjTp returns None in all cases

                    output_shape = layer.forward(x).shape
                    batch_size = x.shape[0]
                    jacobian = layer.jacobian(x, None, wrt="weight")

                    if from_diag is False:
                        tangent_matrix_output = torch.randn(
                            batch_size, output_shape[1:].numel(), output_shape[1:].numel()
                        )

                        if to_diag is False:
                            # full -> full
                            jTmjp_slow = torch.einsum("bji, bjk, bkq -> biq", jacobian, tangent_matrix_output, jacobian)
                            jTmjp_fast = layer.jTmjp(
                                x, None, tangent_matrix_output, wrt="weight", from_diag=False, to_diag=False
                            )
                        elif to_diag is True:
                            # full -> diag
                            jTmjp_slow = torch.einsum("bji, bjk, bki -> bi", jacobian, tangent_matrix_output, jacobian)
                            jTmjp_fast = layer.jTmjp(
                                x, None, tangent_matrix_output, wrt="weight", from_diag=False, to_diag=True
                            )

                    elif from_diag is True:
                        tangent_diagonal_matrix_output = torch.randn(batch_size, output_shape[1:].numel())

                        if to_diag is False:
                            # diag -> full
                            jTmjp_slow = torch.einsum(
                                "bji, bj, bjk -> bik", jacobian, tangent_diagonal_matrix_output, jacobian
                            )
                            jTmjp_fast = layer.jTmjp(
                                x, None, tangent_diagonal_matrix_output, wrt="weight", from_diag=True, to_diag=False
                            )
                        elif to_diag is True:
                            # diag -> diag
                            jTmjp_slow = torch.einsum(
                                "bji, bj, bji -> bi", jacobian, tangent_diagonal_matrix_output, jacobian
                            )
                            jTmjp_fast = layer.jTmjp(
                                x, None, tangent_diagonal_matrix_output, wrt="weight", from_diag=True, to_diag=True
                            )

                    # assert jTmjp_fast.shape == jTmjp_slow.shape
                    # assert torch.isclose(jTmjp_fast, jTmjp_slow, atol=1e-4).all()
                    assert is_close_block_diagonal(jTmjp_fast, jTmjp_slow, atol=1e-4).all()
