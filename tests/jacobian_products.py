"""
Test that the methods 
.jvp() .vjp() .jmp() .mjp() .jmjTp() .jTmjp()
return the correct tensor

NOTE: ensure first that test/jacobians.py runs correctly, 
since this tests assume that each layer has the correct implementation of .jacobian()
"""

import torch

import nnj

# define input sizes
batch_size = 1
shape_1D = (3,)
shape_2D = (3, 4)
shape_3D = (5, 6, 6)

# define input data
xs_nD = []
for shape in [shape_1D, shape_2D, shape_3D]:
    xs_nD.append(
        [
            torch.randn(batch_size, *shape),
            torch.ones(batch_size, *shape),
            torch.randn(batch_size, *shape) + torch.ones(batch_size, *shape),
        ]
    )
xs_1D, xs_2D, xs_3D = xs_nD

# define the layers to test
layers_on_allx = [
    nnj.Tanh(),
    nnj.ReLU(),
    nnj.Sigmoid(),
    nnj.Sinusoidal(),
    nnj.TruncExp(),
    nnj.Softplus(),
    nnj.Flatten(),
    nnj.L2Norm(),
]
layers_on_x1D = [
    nnj.Linear(*shape_1D, 5, bias=False),
    nnj.Linear(*shape_1D, 5),
    nnj.Sequential(nnj.Linear(*shape_1D, 5), nnj.Tanh(), nnj.Linear(5, 13), add_hooks=True),
    nnj.Sequential(nnj.Linear(*shape_1D, 5), nnj.ReLU(), nnj.Linear(5, 13), add_hooks=True),
    nnj.Sequential(
        nnj.Linear(*shape_1D, 6),
        nnj.Tanh(),
        nnj.Reshape(1, 2, 3),
        nnj.Flatten(),
        nnj.Tanh(),
        add_hooks=True,
    ),
    nnj.Sequential(
        nnj.Linear(*shape_1D, 5),
        nnj.Tanh(),
        nnj.Linear(5, 2),
        nnj.Tanh(),
        nnj.TruncExp(),
        nnj.Sinusoidal(),
        nnj.Linear(2, 12),
        nnj.Reshape(3, 4),
        nnj.Tanh(),
        nnj.Reshape(12),
        nnj.Tanh(),
        add_hooks=True,
    ),
    nnj.Sequential(
        nnj.Linear(*shape_1D, 5),
        nnj.Tanh(),
        nnj.Sequential(
            nnj.Linear(5, 5),
            nnj.Tanh(),
            nnj.Linear(5, 2),
            nnj.TruncExp(),
            add_hooks=True,
        ),
        nnj.ReLU(),
        nnj.Linear(2, 13),
        add_hooks=True,
    ),
    nnj.SkipConnection(nnj.Linear(*shape_1D, 5, bias=False)),
    nnj.SkipConnection(
        nnj.Linear(*shape_1D, 5),
        nnj.Tanh(),
        nnj.Linear(5, 2),
    ),
    nnj.SkipConnection(
        nnj.Linear(*shape_1D, 5),
        nnj.Tanh(),
        nnj.SkipConnection(
            nnj.Linear(5, 5),
            nnj.Tanh(),
            nnj.Linear(5, 2),
            nnj.TruncExp(),
        ),
        nnj.ReLU(),
        nnj.Linear(5 + 2, 13),
    ),
]
layers_on_x2D = [
    nnj.Reshape(6, 2),
]
layers_on_x3D = [
    nnj.Reshape(10, 9, 2),
    nnj.Sequential(
        nnj.Flatten(),
        nnj.Linear(180, 6),
        nnj.Tanh(),
        nnj.Reshape(1, 2, 3),
    ),
    nnj.SkipConnection(
        nnj.Flatten(),
        nnj.Linear(180, 6),
        nnj.Tanh(),
        nnj.SkipConnection(
            nnj.Linear(6, 2),
            nnj.Tanh(),
            nnj.Linear(2, 3),
        ),
        nnj.Tanh(),
        nnj.Linear(6 + 3, 36),
        nnj.Reshape(1, 6, 6),
    ),
]
layers_on_x3D_forward_only = []
layers_on_x3D_backward_only = [
    nnj.Upsample(scale_factor=2),
    nnj.Upsample(scale_factor=3),
    nnj.MaxPool2d(kernel_size=2, stride=2),
    nnj.MaxPool2d(2),
    nnj.MaxPool2d(3),
    nnj.Conv2d(shape_3D[0], 10, 3, stride=1, padding=1, bias=False),
    nnj.Conv2d(shape_3D[0], 10, 3, stride=1, padding=1, bias=True),
    nnj.Conv2d(shape_3D[0], 10, 3, stride=1, padding=1, bias=False, use_vmap_for_backprop=False),
    nnj.Conv2d(shape_3D[0], 10, 3, stride=1, padding=1, bias=True, use_vmap_for_backprop=False),
    nnj.Sequential(
        nnj.Tanh(),
        nnj.MaxPool2d(2),
        nnj.Upsample(scale_factor=3),
        nnj.Tanh(),
        add_hooks=True,
    ),
    nnj.Sequential(
        nnj.Tanh(),
        nnj.MaxPool2d(2),
        nnj.Flatten(),
        nnj.Linear(45, 12),
        nnj.Tanh(),
        nnj.Reshape(3, 2, 2),
        nnj.Conv2d(3, 10, 2, stride=1, padding=1, bias=True),
        nnj.Upsample(scale_factor=3),
        nnj.Tanh(),
        add_hooks=True,
    ),
    nnj.Sequential(
        nnj.Tanh(),
        nnj.Upsample(scale_factor=2),
        nnj.Flatten(),
        nnj.Linear(720, 4),
        nnj.Reshape(1, 2, 2),
        nnj.MaxPool2d(2),
        nnj.Tanh(),
        add_hooks=True,
    ),
    nnj.SkipConnection(nnj.Conv2d(shape_3D[0], 10, 3, stride=1, padding=1, bias=True)),
    nnj.SkipConnection(
        nnj.Conv2d(shape_3D[0], 10, 3, stride=1, padding=1, bias=True),
        nnj.Tanh(),
        nnj.MaxPool2d(2),
        nnj.SkipConnection(
            nnj.Conv2d(10, 10, 3, stride=1, padding=1, bias=True),
            nnj.Tanh(),
            nnj.Conv2d(10, 10, 3, stride=1, padding=1, bias=True),
        ),
        nnj.Upsample(scale_factor=2),
        nnj.Conv2d(10 + 10, 10, 3, stride=1, padding=1, bias=True),
        nnj.Tanh(),
    ),
]


###################
# vector products #
###################


### forward ###
def test_jvp():
    def test_jvp_wrt_input_on(layer, x):
        tangent_vector_input = torch.randn(x.shape[0], x[0].numel())

        jacobian = layer.jacobian(x, None, wrt="input")
        jvp_slow = torch.einsum("bij, bj -> bi", jacobian, tangent_vector_input)
        jvp_fast = layer.jvp(x, None, tangent_vector_input, wrt="input")

        assert jvp_fast.shape == jvp_slow.shape
        assert torch.isclose(jvp_fast, jvp_slow, atol=1e-4).all()

    def test_jvp_wrt_weight_on(layer, x):
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

    # run tests on all combinations
    for x in xs_1D:
        for layer in layers_on_allx + layers_on_x1D:
            test_jvp_wrt_input_on(layer, x)
            test_jvp_wrt_weight_on(layer, x)
    for x in xs_2D:
        for layer in layers_on_allx + layers_on_x2D:
            test_jvp_wrt_input_on(layer, x)
            test_jvp_wrt_weight_on(layer, x)
    for x in xs_3D:
        for layer in layers_on_allx + layers_on_x3D + layers_on_x3D_forward_only:
            test_jvp_wrt_input_on(layer, x)
            test_jvp_wrt_weight_on(layer, x)


### backward ###
def test_vjp():
    def test_vjp_wrt_input_on(layer, x):
        output = layer.forward(x)
        tangent_vector_output = torch.randn(output.shape[0], output[0].numel())

        jacobian = layer.jacobian(x, None, wrt="input")
        vjp_slow = torch.einsum("bi, bij -> bj", tangent_vector_output, jacobian)
        vjp_fast = layer.vjp(x, None, tangent_vector_output, wrt="input")

        assert vjp_fast.shape == vjp_slow.shape
        assert torch.isclose(vjp_fast, vjp_slow, atol=1e-4).all()

    def test_vjp_wrt_weight_on(layer, x):
        output = layer.forward(x)
        tangent_vector_output = torch.randn(output.shape[0], output[0].numel())

        vjp_fast = layer.vjp(x, None, tangent_vector_output, wrt="weight")
        if layer._n_params == 0:
            assert vjp_fast is None
        else:
            jacobian = layer.jacobian(x, None, wrt="weight")
            vjp_slow = torch.einsum("bi, bij -> bj", tangent_vector_output, jacobian)

            assert vjp_fast.shape == vjp_slow.shape
            assert torch.isclose(vjp_fast, vjp_slow, atol=1e-4).all()

    # run tests on all combinations
    for x in xs_1D:
        for layer in layers_on_allx + layers_on_x1D:
            test_vjp_wrt_input_on(layer, x)
            test_vjp_wrt_weight_on(layer, x)
    for x in xs_2D:
        for layer in layers_on_allx + layers_on_x2D:
            test_vjp_wrt_input_on(layer, x)
            test_vjp_wrt_weight_on(layer, x)
    for x in xs_3D:
        for layer in layers_on_allx + layers_on_x3D + layers_on_x3D_backward_only:
            test_vjp_wrt_input_on(layer, x)
            test_vjp_wrt_weight_on(layer, x)


###################
# matrix products #
###################


### forward ###
def test_jmp():
    def test_jmp_wrt_input_on(layer, x):
        n_columns = 120
        batch_size = x.shape[0]
        tangent_matrix_input = torch.randn(batch_size, x.shape[1:].numel(), n_columns)

        jacobian = layer.jacobian(x, None, wrt="input")
        jmp_slow = torch.einsum("bij, bjk -> bik", jacobian, tangent_matrix_input)
        jmp_fast = layer.jmp(x, None, tangent_matrix_input, wrt="input")

        assert jmp_fast.shape == jmp_slow.shape
        assert torch.isclose(jmp_fast, jmp_slow, atol=1e-4).all()

    def test_jmp_wrt_weight_on(layer, x):
        n_columns = 120
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

    # run tests on all combinations
    for x in xs_1D:
        for layer in layers_on_allx + layers_on_x1D:
            test_jmp_wrt_input_on(layer, x)
            test_jmp_wrt_weight_on(layer, x)
    for x in xs_2D:
        for layer in layers_on_allx + layers_on_x2D:
            test_jmp_wrt_input_on(layer, x)
            test_jmp_wrt_weight_on(layer, x)
    for x in xs_3D:
        for layer in layers_on_allx + layers_on_x3D + layers_on_x3D_forward_only:
            test_jmp_wrt_input_on(layer, x)
            test_jmp_wrt_weight_on(layer, x)


### backward ###
def test_mjp():
    def test_mjp_wrt_input_on(layer, x):
        n_rows = 120
        batch_size = x.shape[0]
        output_shape = layer.forward(x).shape
        tangent_matrix_output = torch.randn(batch_size, n_rows, output_shape[1:].numel())

        jacobian = layer.jacobian(x, None, wrt="input")
        mjp_slow = torch.einsum("bij, bjk -> bik", tangent_matrix_output, jacobian)
        mjp_fast = layer.mjp(x, None, tangent_matrix_output, wrt="input")

        assert mjp_fast.shape == mjp_slow.shape
        assert torch.isclose(mjp_fast, mjp_slow, atol=1e-4).all()

    def test_mjp_wrt_weight_on(layer, x):
        n_rows = 120
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

    # run tests on all combinations
    for x in xs_1D:
        for layer in layers_on_allx + layers_on_x1D:
            test_mjp_wrt_input_on(layer, x)
            test_mjp_wrt_weight_on(layer, x)
    for x in xs_2D:
        for layer in layers_on_allx + layers_on_x2D:
            test_mjp_wrt_input_on(layer, x)
            test_mjp_wrt_weight_on(layer, x)
    for x in xs_3D:
        for layer in layers_on_allx + layers_on_x3D + layers_on_x3D_backward_only:
            test_mjp_wrt_input_on(layer, x)
            test_mjp_wrt_weight_on(layer, x)


#####################
# sandwich products #
#####################


### forward ###
def test_jmjTp():
    def test_jmjTp_wrt_input_on(layer, x, from_diag=False, to_diag=False):
        batch_size = x.shape[0]
        jacobian = layer.jacobian(x, None, wrt="input")

        if from_diag is False:
            tangent_matrix_input = torch.randn(batch_size, x.shape[1:].numel(), x.shape[1:].numel())

            if to_diag is False:
                # full -> full
                jmjTp_slow = torch.einsum("bij, bjk, bqk -> biq", jacobian, tangent_matrix_input, jacobian)
                jmjTp_fast = layer.jmjTp(x, None, tangent_matrix_input, wrt="input", from_diag=False, to_diag=False)
            elif to_diag is True:
                # full -> diag
                jmjTp_slow = torch.einsum("bij, bjk, bik -> bi", jacobian, tangent_matrix_input, jacobian)
                jmjTp_fast = layer.jmjTp(x, None, tangent_matrix_input, wrt="input", from_diag=False, to_diag=True)

        elif from_diag is True:
            tangent_diagonal_matrix_input = torch.randn(batch_size, x.shape[1:].numel())

            if to_diag is False:
                # diag -> full
                jmjTp_slow = torch.einsum("bij, bj, bqj -> biq", jacobian, tangent_diagonal_matrix_input, jacobian)
                jmjTp_fast = layer.jmjTp(
                    x, None, tangent_diagonal_matrix_input, wrt="input", from_diag=True, to_diag=False
                )
            elif to_diag is True:
                # diag -> diag
                jmjTp_slow = torch.einsum("bij, bj, bij -> bi", jacobian, tangent_diagonal_matrix_input, jacobian)
                jmjTp_fast = layer.jmjTp(
                    x, None, tangent_diagonal_matrix_input, wrt="input", from_diag=True, to_diag=True
                )

        assert jmjTp_fast.shape == jmjTp_slow.shape
        assert torch.isclose(jmjTp_fast, jmjTp_slow, atol=1e-4).all()

    def test_jmjTp_wrt_weight_on(layer, x, from_diag=False, to_diag=False):
        # utility functions for sequential layers
        def generate_random_matrix_params(batch_size, module):
            if isinstance(module, nnj.Sequential) or isinstance(module, nnj.SkipConnection):
                if isinstance(module, nnj.SkipConnection):
                    module = module._F
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
                    embedded_matrix[
                        :, i_row : i_row + submatrix.shape[1], i_col : i_col + submatrix.shape[2]
                    ] = submatrix
                    i_row += submatrix.shape[1]
                    i_col += submatrix.shape[2]
                return embedded_matrix
            return matrix

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
                jmjTp_fast = layer.jmjTp(x, None, tangent_matrix_params, wrt="weight", from_diag=False, to_diag=False)
            elif to_diag is True:
                # full -> diag
                jmjTp_slow = torch.einsum(
                    "bij, bjk, bik -> bi", jacobian, embed_block_diagonal(tangent_matrix_params), jacobian
                )
                jmjTp_fast = layer.jmjTp(x, None, tangent_matrix_params, wrt="weight", from_diag=False, to_diag=True)

        elif from_diag is True:
            tangent_diagonal_matrix_params = torch.randn(batch_size, layer._n_params)

            if to_diag is False:
                # diag -> full
                jmjTp_slow = torch.einsum("bij, bj, bqj -> biq", jacobian, tangent_diagonal_matrix_params, jacobian)
                jmjTp_fast = layer.jmjTp(
                    x, None, tangent_diagonal_matrix_params, wrt="weight", from_diag=True, to_diag=False
                )
            elif to_diag is True:
                # diag -> diag
                jmjTp_slow = torch.einsum("bij, bj, bij -> bi", jacobian, tangent_diagonal_matrix_params, jacobian)
                jmjTp_fast = layer.jmjTp(
                    x, None, tangent_diagonal_matrix_params, wrt="weight", from_diag=True, to_diag=True
                )

        assert jmjTp_fast.shape == jmjTp_slow.shape
        assert torch.isclose(jmjTp_fast, jmjTp_slow, atol=1e-4).all()

    # run tests on all combinations
    for from_diag, to_diag in [(False, False), (False, True), (True, False), (True, True)]:
        for x in xs_1D:
            for layer in layers_on_allx + layers_on_x1D:
                test_jmjTp_wrt_input_on(layer, x, from_diag, to_diag)
                test_jmjTp_wrt_weight_on(layer, x, from_diag, to_diag)
        for x in xs_2D:
            for layer in layers_on_allx + layers_on_x2D:
                test_jmjTp_wrt_input_on(layer, x, from_diag, to_diag)
                test_jmjTp_wrt_weight_on(layer, x, from_diag, to_diag)
        for x in xs_3D:
            for layer in layers_on_allx + layers_on_x3D + layers_on_x3D_forward_only:
                test_jmjTp_wrt_input_on(layer, x, from_diag, to_diag)
                test_jmjTp_wrt_weight_on(layer, x, from_diag, to_diag)


### backward ###
def test_jTmjp():
    def test_jTmjp_wrt_input_on(layer, x, from_diag=False, to_diag=False):
        output_shape = layer.forward(x).shape
        batch_size = x.shape[0]
        jacobian = layer.jacobian(x, None, wrt="input")

        if from_diag is False:
            tangent_matrix_output = torch.randn(batch_size, output_shape[1:].numel(), output_shape[1:].numel())

            if to_diag is False:
                # full -> full
                jTmjp_slow = torch.einsum("bji, bjk, bkq -> biq", jacobian, tangent_matrix_output, jacobian)
                jTmjp_fast = layer.jTmjp(x, None, tangent_matrix_output, wrt="input", from_diag=False, to_diag=False)
            elif to_diag is True:
                # full -> diag
                jTmjp_slow = torch.einsum("bji, bjk, bki -> bi", jacobian, tangent_matrix_output, jacobian)
                jTmjp_fast = layer.jTmjp(x, None, tangent_matrix_output, wrt="input", from_diag=False, to_diag=True)

        elif from_diag is True:
            tangent_diagonal_matrix_output = torch.randn(batch_size, output_shape[1:].numel())
            tangent_diagonal_matrix_output = torch.ones(batch_size, output_shape[1:].numel())

            if to_diag is False:
                # diag -> full
                jTmjp_slow = torch.einsum("bji, bj, bjk -> bik", jacobian, tangent_diagonal_matrix_output, jacobian)
                jTmjp_fast = layer.jTmjp(
                    x, None, tangent_diagonal_matrix_output, wrt="input", from_diag=True, to_diag=False
                )
            elif to_diag is True:
                # diag -> diag
                jTmjp_slow = torch.einsum("bji, bj, bji -> bi", jacobian, tangent_diagonal_matrix_output, jacobian)
                jTmjp_fast = layer.jTmjp(
                    x, None, tangent_diagonal_matrix_output, wrt="input", from_diag=True, to_diag=True
                )

        assert jTmjp_fast.shape == jTmjp_slow.shape
        assert torch.isclose(jTmjp_fast, jTmjp_slow, atol=1e-4).all()

    def test_jTmjp_wrt_weight_on(layer, x, from_diag=False, to_diag=False):
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

        if layer._n_params == 0:
            return  # TODO: check that .jmjTp returns None in all cases

        output_shape = layer.forward(x).shape
        batch_size = x.shape[0]
        jacobian = layer.jacobian(x, None, wrt="weight")

        if from_diag is False:
            tangent_matrix_output = torch.randn(batch_size, output_shape[1:].numel(), output_shape[1:].numel())

            if to_diag is False:
                # full -> full
                jTmjp_slow = torch.einsum("bji, bjk, bkq -> biq", jacobian, tangent_matrix_output, jacobian)
                jTmjp_fast = layer.jTmjp(x, None, tangent_matrix_output, wrt="weight", from_diag=False, to_diag=False)
            elif to_diag is True:
                # full -> diag
                jTmjp_slow = torch.einsum("bji, bjk, bki -> bi", jacobian, tangent_matrix_output, jacobian)
                jTmjp_fast = layer.jTmjp(x, None, tangent_matrix_output, wrt="weight", from_diag=False, to_diag=True)

        elif from_diag is True:
            tangent_diagonal_matrix_output = torch.randn(batch_size, output_shape[1:].numel())

            if to_diag is False:
                # diag -> full
                jTmjp_slow = torch.einsum("bji, bj, bjk -> bik", jacobian, tangent_diagonal_matrix_output, jacobian)
                jTmjp_fast = layer.jTmjp(
                    x, None, tangent_diagonal_matrix_output, wrt="weight", from_diag=True, to_diag=False
                )
            elif to_diag is True:
                # diag -> diag
                jTmjp_slow = torch.einsum("bji, bj, bji -> bi", jacobian, tangent_diagonal_matrix_output, jacobian)
                jTmjp_fast = layer.jTmjp(
                    x, None, tangent_diagonal_matrix_output, wrt="weight", from_diag=True, to_diag=True
                )

        assert is_close_block_diagonal(jTmjp_fast, jTmjp_slow, atol=1e-4).all()

    # run tests on all combinations
    for from_diag, to_diag in [(False, False), (False, True), (True, False), (True, True)]:
        for x in xs_1D:
            for layer in layers_on_allx + layers_on_x1D:
                test_jTmjp_wrt_input_on(layer, x, from_diag, to_diag)
                test_jTmjp_wrt_weight_on(layer, x, from_diag, to_diag)
        for x in xs_2D:
            for layer in layers_on_allx + layers_on_x2D:
                test_jTmjp_wrt_input_on(layer, x, from_diag, to_diag)
                test_jTmjp_wrt_weight_on(layer, x, from_diag, to_diag)
        for x in xs_3D:
            for layer in layers_on_allx + layers_on_x3D + layers_on_x3D_backward_only:
                test_jTmjp_wrt_input_on(layer, x, from_diag, to_diag)
                test_jTmjp_wrt_weight_on(layer, x, from_diag, to_diag)
