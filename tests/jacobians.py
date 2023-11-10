"""
Test that the method .jacobian() returns the correct matrix

-wrt input is checked against the row-by-row construction made with 
        torch.autograd.functional.vjp
-wrt weight is checked against the row-by-row construction made with 
        .backward()
"""

import torch

import nnj

# define input sizes
batch_size = 7
shape_1D = (3,)
shape_2D = (3, 4)
shape_3D = (5, 6, 4)

# define input data
xs_nD = []
for shape in [shape_1D, shape_2D, shape_3D]:
    xs_nD.append(
        [
            torch.randn(batch_size, *shape),
            torch.ones(batch_size, *shape),
            torch.randn(batch_size, *shape) + torch.ones(batch_size, *shape),
            10 * torch.rand(batch_size, *shape),
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
    nnj.Sequential(nnj.Linear(*shape_1D, 5, bias=False)),
    nnj.Sequential(nnj.Linear(*shape_1D, 5), nnj.Tanh(), nnj.Linear(5, 13), add_hooks=True),
    nnj.Sequential(nnj.Linear(*shape_1D, 5), nnj.ReLU(), nnj.Linear(5, 13), add_hooks=True),
    nnj.Sequential(
        nnj.Linear(*shape_1D, 6),
        nnj.Tanh(),
        nnj.Reshape(1, 2, 3),
        nnj.Upsample(scale_factor=2),
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
    nnj.Upsample(scale_factor=2),
    nnj.Upsample(scale_factor=3),
    nnj.MaxPool2d(kernel_size=2, stride=2),
    nnj.Reshape(6, 10, 2),
    nnj.Sequential(
        nnj.Flatten(),
        nnj.Linear(120, 6),
        nnj.Tanh(),
        nnj.Reshape(1, 2, 3),
        add_hooks=True,
    ),
    nnj.Sequential(
        nnj.MaxPool2d(2),
        nnj.Flatten(),
        nnj.Linear(30, 12),
        nnj.Reshape(3, 2, 2),
        nnj.Conv2d(3, 7, 2, stride=1, padding=1, bias=True),
        nnj.Upsample(scale_factor=3),
        add_hooks=True,
    ),
    nnj.Conv2d(5, 10, 2, stride=1, padding=1, bias=False),
    nnj.Conv2d(5, 10, 2, stride=1, padding=1, bias=True),
    nnj.SkipConnection(
        nnj.Flatten(),
        nnj.Linear(120, 6),
        nnj.Tanh(),
        nnj.SkipConnection(
            nnj.Linear(6, 2),
            nnj.Tanh(),
            nnj.Linear(2, 13),
        ),
        nnj.Tanh(),
        nnj.Linear(6 + 13, 24),
        nnj.Reshape(1, 6, 4),
    ),
]


def test_jacobian_wrt_input():
    # define test on specific layer and input
    def test_jacobian_wrt_input_on(layer, x):
        val = layer(x)
        batch_size = x.shape[0]
        in_size = x[0].numel()
        out_size = val[0].numel()
        out_shape = val.shape[1:]

        jacobian_nnj = layer.jacobian(x, None, wrt="input")

        jacobian_nn = torch.zeros(batch_size, out_size, in_size)
        for o in range(out_size):
            basis_vector = torch.zeros_like(val).reshape(batch_size, out_size)
            for b in range(batch_size):
                basis_vector[b, o] = 1.0
            basis_vector = basis_vector.reshape(batch_size, *out_shape)
            row_o = torch.autograd.functional.vjp(layer, x, v=basis_vector)[1]
            row_o = row_o.reshape(batch_size, in_size)
            jacobian_nn[:, o, :] = row_o

        assert jacobian_nnj.shape == jacobian_nn.shape
        assert torch.isclose(jacobian_nnj, jacobian_nn, atol=1e-4).all()

    # run tests on all combinations
    for x in xs_1D:
        for layer in layers_on_allx + layers_on_x1D:
            test_jacobian_wrt_input_on(layer, x)
    for x in xs_2D:
        for layer in layers_on_allx + layers_on_x2D:
            test_jacobian_wrt_input_on(layer, x)
    for x in xs_3D:
        for layer in layers_on_allx + layers_on_x3D:
            test_jacobian_wrt_input_on(layer, x)


def test_jacobian_wrt_weight():
    # define test on specific layer and input
    def test_jacobian_wrt_weight_on(layer, x):
        val = layer(x)
        batch_size = x.shape[0]
        out_size = val[0].numel()
        param_size = layer._n_params

        jacobian_nnj = layer.jacobian(x, None, wrt="weight")
        if layer._n_params == 0:
            assert jacobian_nnj is None
            return

        jacobian_nn = torch.zeros(batch_size, out_size, param_size)
        for b in range(batch_size):
            val = layer(x[b : b + 1]).reshape(out_size)
            for o in range(out_size):
                loss = val[o]
                loss.backward(retain_graph=True)
                row_o = []
                for p in layer.parameters():
                    row_o.append(torch.clone(p.grad).view(-1))
                    p.grad.zero_()
                row_o = torch.cat(row_o)
                jacobian_nn[b, o, :] = row_o

        assert jacobian_nnj.shape == jacobian_nn.shape
        assert torch.isclose(jacobian_nnj, jacobian_nn, atol=1e-4).all()

    # run tests on all combinations
    for x in xs_1D:
        for layer in layers_on_allx + layers_on_x1D:
            test_jacobian_wrt_weight_on(layer, x)
    for x in xs_2D:
        for layer in layers_on_allx + layers_on_x2D:
            test_jacobian_wrt_weight_on(layer, x)
    for x in xs_3D:
        for layer in layers_on_allx + layers_on_x3D:
            test_jacobian_wrt_weight_on(layer, x)
