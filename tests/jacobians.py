"""
Test that the method .jacobian() returns the correct matrix

-wrt input is checked against the row-by-row construction made with 
        torch.autograd.functional.vjp
-wrt weight is checked against the row-by-row construction made with 
        .backward()
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
    nnj.Sequential(nnj.Linear(3, 5), nnj.Tanh(), nnj.Linear(5, 13), add_hooks=True),
    nnj.Sequential(nnj.Linear(3, 5), nnj.ReLU(), nnj.Linear(5, 13), add_hooks=True),
    nnj.Sequential(
        nnj.Linear(3, 5),
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
        nnj.Linear(3, 5),
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
    nnj.Sigmoid(),
    nnj.Softplus(),
    nnj.TruncExp(),
]


def test_jacobian_wrt_input():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            val = layer(x)
            assert len(x.shape) == 2 and len(val.shape) == 2
            batch_size, out_size = val.shape
            _, in_size = x.shape

            jacobian_nnj = layer.jacobian(x, None, wrt="input")

            jacobian_nn = torch.zeros(batch_size, out_size, in_size)
            for o in range(out_size):
                basis_vector = torch.zeros_like(val)
                for b in range(batch_size):
                    basis_vector[b, o] = 1.0
                row_o = torch.autograd.functional.vjp(layer, x, v=basis_vector)[1]
                jacobian_nn[:, o, :] = row_o

            assert jacobian_nnj.shape == jacobian_nn.shape
            assert torch.isclose(jacobian_nnj, jacobian_nn, atol=1e-4).all()


def test_jacobian_wrt_weight():
    for layer in to_test_easy + to_test_advanced:
        for x in xs:
            val = layer(x)
            assert len(val.shape) == 2
            batch_size, out_size = val.shape
            _, in_size = x.shape
            param_size = layer._n_params

            jacobian_nnj = layer.jacobian(x, None, wrt="weight")
            if layer._n_params == 0:
                assert jacobian_nnj is None
                continue

            jacobian_nn = torch.zeros(batch_size, out_size, param_size)
            for b in range(batch_size):
                val = layer(x[b : b + 1, :])
                for o in range(out_size):
                    loss = val[0, o]
                    loss.backward(retain_graph=True)
                    row_o = []
                    for p in layer.parameters():
                        row_o.append(torch.clone(p.grad).view(-1))
                        p.grad.zero_()
                    row_o = torch.cat(row_o)
                    jacobian_nn[b, o, :] = row_o

            assert jacobian_nnj.shape == jacobian_nn.shape
            assert torch.isclose(jacobian_nnj, jacobian_nn, atol=1e-4).all()
