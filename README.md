# nnj

**tl;dr. The nnj repo implements efficient jacobian and approximate hessian computation in native pytorch.**

## Installation

```bash
git clone https://github.com/IlMioFrizzantinoAmabile/nnj
cd nnj
pip install -e .
```

## Explanation

A classic backpropagation recovers the gradient as a backward pass of a vector, the residual. A similar propagation can be done with a forward pass. 
These two methods are called **vjp** and **jvp**, respectively. Similarly to JAX, we implement such functions for layers in *torch.nn*.

These forward and backward passes propagate directions (i.e. vectors). We extend them to propagate also metrics (i.e. matrixes) with the methods **jTmjp** and **jmjTp**.
Moreover we implement an approximate but faster version of these methods that has the same order of complexity of a standard backpropagation.

Full explanation and documentation at https://ilmiofrizzantinoamabile.github.io/nnj/


## Example usage

```python
import torch
import nnj

# Define you sequential model
model = nnj.Sequential(
    nnj.Linear(),
    nnj.Tanh(),
    nnj.Linear(),
)

val = model(x)

# Compute gradient (of the l2 loss) as backward pass of the residual
residual = val - target
gradient = model.vjp(x, val, residual, wrt="weight")

# Compute the Generalized-Gauss Newton (an approximation of the hessian) as a backward pass of the Euclidean metric
jacobianTranspose_jacobian = model.jTmjp(
    x, 
    val,
    None,               # None means identity (i.e. Euclidean metric)
    wrt="weights",      # computes the jacobian wrt weights or inputs
    to_diag=True,       # computes the diagonal elements only
    diag_backprop=True, # approximates the diagonal elements, which speeds up the computations
)

# Average along batch size
gradient = torch.mean(gradient, dim=0)
jacobianTranspose_jacobian = torch.mean(jacobianTranspose_jacobian, dim=0)
```

For extended examples of using the repo in the context of the Laplace approximation, please checkout: https://github.com/FrederikWarburg/pytorch-laplace


## Comparison with jax, backpack, asdfghj, and torch

TODO: compare speed and memory usage of implementation.


## Add a custom layer

It is straightforward to add a custom layer: all you need to do is to implement the *jacobian* wrt to weights and inputs of your layer. 

Then all the forward and backward passes with your custom layer are accessible thanks to the *AbstractJacobian* class. 
Although this will work, it won't lead to the optimal performance (probably) as the Jacobian-product might be sparse. For popular layers, such as nnj.Linear and nnj.Conv2d, we show show how Jacobian products can be overwritten (e.g. layer.jmjTp) for faster performance. 

```python
class nnjCustomLayer(AbstractJacobian, nn.CustomLayer):

    def forward(self, x):
        # define your forward function or inherit it from nn.CustomLayer

    def jacobian(self, x: Tensor, val: Union[Tensor, None] = None, wrt: Literal["input", "weight"] = "input") -> Tensor:

        # val is the output of the model, if this is not given, make a forward pass
        if val is None:
            val = self.forward(x)

        """Returns the Jacobian matrix"""
        if wrt == "input":            
            # returns jacobian with respect to inputs
        elif wrt == "weight":
            # return the jacobian with respect to weights
            # this is none if the layer is non-parametric (i.e. doesn't have learnable weights)
```

![Bender](https://github.com/IlMioFrizzantinoAmabile/nnj/blob/main/docs/source/_static/images/Bender.png)
