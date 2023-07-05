# nnj

**tl;dr. The nnj repo implements efficient jacobian and approximate hessian computation in native pytorch.**

## Installation

```bash
pip install -e .
```

## Comparison with jax, backpack, asdfghj, and torch

TODO: compare speed and memory usage of implementation.

## Example usage

```python
import torch.nn as nn
import nnj

# Define you sequential model
network_nn = torch.nn.Sequential(
    nn.Linear(),
    nn.Tanh(),
    nn.Linear(),
)

# convert to nnj
network_nnj = nnj.utils.convert_to_nnj(
    network_nn,
)

# calling _jTmjp on a sequential will iterate through the network and compute jacobian-tranposed matrix jacobian product (e.g. Generalized-Gauss Newton approximation of the hessian) of the nnj network. 
with torch.no_grad():
    val = nnj_module(x)

    # backpropagate through the network
    Jt_J = nnj_module._jTmjp(
        x, 
        val,
        None,
        wrt="weights",      # computes the jacobian wrt weights or inputs
        to_diag=True,       # computes the diagonal elements only
        diag_backprop=True, # approximates the diagonal elements, which speeds up the computations
    )
    # average along batch size
    Jt_J = torch.mean(Jt_J, dim=0)
    return Jt_J
```

## Explaination

For each layer in torch.nn (and some additional layers, we found useful), we implement the following additional operations:


## vector-jacobian products
```python
## Bector-jacobian-product wrt. weight (vector multipled from the right)
layer._vjp(x=x, value=None, vector=v, wrt="weight") # $$ v J_{\theta} $$

## Vector-jacobian-product wrt. input (vector multipled from the right)
layer._vjp(x=x, value=None, vector=v, wrt="input") # $$v J_{x}$$

## Jacobian-vector-product wrt. weight (vector multipled from the left)
layer._jvp(x=x, value=None, vector=v, wrt="weight") # $$ J_{\theta} v$$

## Jacobian-vector-product wrt. input (vector multipled from the left)
layer._jvp(x=x, value=None, vector=v, wrt="input") # $$J_{x} v$$
```

## matrix-jacobian products
These expend vector-jacobian products.
```python
## Jacobian-matrix-product wrt. weight (matrix multipled from the right)
layer._jmp(x=x, matrix=m, wrt="weight") # $$J_{\theta} m$$

## Jacobian-matrix-product wrt. input (matrix multiplied from the right)
layer._jmp(x=x, matrix=m, wrt="input") # $$J_{x} m$$

## Matrix-jacobian-product wrt. weight (matrix multipled from the left)
layer._mjp(x=x, matrix=m, wrt="weight") # $$m J_{\theta}$$

## Matrix-jacobian-product wrt. input (matrix multiplied from the left)
layer._mjp(x=x, matrix=m, wrt="input") # $$m J_{x}$$
```

## sandwitch-jacobian products
These are for from_diag [True/False], to_diag [True/False], approximate diag [True/False] (only works with from_diag=True, to_diag=True)
```python
## Jacobian-Matrix-jacobian-transpose-product wrt. input 
layer._jmjTp(x=x, matrix=m, wrt="input") # $$J_{x} m J_{x}^T$$

## Jacobian-Matrix-jacobian-transpose-product wrt. weight 
layer._jmjTp(x=x, matrix=m, wrt="weight") # $$J_{\theta} m J_{\theta}^T$$

## Jacobian-transpose-Matrix-jacobian-product wrt. input 
layer._jTmjp(x=x, matrix=m, wrt="input") # $$J_{x}^T m J_{x}$$

## Jacobian-transpose-Matrix-jacobian-product wrt. weight 
layer._jTmjp(x=x, matrix=m, wrt="weight") # $$J_{\theta}^T m J_{\theta}$$
```

## TODO: describe (maybe we should ignore for now...?)
I assume few people will find it useful. Maybe we can put on a branch for now...
```python
layer._jTmjp_batch2(x1=x1, x2=x2, matrixes=m) # TODO: write
```

## Jacobian of a layer
If you are interested in the Jacobian of a specific layer, then set the value to the identity, e.g. 

```python
## Jacobian wrt. input
b, c, l = x.shape
v = torch.identity((b, c, l))
layer._jvp(x=x, vector=v, wrt="input") # $$J_{x}$$

## Jacobian wrt. weight
layer._jacobian(x=x, vector=v, wrt="weight") # $$J_{\theta}$$
```

## Approximate Jacobians

It is oftened useful to consider approximate Jacobians such as diagonal or approximate diagonal jacobians. This repo supports four type of jacobian approximations. 




For extended examples of using the repo for the Laplace approximation, please checkout: https://github.com/FrederikWarburg/pytorch-laplace

## Add a custom layer

It is straightforward to add a custom layer. All you need to do is to implement the jacobian wrt to weights and inputs of your layer. This will allow you to compute all of the above functions with your custom layer. This will work, but this is not guarantee to lead to the optimal performance as the Jacobian-product might be sparse. For popular layers, such as nnj.Linear and nnj.Conv2d, we show show how Jacobian products can be overwritten (e.g. layer._jmjTp) for faster performance. 

```python
class MyCustomLayer(AbstractJacobian, nn.Linear):
    def __init__(self,):
        # initialize the layer

    def forward(self, x):
        # define your forward function

    def _jacobian(self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input") -> Tensor:

        # val is the output of the model, if this is not given, make a forward pass
        if val is None:
            val = self.forward(x)

        """Returns the Jacobian matrix"""
        if wrt == "input":            
            # returns jacobian with respect to inputs
        elif wrt == "weight":
            # return the jacobian with respect to weights
            # this is none if the layer is non-parametric (aka doesn't have learnable weights)
```

