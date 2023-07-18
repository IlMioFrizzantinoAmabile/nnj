NNJ Documentation
===================================

NNJ = torch.nn + Jacobian. 

Both forward and backward passes for both vector and metrics in tangent spaces are implemented for every module.
This library is a fast and memory efficient extension of PyTorch.


Github: https://github.com/IlMioFrizzantinoAmabile/nnj

Authors: Marco Miani and Frederik Warburg 


Installation
===============================

**Dependence**: Please install Pytorch first.

The easiest way is to install from PyPI:

.. code-block:: console

   $ pip install nnj

Or install from source:

.. code-block:: console

   $ git clone https://github.com/IlMioFrizzantinoAmabile/nnj
   $ cd nnj
   $ pip install -e .


Want to learn more about nnj?
==============================================================


Check out our :ref:`Introduction to nnj<introduction>`  to learn more about the theory behind nnj.
The document seek to provide a simple introduction to nnj and to the functions it provides.

.. toctree::
   :glob:
   :hidden:
   :caption: Learn about nnj 

   nnj/*



Usage example
===============================

Declare your neural network as you would normally do with PyTorch, just with an extra j.

.. code-block:: python

   import nnj

   # Define you sequential model
   model = nnj.Sequential(
      nnj.Linear(),
      nnj.Tanh(),
      nnj.Linear(),
   )

   # Standard forward pass
   val = model(x)

Compute gradient (with respect to weight) of the l2 loss as backward pass of the residual vector, and perform a gradient step.

.. code-block:: python

   # The residual is the derivative of the loss with respect to the nn output
   residual = val - target
   # Backpropagate the residual vector
   gradient = model.vjp(
      x,             # input
      val,           # output
      residual,      # residual vector
      wrt="weight"
   )
   # Average over batch size
   gradient = torch.mean(gradient, dim=0)

   # Do a gradient step
   param = model.get_weight()
   param -= lr * gradient
   model.set_weight(param)

Compute the Generalized-Gauss Newton (which is an approximation of the hessian) as a backward pass of the Euclidean metric.

.. code-block:: python

   jacobianTranspose_jacobian = model._jTmjp(
      x,                  # input
      val,                # output
      None,               # None means identity (i.e. Euclidean metric)
      wrt="weights",      # computes the jacobian wrt weights or inputs
      to_diag=True,       # computes the diagonal elements only
      diag_backprop=True, # approximates the diagonal elements, which speeds up the computations
   )




Why not just use Jax?
==============================================================


Check out our comparison, e.g. start reading our :ref:`Ok, but why<why>`  to learn more about it. (WORK IN PROGRESS)

.. toctree::
   :glob:
   :hidden:
   :caption: Ok, but why?

   why/*
   


Links:
-------------

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Python API

   apis/*
