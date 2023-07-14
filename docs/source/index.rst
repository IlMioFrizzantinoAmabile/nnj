NNJ Documentation
===================================

NNJ is torch.nn plus Jacobian operations. It efficiently implements both forward and backward passes of both directions and metrics in tangent spaces.
It is a fast and more memory efficient extension of PyTorch.


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


Check out our educational content, e.g. start reading our :ref:`Introduction to nnj<introduction>`  to learn more about the nnj.
The document seek to provide a simple introduction to the nnj and how to use it in PyTorch.

.. toctree::
   :glob:
   :hidden:
   :caption: Learn about nnj 

   nnj/*



Wandering why not just use Jax?
==============================================================


Check out our comparison, e.g. start reading our :ref:`Ok, but why<why>`  to learn more about it.

.. toctree::
   :glob:
   :hidden:
   :caption: Ok, but why?

   why/*
   


Usage
===============================


.. code-block:: python

   import torch.nn as nn
   import nnj

   # Define you sequential model
   network = torch.nn.Sequential(
      nn.Linear(),
      nn.Tanh(),
      nn.Linear(),
   )

   # convert to nnj
   network_nnj = nnj.utils.convert_to_nnj(
      network_nn,
   )

   # calling jTmjp on a sequential will iterate through the network and compute jacobian-tranposed matrix jacobian product (e.g. Generalized-Gauss Newton approximation of the hessian) of the nnj network. 
   with torch.no_grad():
      val = nnj_module(x)

      # backpropagate through the network
      Jt_J = nnj_module.jTmjp(
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



Links:
-------------

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Python API

   apis/*
