NNJ Documentation
===================================

nnj extends torch.nn with fast jacobian vector/matrix product (jvp/mvp).
It is a order of magnitude faster and more memory efficient than alternatives.
All jacobian computations are implemented in native PyTorch.


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


Usage
===============================


.. code-block:: python

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
