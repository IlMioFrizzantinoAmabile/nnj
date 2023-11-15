.. _introduction:

Overview
=============

A function :math:`f: \mathcal{X}\rightarrow\mathcal{Y}` induces a map between directions :math:`v\in\mathcal{X}` to directions :math:`w\in\mathcal{Y}` such that :math:`f(x + v) = f(x) + w`, when the step is *small*. 
This map is known as the derivative :math:`\nabla f` and, when evaluated in a specific point :math:`\nabla f: x\in\mathcal{X}`, it is a linear operator :math:`\nabla f(x): T_x\mathcal{X}\rightarrow T_{f(x)}\mathcal{Y}` between the tangent space in :math:`x` and the tangent space in :math:`f(x)`. 

The linear operator is just a matrix multiplication, the matrix is called **Jacobian** and it is usually identified with the same notation as the linear operator itself :math:`J=\nabla f(x)`.

The forward *direction* pass **JVP** (Jacobian Vector Product) is

.. math::
    \begin{aligned}
    J \cdot v \in  T_{f(x)}\mathcal{Y}
    \qquad\qquad \forall v\in  T_x\mathcal{X}
    \end{aligned}

The backward *direction* pass **VJP** (Vector Jacobian Product) is

.. math::
    \begin{aligned}
    J^\top \cdot w \in  T_x\mathcal{X}
    \qquad \forall w\in  T_{f(x)}\mathcal{Y}
    \end{aligned}


The forward *metric* pass **JMJtP** (Jacobian Matrix Jacobian transpose Product) is

.. math::
    \begin{aligned}
    J \cdot M \cdot J^\top \in  \mathfrak{M}(T_{f(x)}\mathcal{Y})
    \qquad\qquad \forall M\in \mathfrak{M}(T_x\mathcal{X})
    \end{aligned}

The backward *metric* pass **JtMJP** (Jacobian transpose Matrix Jacobian Product) is

.. math::
    \begin{aligned}
    J^\top \cdot M \cdot J \in  \mathfrak{M}(T_x\mathcal{X})
    \qquad\qquad \forall M\in \mathfrak{M}(T_{f(x)}\mathcal{Y})
    \end{aligned}

The first two jacobian products are more common and supported by several repos (like JAX), while the latter are more specific and not commonly supported. 
The scope of this repository is to provide efficent implementation of these jacobian products for all PyTorch *modules*.
In PyTorch, objects of the class *module* are the building blocks for functions, they can be either parametric layer (as linear or convolutions) or non-parametric layer (as tanh or relu). 
Moreover, composition of modules is a module itself (sequential) and even arbitrary function of other modules can be a module (as skip-connection or res-block or attention). 


Jacobian in a neural network
==============================
A neural network is actually a function with two inputs: data and parameters. Formally 

.. math::
    f: \mathcal{X}\times\Theta\rightarrow\mathcal{Y}
    
And thus, depending on which input we take the derivative with respct to, there are two different Jacobian to consider: 
 * :math:`J=\nabla_x f(x,\theta)` when computed with respect to the data
 * :math:`J=\nabla_\theta f(x,\theta)` when computed with respect to parameter

Following the common usage of neural networks, we use the keyword *input* for data and *weight* for parameter.



Efficient implementation
==========================
The building block for neural networks, as in PyTorch, is the **Module**: a black-box function with two inputs: data and parameters. Such black-box can either be defined *explicitly* or *implicitly*.

Explicitly defined modules further split into:
 * parametric layer (as linear or convolutions)
 * non-parametric layer (as tanh or relu)

Implicitly defined modules can be:
 * composition of other modules ( :ref:`Introduction to sequential<sequential>` )
 * arbitrary function of other modules (as skip-connection or res-block or attention)
 
Efficient implementation of Jacobian products are based on different levels of abstraction. 
Explicitly defined modules allow access to the explicit form of the Jacobian product and this allows to implement them **without ever instantiating the full Jacobian matrix** (which is memory consuming and practically often means storing a bunch of useless zeros).
Implicitly defined modules, instead, rely on the chain rule and on the efficient implementation of the Jacobian product of the building blocks.




Getting Started
=================
Defining a nnj neural network can be done either directly as


.. code-block:: python

   import nnj

   # Define you sequential model in nnj
   model = nnj.Sequential(
      nnj.Linear(),
      nnj.Tanh(),
      nnh.Linear(),
   )

or through standard torch.nn and convertion

.. code-block:: python

   import torch.nn as nn
   import nnj

   # Define you sequential model in torch.nn
   model = torch.nn.Sequential(
      nn.Linear(),
      nn.Tanh(),
      nn.Linear(),
   )

   # convert to nnj
   model = nnj.utils.convert_to_nnj(
      model,
   )

And computing jacobian products is as simple as

.. code-block:: python

    val = model(x)

    # Compute gradient (of the l2 loss) as backward pass of the residual
    residual = val - target
    gradient = model.vjp(x, val, residual, wrt="weight")

    # Compute the Generalized-Gauss Newton (an approximation of the hessian) as a backward pass of the Euclidean metric
    jacobianTranspose_jacobian = model._jTmjp(
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