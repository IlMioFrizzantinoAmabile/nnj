.. _introduction:

Introduction
===================================

Blablabla


Notation
===================================

Given a data space :math:`\mathcal{X}` and a label space :math:`\mathcal{Y}`, consider a Neural Network (NN) :math:`f_\theta:\mathcal{X}\rightarrow\mathcal{Y}` with  :math:`L` layers. The parameter  :math:`\theta = (\theta_1, \dots, \theta_L) \in\Theta` is the concatenation of the parameters :math:`\theta_i` for each layer  :math:`i \in \{1,...,L \}`. 

The NN is a composition of  :math:`L` functions  :math:`f^{(1)},f^{(2)},\dots,f^{(L)}`, where :math:`f^{(i)}` is parametrized by :math:`\theta_{i}`.

.. math::
    \begin{aligned}
    f_\theta
        :=
        f^{(L)}_{\theta_L}\circ f^{(L-1)}_{\theta_{L-1}} 
        \circ\,\dots\,\circ 
        f^{(2)}_{\theta_2} \circ f^{(1)}_{\theta_1}.
    \end{aligned}


Since we need explicit access to the intermediate values, we call the input :math:`x_0\in\mathcal{X}` and iteratively define :math:`x_i:=f^{(i)}_{\theta_i}(x_{i-1})` for :math:`i=1,\dots,L`, such that the NN output is :math:`x_L\in\mathcal{Y}`. This notation can be visually presented as

.. math::
    \begin{aligned}
    \mathcal{X}
    & \overset{f_\theta}{\xrightarrow{\qquad\qquad\text{a}}}
        \mathcal{Y} 
    \\
    x_0 
    & \underset{f^{(1)}_{\theta_1}}{\xrightarrow{\quad}} x_1 \longrightarrow 
    \quad\dots\quad 
    \longrightarrow 
    x_{i-1} \underset{f^{(i)}_{\theta_i}}{\xrightarrow{\quad}} x_i \longrightarrow 
    \quad\dots\quad 
    \longrightarrow
    x_{L-1} \underset{f^{(L)}_{\theta_L}}{\xrightarrow{\quad}} 
    x_L
    \end{aligned}



:math:`\xrightarrow{\hspace*{11em}}`
:math:`  `





Chain rule of Jacobian
======================================================================
We are interested in the Jacobian :math:`\nabla_\theta f_\theta(x_0)` of the NN with respect to the parameter :math:`\theta`. Each column of the Jacobian is the derivative of the output vector w.r.t.\@ a single parameter. We can then group the parameters (i.e. columns) layer by layer

.. math::
    \begin{align*}
    J_\theta f_\theta(x_0) 
    & = 
    \left(\begin{array}{c|c|c|c|c}
        & & & &\\
        J_{\theta_1}f_\theta(x_0) &
        \,\dots\, &
        J_{\theta_i}f_\theta(x_0) &
        \,\dots\, &
        J_{\theta_L}f_\theta(x_0) \\
        & & & &
    \end{array}\right)
    \\
    & = 
    \left(\begin{array}{c|c|c|c|c}
        & & & & \\
        J_{\theta_1}
        \left(
            f^{(1)}_{\theta_1}
            \circ\dots\circ
            f^{(L)}_{\theta_L}
        \right)
        (x_0) &
        \,\dots\, &
        J_{\theta_i}
        \left(
            f^{(i)}_{\theta_i}
            \circ\dots\circ
            f^{(L)}_{\theta_L}
        \right)
        (x_{i-1}) &
        \,\dots\, &
        J_{\theta_L}f^{(L)}_{\theta_L}(x_{L-1}) \\
        & & & &
    \end{array}\right),
    \end{align*}

where the second equality comes from the fact that each layer only depends on its respective parameters, i.e.

.. math::
    J_{\theta_j} f^{(i)}_{\theta_i} (x_{i-1}) = 0 
    \quad \text{ if }i\not=j.

Exploiting this block-structure, we can focus on a single layer Jacobian :math:`J_{\theta_i}f_{\theta}(x_0)`, and concatenate them afterwards. With the chain rule we get 

.. math::
    \begin{equation}
    J_{\theta_i}f_{\theta}(x_0)
    =
    J_{\theta_i}
    \left(
            f^{(i)}_{\theta_i}
            \circ\dots\circ
            f^{(L)}_{\theta_L}
        \right)
    (x_{i-1}) 
    =
    \left(
        \prod_{j=L}^{i+1} 
        J_{x_{j-1}}f^{(j)}_{\theta_j}(x_{j-1})
    \right)
    J_{\theta_i}f^{(i)}_{\theta_i}(x_{i-1}).
    \end{equation}

The intuition for the chain rule is that the Jacobian :math:`J_{\theta_i}f_{\theta}(x_0)` for layer :math:`i` is the composition of the Jacobians w.r.t.\@ the \emph{input} :math:`J_{x_{j-1}}f^{(j)}_{\theta_j}(x_{j-1})` of subsequent layers :math:`j=L,L-1,\dots,i+2,i+1`, times 
the Jacobian w.r.t.\@ the \emph{parameters} :math:`J_{\theta_i}f^{(i)}_{\theta_i}(x_{i-1})` of the specific layer :math:`i`. Thus, we can reuse computation for one layer to improve the computation of other layers, specifically the product of Jacobians w.r.t.\@ the input.



Chain rule of Hessian (backpropagation of Hessian)
======================================================================





NNJ: A more efficient backpropagation of Hessian approximations
======================================================================




Getting Started
===================================