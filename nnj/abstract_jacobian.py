from typing import List, Literal, Union

import torch
from torch import Tensor


class AbstractJacobian:
    """Abstract class that:
        - will overwrite the default behaviour of the forward method such that it
        is also possible to return the jacobian
        - propagate jacobian vector and jacobian matrix products, both forward and backward
        - pull back and push forward metrics
    """

    def jacobian(
        self,
        x: Tensor,
        val: Union[Tensor, None] = None,
        wrt: Literal["input", "weight"] = "input",
    ) -> Union[Tensor, None]:
        """
        Compute the Jacobian matrix of the layer evaluated in x

        .. math::
            ∇_{wrt} \,\, layer(x)

        .. note::
            This method has to be implemented for every new nnj layer. Then all other jacobian products are usable.

        Args:
            x: The input of the layer.
            val: The output of the layer.
            wrt: The variable with respect to the derivative is computed: "input" for x, "weight" for parameters.

        """
        raise NotImplementedError

    ######################
    ### forward passes ###
    ######################

    def jvp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        vector: Tensor,
        wrt: Literal["input", "weight"] = "input",
    ) -> Union[Tensor, None]:
        """
        Returns the Jacobian vector product

        .. math::
            jvp(x,vector) = ∇_{wrt} \,\, layer(x) * vector
    
        Args:
            x: The input of the layer.
            val: The output of the layer.
            vector: The vector in the tangent space to propagate. It has to be of same shape of x if wrt="weight", and the same shape as parameter if wrt="input". 
            wrt: The variable with respect to the derivative is computed: "input" for x, "weight" for parameters.

        Shape:
            - x: 
            - val:
            - vector:
            - output:
        """
        jacobian = self.jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bij,bj->bi", jacobian, vector)

    def jmp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Tensor,
        wrt: Literal["input", "weight"] = "input",
    ) -> Union[Tensor, None]:
        """
        jacobian matrix product
        """
        jacobian = self.jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bij,bjk->bik", jacobian, matrix)

    def jmjTp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Tensor,
        wrt: Literal["input", "weight"] = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
        """
        jacobian matrix jacobian.T product
        """
        if diag_backprop:  # TODO
            raise NotImplementedError
        jacobian = self.jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        if not from_diag and not to_diag:
            # full -> full
            return torch.einsum("bij,bjk,blk->bil", jacobian, matrix, jacobian)
        elif from_diag and not to_diag:
            # diag -> full
            return torch.einsum("bij,bj,blj->bil", jacobian, matrix, jacobian)
        elif not from_diag and to_diag:
            # full -> diag
            return torch.einsum("bij,bjk,bik->bi", jacobian, matrix, jacobian)
        elif from_diag and to_diag:
            # diag -> diag
            return torch.einsum("bij,bj,bij->bi", jacobian, matrix, jacobian)

    #######################
    ### backward passes ###
    #######################

    def vjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        vector: Tensor,
        wrt: Literal["input", "weight"] = "input",
    ) -> Union[Tensor, None]:
        """
        vector jacobian product
        """
        jacobian = self.jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bi,bij->bj", vector, jacobian)

    def mjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Tensor,
        wrt: Literal["input", "weight"] = "input",
    ) -> Union[Tensor, None]:
        """
        matrix jacobian product
        """
        jacobian = self.jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bij,bjk->bik", matrix, jacobian)

    def jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Tensor,
        wrt: Literal["input", "weight"] = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, List[Tensor], None]:
        """
        jacobian.T matrix jacobian product
        """
        if diag_backprop:
            # TODO: better error message
            raise NotImplementedError
        jacobian = self.jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        if not from_diag and not to_diag:
            # full -> full
            return torch.einsum("bji,bjk,bkl->bil", jacobian, matrix, jacobian)
        elif from_diag and not to_diag:
            # diag -> full
            return torch.einsum("bij,bj,bjl->bil", jacobian, matrix, jacobian)
        elif not from_diag and to_diag:
            # full -> diag
            return torch.einsum("bij,bjk,bki->bi", jacobian, matrix, jacobian)
        elif from_diag and to_diag:
            # diag -> diag
            return torch.einsum("bij,bj,bji->bi", jacobian, matrix, jacobian)
