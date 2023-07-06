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

    def jacobian(self, x: Tensor, val: Union[Tensor, None] = None, wrt: Literal = "input") -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        # this function has to be implemented for every new nnj layer
        raise NotImplementedError

    ######################
    ### forward passes ###
    ######################

    def jvp(self, x: Tensor, val: Tensor, vector: Tensor, wrt: Literal = "input") -> Union[Tensor, None]:
        jacobian = self.jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bij,bj->bi", jacobian, vector)

    def jmp(self, x: Tensor, val: Tensor, matrix: Tensor, wrt: Literal = "input") -> Union[Tensor, None]:
        jacobian = self.jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bij,bjk->bik", jacobian, matrix)

    def jmjTp(
        self,
        x: Tensor,
        val: Tensor,
        matrix: Tensor,
        wrt: Literal = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
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

    def vjp(self, x: Tensor, val: Tensor, vector: Tensor, wrt: Literal = "input") -> Union[Tensor, None]:
        jacobian = self.jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bi,bij->bj", vector, jacobian)

    def mjp(self, x: Tensor, val: Tensor, matrix: Tensor, wrt: Literal = "input") -> Union[Tensor, None]:
        jacobian = self.jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bij,bjk->bik", matrix, jacobian)

    def jTmjp(
        self,
        x: Tensor,
        val: Tensor,
        matrix: Tensor,
        wrt: Literal = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, List[Tensor], None]:
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
