import torch
import torch.nn.functional as F
from torch import nn, Tensor

from typing import Optional, Tuple, List, Union


class AbstractJacobian:
    """Abstract class that:
    - will overwrite the default behaviour of the forward method such that it
    is also possible to return the jacobian
    - propagate jacobian vector and jacobian matrix products, both forward and backward
    - pull back and push forward metrics
    """

    def __init__(self) -> None:
        self._n_params = sum([torch.numel(w) for w in list(self.parameters())])

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        # this function has to be implemented for every new nnj layer
        raise NotImplementedError

    ######################
    ### forward passes ###
    ######################

    def _jvp(self, x: Tensor, val: Tensor, vector: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bij,bj->bi", jacobian, vector)

    def _jmp(self, x: Tensor, val: Tensor, matrix: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bij,bjk->bik", jacobian, matrix)

    def _jmjTp(
        self,
        x: Tensor,
        val: Tensor,
        matrix: Tensor,
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
        if diag_backprop:  # TODO
            raise NotImplementedError
        jacobian = self._jacobian(x, val, wrt=wrt)
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

    def _vjp(self, x: Tensor, val: Tensor, vector: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bi,bij->bj", vector, jacobian)

    def _mjp(self, x: Tensor, val: Tensor, matrix: Tensor, wrt: str = "input") -> Union[Tensor, None]:
        jacobian = self._jacobian(x, val, wrt=wrt)
        if jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bij,bjk->bik", matrix, jacobian)

    def _jTmjp(
        self,
        x: Tensor,
        val: Tensor,
        matrix: Tensor,
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, List[Tensor], None]:
        if diag_backprop:
            # TODO: better error message
            raise NotImplementedError
        jacobian = self._jacobian(x, val, wrt=wrt)
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
