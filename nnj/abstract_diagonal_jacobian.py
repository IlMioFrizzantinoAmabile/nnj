from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from nnj.abstract_jacobian import AbstractJacobian


class AbstractDiagonalJacobian(AbstractJacobian):
    """
    Superclass specific for layers whose Jacobian is a diagonal matrix.
    In these cases the forward and backward functions can be efficiently implemented in a general form
    """

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input", diag: bool = False
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        # this function has to be implemented for every new nnj layer
        raise NotImplementedError

    ######################
    ### forward passes ###
    ######################

    def _jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        jacobian vector product
        """
        diag_jacobian = self._jacobian(x, val, wrt=wrt, diag=True)
        if diag_jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bj,bj->bj", diag_jacobian, vector)

    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        jacobian matrix product
        """
        if matrix is None:
            return self._jacobian(x, val, wrt=wrt, diag=False)
        diag_jacobian = self._jacobian(x, val, wrt=wrt, diag=True)
        if diag_jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bi,bi...->bi...", diag_jacobian, matrix)

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
        """
        jacobian matrix jacobian.T product
        """
        b = x.shape[0]
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(x).reshape(b, -1)
            from_diag = True
        diag_jacobian = self._jacobian(x, val, wrt=wrt, diag=True)
        if diag_jacobian is None:  # non parametric layer
            return None
        if diag_backprop:
            raise NotImplementedError
        if not from_diag and not to_diag:
            # full -> full
            return torch.einsum("bi,bik,bk->bik", diag_jacobian, matrix, diag_jacobian)
        elif from_diag and not to_diag:
            # diag -> full
            diag_jacobian_square = diag_jacobian**2
            return torch.diag_embed(torch.einsum("bi,bi->bi", diag_jacobian_square, matrix))
        elif not from_diag and to_diag:
            # full -> diag
            return torch.einsum("bi,bii,bi->bi", diag_jacobian, matrix, diag_jacobian)
        elif from_diag and to_diag:
            # diag -> diag
            diag_jacobian_square = diag_jacobian**2
            return torch.einsum("bi,bi->bi", diag_jacobian_square, matrix)

    #######################
    ### backward passes ###
    #######################

    def _vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        vector jacobian product
        """
        diag_jacobian = self._jacobian(x, val, wrt=wrt, diag=True)
        if diag_jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bi,bi->bi", vector, diag_jacobian)

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Union[Tensor, None]:
        """
        matrix jacobian product
        """
        if matrix is None:
            return self._jacobian(x, val, diag=False)
        diag_jacobian = self._jacobian(x, val, diag=True)
        if diag_jacobian is None:  # non parametric layer
            return None
        return torch.einsum("bij,bj->bij", matrix, diag_jacobian)

    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
        """
        jacobian.T matrix jacobian product
        """
        b = x.shape[0]
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val).reshape(b, -1)
            from_diag = True
        diag_jacobian = self._jacobian(x, val, wrt=wrt, diag=True)
        if diag_jacobian is None:  # non parametric layer
            return None
        if not from_diag and not to_diag:
            # full -> full
            return torch.einsum("bi,bik,bk->bik", diag_jacobian, matrix, diag_jacobian)
        elif from_diag and not to_diag:
            # diag -> full
            diag_jacobian_square = diag_jacobian**2
            return torch.diag_embed(torch.einsum("bi,bi->bi", diag_jacobian_square, matrix))
        elif not from_diag and to_diag:
            # full -> diag
            return torch.einsum("bi,bii,bi->bi", diag_jacobian, matrix, diag_jacobian)
        elif from_diag and to_diag:
            # diag -> diag
            diag_jacobian_square = diag_jacobian**2
            return torch.einsum("bi,bi->bi", diag_jacobian_square, matrix)
