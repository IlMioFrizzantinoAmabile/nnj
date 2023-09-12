from typing import Literal, Union

import torch
from torch import nn, Tensor

from nnj.abstract_diagonal_jacobian import AbstractDiagonalJacobian


class Reshape(AbstractDiagonalJacobian, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dims = args
        self._n_params = 0

    # @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        val = x.reshape(x.shape[0], *self.dims)
        return val

    @torch.no_grad()
    def jacobian(
        self,
        x: Tensor,
        val: Union[Tensor, None] = None,
        wrt: Literal["input", "weight"] = "input",
        diag: bool = False,
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            diag_jacobian = torch.ones_like(x).reshape(x.shape[0], -1)
            if diag:
                return diag_jacobian
            else:
                return torch.diag_embed(diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

    ######################
    ### forward passes ###
    ######################

    @torch.no_grad()
    def jvp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        vector: Tensor,
        wrt: Literal["input", "weight"] = "input",
    ) -> Union[Tensor, None]:
        """
        jacobian vector product
        """
        if wrt == "input":
            return vector
        elif wrt == "weight":  # non parametric layer
            return None

    @torch.no_grad()
    def jmp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: Literal["input", "weight"] = "input",
    ) -> Union[Tensor, None]:
        """
        jacobian matrix product
        """
        if wrt == "input":
            if matrix is None:
                raise NotImplementedError
            return matrix
        elif wrt == "weight":  # non parametric layer
            return None

    @torch.no_grad()
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
        if matrix is None:
            b = x.shape[0]
            matrix = torch.ones_like(x).reshape(b, -1)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return matrix
            elif from_diag and not to_diag:
                # diag -> full
                return torch.diag_embed(matrix)
            elif not from_diag and to_diag:
                # full -> diag
                return torch.einsum("bii->bi", matrix)
            elif from_diag and to_diag:
                # diag -> diag
                return matrix
        elif wrt == "weight":  # non parametric layer
            return None

    #######################
    ### backward passes ###
    #######################

    @torch.no_grad()
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
        if wrt == "input":
            return vector
        elif wrt == "weight":  # non parametric layer
            return None

    @torch.no_grad()
    def mjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: Literal["input", "weight"] = "input",
    ) -> Union[Tensor, None]:
        """
        matrix jacobian product
        """
        if wrt == "input":
            if matrix is None:
                raise NotImplementedError
            return matrix
        elif wrt == "weight":  # non parametric layer
            return None

    @torch.no_grad()
    def jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: Literal["input", "weight"] = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, None]:
        """
        jacobian.T matrix jacobian product
        """
        if val is None:
            val = self.forward(x)
        if matrix is None:
            b = x.shape[0]
            matrix = torch.ones_like(val).reshape(b, -1)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return matrix
            elif from_diag and not to_diag:
                # diag -> full
                return torch.diag_embed(matrix)
            elif not from_diag and to_diag:
                # full -> diag
                return torch.einsum("bii->bi", matrix)
            elif from_diag and to_diag:
                # diag -> diag
                return matrix
        elif wrt == "weight":  # non parametric layer
            return None
