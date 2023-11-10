from typing import List, Literal, Tuple, Union

import torch
from torch import nn, Tensor

from nnj.abstract_jacobian import AbstractJacobian
from nnj.sequential import Sequential


class SkipConnection(nn.Module, AbstractJacobian):
    def __init__(self, *args, add_hooks: bool = True):
        super().__init__()

        self._F = Sequential(*args, add_hooks=add_hooks)
        self._n_params = self._F._n_params

    def forward(self, x: Tensor):
        return torch.cat([x, self._F(x)], dim=1)

    @torch.no_grad()
    def jacobian(
        self,
        x: Tensor,
        val: Union[Tensor, None] = None,
        wrt: Literal["input", "weight"] = "input",
    ) -> Tensor:
        """Returns the Jacobian matrix"""
        jac = self._F.jacobian(x, val, wrt=wrt)
        b = x.shape[0]
        l = x[0].numel()
        if val is None:
            val = self.forward(x)
        if wrt == "input":
            identity = torch.diag_embed(torch.ones_like(x).reshape(b, l))
            return torch.cat([identity, jac], dim=1)
        elif wrt == "weight":
            zeros = torch.zeros(b, l, self._n_params, device=x.device, dtype=x.dtype)
            return torch.cat([zeros, jac], dim=1)

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
    ) -> Tensor:
        """
        jacobian vector product
        """
        b = x.shape[0]
        l = x[0].numel()
        jvp = self._F.jvp(x, None if val is None else val[:, l:], vector, wrt=wrt)
        if wrt == "input":
            return torch.cat([vector, jvp], dim=1)
        elif wrt == "weight":
            zeros = torch.zeros_like(x).reshape(b, l)
            return torch.cat([zeros, jvp], dim=1)

    @torch.no_grad()
    def jmp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: Literal["input", "weight"] = "input",
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if matrix is None:
            return self.jacobian(x, val, wrt=wrt)
        b = x.shape[0]
        l = x[0].numel()
        jmp = self._F.jmp(x, None if val is None else val[:, l:], matrix, wrt=wrt)
        if wrt == "input":
            return torch.cat([matrix, jmp], dim=1)
        elif wrt == "weight":
            zeros = torch.zeros(b, l, matrix.shape[2], device=x.device, dtype=x.dtype)
            return torch.cat([zeros, jmp], dim=1)

    @torch.no_grad()
    def jmjTp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, List, None],
        wrt: Literal["input", "weight"] = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """
        jacobian matrix jacobian.T product
        """
        assert (not diag_backprop) or (diag_backprop and from_diag and to_diag)
        if val is None:
            val = self.forward(x)
        b = x.shape[0]
        l1, l2 = x[0].numel(), val[0].numel()
        jmjTp = self._F.jmjTp(
            x,
            None if val is None else val[:, l1:],
            matrix,
            wrt=wrt,
            from_diag=from_diag,
            to_diag=to_diag,
            diag_backprop=diag_backprop,
        )
        if wrt == "input":
            if to_diag:
                if from_diag:
                    return torch.cat([matrix, jmjTp], dim=1)
                if not from_diag:
                    return torch.cat([torch.einsum("bii->bi", matrix), jmjTp], dim=1)
            if from_diag:
                matrix = torch.diag_embed(matrix)
                jmp = self._F.jmp(x, None if val is None else val[:, l1:], matrix, wrt=wrt)
                return torch.cat(
                    [torch.cat([matrix, jmp.transpose(1, 2)], dim=2), torch.cat([jmp, jmjTp], dim=2)], dim=1
                )
            jmp = self._F.jmp(x, None if val is None else val[:, l1:], matrix, wrt=wrt)
            mjTp = self._F.jmp(x, None if val is None else val[:, l1:], matrix.transpose(1, 2), wrt=wrt).transpose(1, 2)
            return torch.cat([torch.cat([matrix, mjTp], dim=2), torch.cat([jmp, jmjTp], dim=2)], dim=1)
        elif wrt == "weight":
            if to_diag:
                return torch.cat([torch.zeros_like(x).reshape(b, l1), jmjTp], dim=1)
            zeros = torch.zeros(b, l2 - l1, l1, device=x.device, dtype=x.dtype)
            more_zeros = torch.zeros(b, l1, l2, device=x.device, dtype=x.dtype)
            return torch.cat([more_zeros, torch.cat([zeros, jmjTp], dim=2)], dim=1)

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
        l = x[0].numel()
        vjp = self._F.vjp(x, None if val is None else val[:, l:], vector[:, l:], wrt=wrt)
        if wrt == "input":
            return vector[:, :l] + vjp
        elif wrt == "weight":
            return vjp

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
        if matrix is None:
            return self.jacobian(x, val, wrt=wrt)
        l = x[0].numel()
        mjp = self._F.mjp(x, None if val is None else val[:, l:], matrix[:, :, l:], wrt=wrt)
        if wrt == "input":
            return matrix[:, :, :l] + mjp
        elif wrt == "weight":
            return mjp

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
    ) -> Union[Tensor, List, None]:
        """
        jacobian.T matrix jacobian product
        """
        assert (not diag_backprop) or (diag_backprop and from_diag and to_diag)
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val)
            from_diag = True
        l = x[0].numel()
        jTmjp = self._F.jTmjp(
            x,
            None if val is None else val[:, l:],
            matrix[:, l:, l:] if not from_diag else matrix[:, l:],
            wrt=wrt,
            from_diag=from_diag,
            to_diag=to_diag,
            diag_backprop=diag_backprop,
        )
        if wrt == "input":
            if from_diag and to_diag:
                return jTmjp + matrix[:, :l]
            if from_diag:
                return jTmjp + torch.diag_embed(matrix[:, :l])
            mjp = self._F.mjp(x, None if val is None else val[:, l:], matrix[:, :l, l:], wrt=wrt)
            jTmp = self._F.mjp(
                x, None if val is None else val[:, l:], matrix[:, l:, :l].transpose(1, 2), wrt=wrt
            ).transpose(1, 2)
            if to_diag:
                return (
                    jTmjp
                    + torch.einsum("bii->bi", mjp)
                    + torch.einsum("bii->bi", jTmp)
                    + torch.einsum("bii->bi", matrix[:, :l, :l])
                )
            return jTmjp + mjp + jTmp + matrix[:, :l, :l]
        elif wrt == "weight":
            return jTmjp
