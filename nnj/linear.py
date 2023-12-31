from typing import Literal, Tuple, Union

import torch
from torch import nn, Tensor

from nnj.abstract_jacobian import AbstractJacobian


class Linear(nn.Linear, AbstractJacobian):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_params = sum([torch.numel(w) for w in list(self.parameters())])

    @torch.no_grad()
    def jacobian(
        self,
        x: Tensor,
        val: Union[Tensor, None] = None,
        wrt: Literal["input", "weight"] = "input",
    ) -> Tensor:
        """Returns the Jacobian matrix"""
        b, c1 = x.shape
        if wrt == "input":
            return self.weight.unsqueeze(0).expand(b, *self.weight.shape)
        elif wrt == "weight":
            if val is None:
                val = self.forward(x)
            c2 = val.shape[1]
            out_identity = torch.diag_embed(torch.ones(c2, dtype=val.dtype, device=val.device))
            jacobian = torch.einsum("bk,ij->bijk", x, out_identity).reshape(b, c2, c2 * c1)
            if self.bias is not None:
                jacobian = torch.cat([jacobian, out_identity.unsqueeze(0).expand(b, c2, c2)], dim=2)
            return jacobian

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
        if wrt == "input":
            return torch.einsum("kj,bj->bk", self.weight, vector)
        elif wrt == "weight":
            b, c1 = x.shape
            if val is None:
                val = self.forward(x)
            c2 = val.shape[1]
            assert self._n_params == vector.shape[1]
            if self.bias is None:
                return torch.einsum("bkj,bj->bk", vector.reshape(b, c2, c1), x)
            else:
                return torch.einsum("bkj,bj->bk", vector[:, : c2 * c1].reshape(b, c2, c1), x) + vector[:, c2 * c1 :]

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
        if wrt == "input":
            return torch.einsum("kj,bji->bki", self.weight, matrix)
        elif wrt == "weight":
            b, c1 = x.shape
            if val is None:
                val = self.forward(x)
            c2 = val.shape[1]
            assert self._n_params == matrix.shape[1]
            if self.bias is None:
                return torch.einsum("bkji,bj->bki", matrix.reshape(b, c2, c1, -1), x)
            else:
                return (
                    torch.einsum("bkji,bj->bki", matrix[:, : c2 * c1, :].reshape(b, c2, c1, -1), x)
                    + matrix[:, c2 * c1 :, :]
                )

    @torch.no_grad()
    def jmjTp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: Literal["input", "weight"] = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tensor:
        """
        jacobian matrix jacobian.T product
        """
        if val is None:
            val = self.forward(x)
        if wrt == "input":
            if matrix is None:
                matrix = torch.ones_like(x)
                from_diag = True
            if not from_diag and not to_diag:
                # full -> full
                return torch.einsum("mn,bnj,kj->bmk", self.weight, matrix, self.weight)
            elif from_diag and not to_diag:
                # diag -> full
                return torch.einsum("mn,bn,kn->bmk", self.weight, matrix, self.weight)
            elif not from_diag and to_diag:
                # full -> diag
                return torch.einsum("mn,bnj,mj->bm", self.weight, matrix, self.weight)
            elif from_diag and to_diag:
                # diag -> diag
                return torch.einsum("mn,bn,mn->bm", self.weight, matrix, self.weight)
        elif wrt == "weight":
            if matrix is None:
                matrix = torch.ones((x.shape[0], self._n_params), dtype=x.dtype, device=x.device)
                from_diag = True
            if not from_diag and not to_diag:
                # full -> full
                matrixT = matrix.transpose(1, 2)
                jmTp = self.jmp(x, val, matrixT, wrt=wrt)
                mjTp = jmTp.transpose(1, 2)
                jmjTp = self.jmp(x, val, mjTp, wrt=wrt)
                return jmjTp
            elif from_diag and not to_diag:
                # diag -> full
                bs, _ = matrix.shape
                c1 = x.shape[1]
                c2 = val.shape[1]
                x_sq = x * x
                if self.bias is None:
                    return torch.diag_embed(torch.einsum("bi,bji->bj", x_sq, matrix.reshape(bs, c2, c1)))
                else:
                    return torch.diag_embed(
                        torch.einsum("bi,bji->bj", x_sq, matrix[:, : c2 * c1].reshape(bs, c2, c1))
                        + matrix[:, c2 * c1 :]
                    )
            elif not from_diag and to_diag:
                # full -> diag
                # TODO: do it properly
                return torch.einsum("bii -> bi", self.jmjTp(x, val, matrix, wrt=wrt, from_diag=False, to_diag=False))
            elif from_diag and to_diag:
                # diag -> diag
                bs, _ = matrix.shape
                c1 = x.shape[1]
                c2 = val.shape[1]
                x_sq = x * x
                if self.bias is None:
                    return torch.einsum("bi,bji->bj", x_sq, matrix.reshape(bs, c2, c1))
                else:
                    return (
                        torch.einsum("bi,bji->bj", x_sq, matrix[:, : c2 * c1].reshape(bs, c2, c1))
                        + matrix[:, c2 * c1 :]
                    )

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
    ) -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            return torch.einsum("bj,jk->bk", vector, self.weight)
        elif wrt == "weight":
            b, l = x.shape
            if self.bias is None:
                return torch.einsum("bi,bj->bij", vector, x).reshape(b, -1)
            else:
                return torch.cat([torch.einsum("bi,bj->bij", vector, x).reshape(b, -1), vector], dim=1)

    @torch.no_grad()
    def mjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: Literal["input", "weight"] = "input",
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if matrix is None:
            return self.jacobian(x, val, wrt=wrt)
        if wrt == "input":
            return torch.einsum("bij,jk->bik", matrix, self.weight)
        elif wrt == "weight":
            b, c1 = x.shape
            r = matrix.shape[1]
            assert x.shape[0] == matrix.shape[0]
            if self.bias is None:
                return torch.einsum("bri,bj->brij", matrix, x).reshape(b, r, -1)
            else:
                return torch.cat([torch.einsum("bri,bj->brij", matrix, x).reshape(b, r, -1), matrix], dim=2)

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
    ) -> Tensor:
        """
        jacobian.T matrix jacobian product
        """
        if val is None:
            val = self.forward(x)
        if matrix is None:
            matrix = torch.ones_like(val)
            from_diag = True
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return torch.einsum("nm,bnj,jk->bmk", self.weight, matrix, self.weight)
            elif from_diag and not to_diag:
                # diag -> full
                return torch.einsum("nm,bn,nk->bmk", self.weight, matrix, self.weight)
            elif not from_diag and to_diag:
                # full -> diag
                return torch.einsum("nm,bnj,jm->bm", self.weight, matrix, self.weight)
            elif from_diag and to_diag:
                # diag -> diag
                return torch.einsum("nm,bn,nm->bm", self.weight, matrix, self.weight)
        elif wrt == "weight":
            if not from_diag and not to_diag:
                # full -> full
                matrixT = matrix.transpose(1, 2)
                mTjp = self.mjp(x, val, matrixT, wrt=wrt)
                jTmp = mTjp.transpose(1, 2)
                jTmjp = self.mjp(x, val, jTmp, wrt=wrt)
                return jTmjp
            elif from_diag and not to_diag:
                # diag -> full
                # TODO: improve efficiency (?)
                bs, c1 = x.shape
                c2 = matrix.shape[1]
                x_outer = torch.einsum("bi,bj->bij", x, x)
                matrix = torch.diag_embed(matrix)
                if self.bias is None:
                    return torch.einsum("bij,bkq->bkiqj", x_outer, matrix).reshape(bs, c1 * c2, c1 * c2)
                else:
                    first_block = torch.einsum("bij,bkq->bkiqj", x_outer, matrix).reshape(bs, c1 * c2, c1 * c2)
                    outer_diag_block = torch.einsum("bi,bjk->bjik", x, matrix).reshape(bs, c1 * c2, c2)
                    return torch.cat(
                        [
                            torch.cat([first_block, outer_diag_block], dim=2),
                            torch.cat([outer_diag_block.transpose(1, 2), matrix], dim=2),
                        ],
                        dim=1,
                    )
            elif not from_diag and to_diag:
                # full -> diag
                bs, _, _ = matrix.shape
                x_sq = x * x
                if self.bias is None:
                    return torch.einsum("bj,bii->bij", x_sq, matrix).reshape(bs, -1)
                else:
                    return torch.cat(
                        [
                            torch.einsum("bj,bii->bij", x_sq, matrix).reshape(bs, -1),
                            torch.einsum("bii->bi", matrix),
                        ],
                        dim=1,
                    )
            elif from_diag and to_diag:
                # diag -> diag
                bs, _ = matrix.shape
                x_sq = x * x
                if self.bias is None:
                    return torch.einsum("bj,bi->bij", x_sq, matrix).reshape(bs, -1)
                else:
                    return torch.cat([torch.einsum("bj,bi->bij", x_sq, matrix).reshape(bs, -1), matrix], dim=1)
