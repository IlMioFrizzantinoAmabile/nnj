import torch
import torch.nn.functional as F
from torch import nn, Tensor
from nnj.abstract_jacobian import AbstractJacobian

from typing import Optional, Tuple, List, Union

class Linear(nn.Linear, AbstractJacobian):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_params = args[0]*args[1]
        if self.bias is not None:
            self._n_params += args[1]

    def forward(
        self, x: Tensor, jacobian: Union[Tensor, bool] = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        #val = self._call_impl(x)
        val = F.linear(x, self.weight, bias=self.bias)
        if jacobian is False:
            return val
        elif jacobian is True:
            jac = self._jacobian(x, val, wrt="input")
            return val, jac
        else:
            jac = self._jmp(x, val, jacobian, wrt='input')
            return val, jac

    def _jacobian(
        self, x: Tensor, val: Union[Tensor, None] = None, wrt: str = "input"
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

    def _jvp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
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
                return torch.einsum("bkj,bj->bk", vector.view(b, c2, c1), x)
            else:
                return torch.einsum("bkj,bj->bk", vector[:, :c2*c1].view(b, c2, c1), x) + vector[:, c2*c1:] 
    
    def _jmp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        jacobian matrix product
        """
        if matrix is None:
            return self._jacobian(x, val, wrt = wrt)
        if wrt == "input":
            return torch.einsum("kj,bji->bki", self.weight, matrix)
        elif wrt == "weight":
            b, c1 = x.shape
            if val is None:
                val = self.forward(x)
            c2 = val.shape[1]
            assert self._n_params == matrix.shape[1]
            if self.bias is None:
                return torch.einsum("bkji,bj->bki", matrix.view(b, c2, c1, -1), x)
            else:
                return torch.einsum("bkji,bj->bki", matrix[:, :c2*c1, :].view(b, c2, c1, -1), x) + matrix[:, c2*c1:, :]

    def _jmjTp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Union[Tensor, Tuple]:
        """
        jacobian matrix jacobian.T product
        """
        if matrix is None:
            matrix = torch.ones_like(x)
            from_diag = True
        if val is None:
            val = self.forward(x)
        if wrt == "input":
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
            if not from_diag and not to_diag:
                # full -> full
                matrixT = matrix.transpose(1,2)
                jmTp = self._jmp(x, val, matrixT, wrt=wrt)
                mjTp = jmTp.transpose(1,2)
                jmjTp = self._jmp(x, val, mjTp, wrt=wrt)
                return jmjTp
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag
                raise NotImplementedError

    #######################
    ### backward passes ###
    #######################

    def _vjp(
        self, x: Tensor, val: Union[Tensor, None], vector: Tensor, wrt: str = "input"
    ) -> Tensor:
        """
        vector jacobian product
        """
        if wrt == "input":
            return torch.einsum("bj,jk->bk", vector, self.weight)
        elif wrt == "weight":
            b, l = x.shape
            if self.bias is None:
                return torch.einsum("bi,bj->bij", vector, x).view(b, -1)
            else:
                return torch.cat([torch.einsum("bi,bj->bij", vector, x).view(b, -1), vector], dim=1)

    def _mjp(
        self, x: Tensor, val: Union[Tensor, None], matrix: Union[Tensor, None], wrt: str = "input"
    ) -> Tensor:
        """
        matrix jacobian product
        """
        if matrix is None:
            return self._jacobian(x, val, wrt = wrt)
        if wrt == "input":
            return torch.einsum("bij,jk->bik", matrix, self.weight)
        elif wrt == "weight":
            b, l = x.shape
            r = matrix.shape[1]
            assert x.shape[0]==matrix.shape[0]
            if self.bias is None:
                return torch.einsum("bri,bj->brij", matrix, x).view(b, r, -1)
            else:
                return torch.cat([torch.einsum("bri,bj->brij", matrix, x).view(b, r, -1), 
                                  matrix], dim=2)


    def _jTmjp(
        self,
        x: Tensor,
        val: Union[Tensor, None],
        matrix: Union[Tensor, None],
        wrt: str = "input",
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
                #TODO
                jacobian = self._jacobian(x, val, wrt=wrt)
                return torch.einsum("bji,bjk,bkq->biq", jacobian, matrix, jacobian)
            elif from_diag and not to_diag:
                # diag -> full
                #TODO
                jacobian = self._jacobian(x, val, wrt=wrt)
                return torch.einsum("bji,bj,bjq->biq", jacobian, matrix, jacobian)
            elif not from_diag and to_diag:
                # full -> diag
                bs, _, _ = matrix.shape
                x_sq = x * x
                if self.bias is None:
                    return torch.einsum("bj,bii->bij", x_sq, matrix).view(bs, -1)
                else:
                    return torch.cat(
                        [
                            torch.einsum("bj,bii->bij", x_sq, matrix).view(bs, -1),
                            torch.einsum("bii->bi", matrix),
                        ],
                        dim=1,
                    )
            elif from_diag and to_diag:
                # diag -> diag
                bs, _ = matrix.shape
                x_sq = x * x
                if self.bias is None:
                    return torch.einsum("bj,bi->bij", x_sq, matrix).view(bs, -1)
                else:
                    return torch.cat([torch.einsum("bj,bi->bij", x_sq, matrix).view(bs, -1), matrix], dim=1)

    def _jTmjp_batch2(
        self,
        x1: Tensor,
        x2: Tensor,
        val1: Union[Tensor, None],
        val2: Union[Tensor, None],
        matrixes: Union[Tuple[Tensor, Tensor, Tensor], None],
        wrt: str = "input",
        from_diag: bool = False,
        to_diag: bool = False,
        diag_backprop: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        jacobian.T matrix jacobian product, computed on 2 inputs (considering cross terms)
        """
        assert x1.shape == x2.shape
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full
                return tuple(
                    torch.einsum("nm,bnj,jk->bmk", self.weight, m, self.weight) for m in matrixes
                )
            elif from_diag and not to_diag:
                # diag -> full
                return tuple(
                    torch.einsum("nm,bn,nk->bmk", self.weight, m_diag, self.weight) for m_diag in matrixes
                )
            elif not from_diag and to_diag:
                # full -> diag
                return tuple(
                    torch.einsum("nm,bnj,jm->bm", self.weight, m, self.weight) for m in matrixes
                )
            elif from_diag and to_diag:
                # diag -> diag
                return tuple(
                    torch.einsum("nm,bn,nm->bm", self.weight, m_diag, self.weight) for m_diag in matrixes
                )
        elif wrt == "weight":
            if val1 is None:
                val1 = self.forward(x1)
            if val2 is None:
                val2 = self.forward(x2)
            if not from_diag and not to_diag:
                # full -> full
                m11, m12, m22 = matrixes
                jac_1 = self._jacobian_wrt_weight(x1, val1)
                jac_2 = self._jacobian_wrt_weight(x2, val2)
                return tuple(
                    torch.einsum("bji,bjk,bkq->biq", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [(jac_1, m11, jac_1), (jac_1, m12, jac_2), (jac_2, m22, jac_2)]
                )
            elif from_diag and not to_diag:
                # diag -> full
                m11, m12, m22 = matrixes
                jac_1 = self._jacobian_wrt_weight(x1, val1)
                jac_2 = self._jacobian_wrt_weight(x2, val2)
                return tuple(
                    torch.einsum("bji,bj,bjk->bik", jac_i, m, jac_j)
                    for jac_i, m, jac_j in [(jac_1, m11, jac_1), (jac_1, m12, jac_2), (jac_2, m22, jac_2)]
                )
            elif not from_diag and to_diag:
                # full -> diag
                m11, m12, m22 = matrixes
                bs, _ = x1.shape
                if self.bias is None:
                    return tuple(
                        torch.einsum("bj,bii,bj->bij", x_i, m, x_j).view(bs, -1)
                        for x_i, m, x_j in [(x1, m11, x1), (x1, m12, x2), (x2, m22, x2)]
                    )
                else:
                    return tuple(
                        torch.cat(
                            [
                                torch.einsum("bj,bii,bj->bij", x_i, m, x_j).view(bs, -1),
                                torch.einsum("bii->bi", m),
                            ],
                            dim=1,
                        )
                        for x_i, m, x_j in [(x1, m11, x1), (x1, m12, x2), (x2, m22, x2)]
                    )
            elif from_diag and to_diag:
                # diag -> diag
                m11_diag, m12_diag, m22_diag = matrixes
                bs, _ = x1.shape
                if self.bias is None:
                    return tuple(
                        torch.einsum("bj,bi,bj->bij", x_i, m_diag, x_j).view(bs, -1)
                        for x_i, m_diag, x_j in [(x1, m11_diag, x1), (x1, m12_diag, x2), (x2, m22_diag, x2)]
                    )
                else:
                    return tuple(
                        torch.cat(
                            [torch.einsum("bj,bi,bj->bij", x_i, m_diag, x_j).view(bs, -1), m_diag], dim=1
                        )
                        for x_i, m_diag, x_j in [(x1, m11_diag, x1), (x1, m12_diag, x2), (x2, m22_diag, x2)]
                    )
