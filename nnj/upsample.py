from typing import Literal, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from nnj.abstract_jacobian import AbstractJacobian


class Upsample(nn.Upsample, AbstractJacobian):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_params = 0

    @torch.no_grad()
    def jacobian(
        self,
        x: Tensor,
        val: Union[Tensor, None] = None,
        wrt: Literal["input", "weight"] = "input",
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            b, c1, h1, w1 = x.shape
            _, c2, h2, w2 = val.shape
            assert c1 == c2
            identity = torch.ones(b, c2 * h2 * w2, device=x.device)
            identity = torch.diag_embed(identity)
            j = self.mjp(x, val, identity, wrt="input")
            return j
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
            raise NotImplementedError
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
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
        if matrix is None:
            return self.jacobian(x, val, wrt=wrt)
        if wrt == "input":
            raise NotImplementedError
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None

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
    ) -> Union[Tensor, None]:
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
                raise NotImplementedError
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag
                raise NotImplementedError
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
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
            if val is None:
                val = self.forward(x)
            b, c1, h1, w1 = x.shape
            _, c2, h2, w2 = val.shape
            assert c1 == c2
            assert vector.shape == (b, c2 * h2 * w2)

            weight = torch.ones(1, 1, int(self.scale_factor), int(self.scale_factor), device=x.device)

            vector_J = F.conv2d(
                vector.reshape(b * c2, 1, h2, w2),
                weight=weight,
                bias=None,
                stride=int(self.scale_factor),
                padding=0,
                dilation=1,
                groups=1,
            ).reshape(b, c1 * h1 * w1)

            return vector_J
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
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
        if matrix is None:
            return self.jacobian(x, val, wrt=wrt)
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            b, c1, h1, w1 = x.shape
            _, c2, h2, w2 = val.shape
            assert c1 == c2
            assert matrix.shape[0] == b and matrix.shape[2] == c2 * h2 * w2
            n_rows = matrix.shape[1]

            weight = torch.ones(1, 1, int(self.scale_factor), int(self.scale_factor), device=x.device)

            matrix_J = F.conv2d(
                matrix.reshape(b * n_rows * c2, 1, h2, w2),
                weight=weight,
                bias=None,
                stride=int(self.scale_factor),
                padding=0,
                dilation=1,
                groups=1,
            ).reshape(b, n_rows, c2 * h1 * w1)

            return matrix_J
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
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
        b, c1, h1, w1 = x.shape
        _, c2, h2, w2 = val.shape
        assert c1 == c2
        if matrix is None:
            matrix = torch.ones_like(val).reshape(b, -1)
            from_diag = True
        if from_diag:
            assert matrix.shape == (b, c2 * h2 * w2)
        else:
            assert matrix.shape == (b, c2 * h2 * w2, c2 * h2 * w2)
        if wrt == "input":
            if not from_diag and not to_diag:
                # full -> full

                weight = torch.ones(1, 1, int(self.scale_factor), int(self.scale_factor), device=x.device)

                matrix = matrix.reshape(b, c2, h2 * w2, c2, h2 * w2)
                matrix = matrix.movedim(2, 3)
                matrix_J = F.conv2d(
                    matrix.reshape(b * c2 * c2 * h2 * w2, 1, h2, w2),
                    weight=weight,
                    bias=None,
                    stride=int(self.scale_factor),
                    padding=0,
                    dilation=1,
                    groups=1,
                ).reshape(b * c2 * c2, h2 * w2, h1 * w1)

                Jt_matrixt = matrix_J.movedim(-1, -2)

                Jt_matrixt_J = F.conv2d(
                    Jt_matrixt.reshape(b * c2 * c2 * h1 * w1, 1, h2, w2),
                    weight=weight,
                    bias=None,
                    stride=int(self.scale_factor),
                    padding=0,
                    dilation=1,
                    groups=1,
                ).reshape(b * c2 * c2, h1 * w1, h1 * w1)

                Jt_matrix_J = Jt_matrixt_J.movedim(-1, -2)

                Jt_matrix_J = Jt_matrix_J.reshape(b, c2, c2, h1 * w1, h1 * w1)
                Jt_matrix_J = Jt_matrix_J.movedim(2, 3)
                Jt_matrix_J = Jt_matrix_J.reshape(b, c2 * h1 * w1, c2 * h1 * w1)

                return Jt_matrix_J
            elif from_diag and not to_diag:
                # diag -> full
                # Currently just falling back in the full -> full case
                # TODO: Implement this in a smarter and more memory efficient way
                return self.jTmjp(x, val, torch.diag_embed(matrix), wrt=wrt, from_diag=False, to_diag=False)
            elif not from_diag and to_diag:
                # full -> diag
                # Currently just falling back in the full -> full case
                # TODO: Implement this in a smarter and more memory efficient way
                return torch.diagonal(
                    self.jTmjp(x, val, matrix, wrt=wrt, from_diag=False, to_diag=False),
                    dim1=1,
                    dim2=2,
                )
            elif from_diag and to_diag:
                # diag -> diag
                weight = torch.ones(1, 1, int(self.scale_factor), int(self.scale_factor), device=x.device)

                matrix = F.conv2d(
                    matrix.reshape(b * c2, 1, h2, w2),
                    weight=weight,
                    bias=None,
                    stride=int(self.scale_factor),
                    padding=0,
                    dilation=1,
                    groups=1,
                ).reshape(b, c1 * h1 * w1)

                return matrix
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None
