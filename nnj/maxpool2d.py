from typing import Literal, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from nnj.abstract_jacobian import AbstractJacobian


class MaxPool2d(nn.MaxPool2d, AbstractJacobian):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_params = 0
        self.idx = None

    def forward(self, input: Tensor):
        val, idx = F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            return_indices=True,
        )
        self.idx = idx
        return val

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
            # identity = torch.ones(b, c2 * h2 * w2, device=x.device)
            # identity = torch.diag_embed(identity)
            # j = self.mjp(x, val, identity, wrt="input")

            identity = torch.ones(b, c1 * h1 * w1, device=x.device)
            identity = torch.diag_embed(identity)
            j = self.jmp(x, val, identity, wrt="input")
            # print(j.shape)
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
            if val is None:
                val = self.forward(x)
            b, c1, h1, w1 = x.shape
            _, c2, h2, w2 = val.shape
            matrix_orig_shape = matrix.shape
            assert matrix_orig_shape[0] == b
            assert matrix_orig_shape[1] == c1 * h1 * w1

            matrix = matrix.reshape(b * c1, h1 * w1, c1, h1, w1)
            arange_repeated = torch.repeat_interleave(torch.arange(b * c1), h2 * w2).long()
            idx = self.idx.reshape(-1)
            matrix = matrix[arange_repeated, idx, :, :, :].reshape(*val.shape, c1, h1, w1)
            matrix = matrix.reshape(b, c2 * h2 * w2, matrix_orig_shape[2])
            return matrix
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

            vector = vector.reshape(b * c2, h2 * w2)
            # indexes for batch and channel
            arange_repeated = torch.repeat_interleave(torch.arange(b * c2), h2 * w2).long()
            arange_repeated = arange_repeated.reshape(b * c2, h2 * w2)
            # indexes for col
            idx = self.idx.reshape(b * c2, h2 * w2)

            vector_J = torch.zeros((b * c2, h1 * w1), device=vector.device)
            vector_J[arange_repeated, idx] = vector
            vector_J = vector_J.reshape(b, c1 * h1 * w1)

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
            matrix = matrix.reshape(b * n_rows * c2, h2 * w2)

            # indexes for batch, channel and row
            arange_repeated = torch.repeat_interleave(torch.arange(b * n_rows * c2), h2 * w2).long()
            arange_repeated = arange_repeated.reshape(b * n_rows * c2, h2 * w2)
            # indexes for col
            idx = self.idx.reshape(b, c2, h2 * w2).unsqueeze(1).expand(-1, n_rows, -1, -1)
            idx = idx.reshape(b * n_rows * c2, h2 * w2)

            matrix_J = torch.zeros((b * n_rows * c1, h1 * w1), device=matrix.device)
            matrix_J[arange_repeated, idx] = matrix
            matrix_J = matrix_J.reshape(b, n_rows, c1 * h1 * w1)

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
                matrix = (
                    matrix.reshape(b, c1, h2 * w2, c1, h2 * w2).movedim(-2, -3).reshape(b * c1 * c1, h2 * w2, h2 * w2)
                )
                # indexes for batch and channel
                arange_repeated = torch.repeat_interleave(torch.arange(b * c1 * c1), h2 * w2 * h2 * w2).long()
                arange_repeated = arange_repeated.reshape(b * c1 * c1, h2 * w2, h2 * w2)
                # indexes for height and width
                idx = self.idx.reshape(b, c1, h2 * w2).unsqueeze(2).expand(-1, -1, h2 * w2, -1)
                idx_col = idx.unsqueeze(1).expand(-1, c1, -1, -1, -1).reshape(b * c1 * c1, h2 * w2, h2 * w2)
                idx_row = (
                    idx.unsqueeze(2).expand(-1, -1, c1, -1, -1).reshape(b * c1 * c1, h2 * w2, h2 * w2).movedim(-1, -2)
                )

                Jt_matrix_J = torch.zeros((b * c1 * c1, h1 * w1, h1 * w1), device=matrix.device)
                Jt_matrix_J[arange_repeated, idx_row, idx_col] = matrix
                Jt_matrix_J = (
                    Jt_matrix_J.reshape(b, c1, c1, h1 * w1, h1 * w1)
                    .movedim(-2, -3)
                    .reshape(b, c1 * h1 * w1, c1 * h1 * w1)
                )

                return Jt_matrix_J
            elif from_diag and not to_diag:
                # diag -> full
                raise NotImplementedError
            elif not from_diag and to_diag:
                # full -> diag
                raise NotImplementedError
            elif from_diag and to_diag:
                # diag -> diag

                # indexes for batch and channel
                arange_repeated = torch.repeat_interleave(torch.arange(b * c1), h2 * w2).long()
                arange_repeated = arange_repeated.reshape(b * c2, h2 * w2)
                # indexes for height and width
                idx = self.idx.reshape(b * c2, h2 * w2)

                Jt_matrix_J = torch.zeros_like(x)
                Jt_matrix_J = Jt_matrix_J.reshape(b * c1, h1 * w1)
                Jt_matrix_J[arange_repeated, idx] = matrix.reshape(b * c2, h2 * w2)

                return Jt_matrix_J.reshape(b, c1 * h1 * w1)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None
