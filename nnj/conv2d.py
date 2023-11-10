from typing import Literal, Tuple, Union

import torch
import torch.nn.functional as F
from functorch import vmap
from torch import nn, Tensor

from nnj.abstract_jacobian import AbstractJacobian


def compute_reversed_padding(padding, kernel_size=1):
    return kernel_size - 1 - padding


class Conv2d(nn.Conv2d, AbstractJacobian):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        use_vmap_for_backprop=True,
    ):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )

        dw_padding_h = compute_reversed_padding(self.padding[0], kernel_size=self.kernel_size[0])
        dw_padding_w = compute_reversed_padding(self.padding[1], kernel_size=self.kernel_size[1])
        self.dw_padding = (dw_padding_h, dw_padding_w)
        self._n_params = self.weight.numel() if self.bias is None else self.weight.numel() + out_channels
        self.use_vmap = use_vmap_for_backprop
        if self.stride != (1, 1):
            raise ValueError(
                f"I can't handle stride = {self.stride}, sorry. Only stride 1 is supported (until pytorch extends the conv class)"
            )

    @torch.no_grad()
    def jacobian(
        self,
        x: Tensor,
        val: Union[Tensor, None] = None,
        wrt: Literal["input", "weight"] = "input",
    ) -> Tensor:
        """Returns the Jacobian matrix"""
        if val is None:
            val = self.forward(x)
        b, c2, h2, w2 = val.shape
        identity = torch.ones(b, c2 * h2 * w2, device=x.device)
        identity = torch.diag_embed(identity)
        j = self.mjp(x, val, identity, wrt=wrt)
        return j

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
            raise NotImplementedError
        elif wrt == "weight":
            if self.bias is None:
                raise NotImplementedError
            else:
                raise NotImplementedError

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
            raise NotImplementedError
        elif wrt == "weight":
            if self.bias is None:
                raise NotImplementedError
            else:
                raise NotImplementedError

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
            if matrix is None:
                matrix = torch.ones((x.shape[0], self._n_params), dtype=x.dtype, device=x.device)
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
        if val is None:
            val = self.forward(x)
        b, c1, h1, w1 = x.shape
        _, c2, h2, w2 = val.shape
        assert list(vector.shape) == [b, c2 * h2 * w2]
        if wrt == "input":
            # expand vector as a cube [(output channel)x(output height)x(output width)]
            vector = vector.reshape(b, c2, h2, w2)
            # convolve
            JT_vectorT = F.conv_transpose2d(
                vector,
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            ).reshape(b, c1 * h1 * w1)
            return JT_vectorT
        elif wrt == "weight":
            kernel_h, kernel_w = self.kernel_size

            if self.bias is not None:
                b_term = torch.einsum("bchw->bc", vector.reshape(b, c2, h2, w2))

            # expand vector as a cube [(output channel)x(output height)x(output width)]
            vector = vector.reshape(b, c2, h2, w2)
            # transpose the images in (output height)x(output width)
            vector = torch.flip(vector, [-2, -1])

            if self.use_vmap:
                reversed_inputs = torch.flip(x, [-2, -1])

                def single_batch_fun(input_single_batch, vector_single_batch):
                    reversed_input_single_batch = input_single_batch.unsqueeze(0).movedim(0, 1)
                    vector_single_batch = vector_single_batch.unsqueeze(0).movedim(0, 1)
                    # convolve each column
                    vector_J_single_batch = F.conv2d(
                        vector_single_batch.reshape(-1, 1, h2, w2),
                        weight=reversed_input_single_batch,
                        bias=None,
                        stride=self.stride,
                        padding=self.dw_padding,
                        dilation=self.dilation,
                        groups=self.groups,
                    ).reshape(c2 * c1 * kernel_h * kernel_w)
                    return vector_J_single_batch

                vector_J = vmap(single_batch_fun)(reversed_inputs, vector)
            else:
                # switch batch size and output channel
                vector = vector.movedim(0, 1)

                vector_J = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, device=x.device)
                for i in range(b):
                    # set the weight to the convolution
                    input_single_batch = x[i : i + 1, :, :, :]
                    reversed_input_single_batch = torch.flip(input_single_batch, [-2, -1]).movedim(0, 1)
                    vector_single_batch = vector[:, i : i + 1, :, :]

                    # convolve each column
                    vector_J_single_batch = F.conv2d(
                        vector_single_batch.reshape(-1, 1, h2, w2),
                        weight=reversed_input_single_batch,
                        bias=None,
                        stride=self.stride,
                        padding=self.dw_padding,
                        dilation=self.dilation,
                        groups=self.groups,
                    ).reshape(c2 * c1 * kernel_h * kernel_w)

                    # reshape as a (num of weights)x(num of column) matrix
                    vector_J[i, :] = vector_J_single_batch

            if self.bias is None:
                return vector_J
            else:
                return torch.cat([vector_J, b_term], dim=1)

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
        if val is None:
            val = self.forward(x)
        if matrix is None:
            return self.jacobian(x, val, wrt=wrt)
        b, c1, h1, w1 = x.shape
        _, c2, h2, w2 = val.shape
        num_of_rows = matrix.shape[1]
        assert list(matrix.shape) == [b, num_of_rows, c2 * h2 * w2]
        if wrt == "input":
            # expand rows as cubes [(output channel)x(output height)x(output width)]
            matrix_rows = matrix.movedim(-1, -2).reshape(b, c2, h2, w2, num_of_rows)
            # see rows as columns of the transposed matrix
            matrixT_cols = matrix_rows
            # convolve each column
            JT_matrixT_cols = (
                F.conv_transpose2d(
                    matrixT_cols.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c2, h2, w2),
                    weight=self.weight,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    output_padding=self.output_padding,
                )
                .reshape(b, *matrixT_cols.shape[4:], c1, h1, w1)
                .movedim((-3, -2, -1), (1, 2, 3))
            )
            # reshape as a (num of input)x(num of output) matrix, one for each batch size
            JT_matrixT_cols = JT_matrixT_cols.reshape(b, c1 * h1 * w1, num_of_rows)
            # transpose
            matrix_J = JT_matrixT_cols.movedim(1, 2)
            return matrix_J

        elif wrt == "weight":
            kernel_h, kernel_w = self.kernel_size

            # expand rows as cubes [(output channel)x(output height)x(output width)]
            matrix_rows = matrix.movedim(-1, -2).reshape(b, c2, h2, w2, num_of_rows)
            # see rows as columns of the transposed matrix
            matrixT_cols = matrix_rows
            # transpose the images in (output height)x(output width)
            matrixT_cols = torch.flip(matrixT_cols, [-3, -2])

            if self.use_vmap:
                reversed_inputs = torch.flip(x, [-2, -1])

                def single_batch_fun(input_single_batch, matrixT_single_batch):
                    reversed_input_single_batch = input_single_batch.unsqueeze(0).movedim(0, 1)
                    matrix_single_batch = matrixT_single_batch.unsqueeze(0).movedim(0, 1)
                    # convolve each column
                    matrix_J_single_batch = (
                        F.conv2d(
                            matrix_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                            weight=reversed_input_single_batch,
                            bias=None,
                            stride=self.stride,
                            padding=self.dw_padding,
                            dilation=self.dilation,
                            groups=self.groups,
                        )
                        .reshape(c2, *matrix_single_batch.shape[4:], c1, kernel_h, kernel_w)
                        .movedim((-3, -2, -1), (1, 2, 3))
                    )
                    # reshape as a (num of weights)x(num of column) matrix
                    matrix_J_single_batch = matrix_J_single_batch.reshape(c2 * c1 * kernel_h * kernel_w, num_of_rows)
                    return matrix_J_single_batch

                matrix_J = vmap(single_batch_fun)(reversed_inputs, matrixT_cols)
            else:
                # switch batch size and output channel
                matrixT_cols = matrixT_cols.movedim(0, 1)

                matrix_J = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, num_of_rows, device=x.device)
                for i in range(b):
                    # set the weight to the convolution
                    input_single_batch = x[i : i + 1, :, :, :]
                    reversed_input_single_batch = torch.flip(input_single_batch, [-2, -1]).movedim(0, 1)
                    matrix_single_batch = matrixT_cols[:, i : i + 1, :, :, :]

                    # convolve each column
                    matrix_J_single_batch = (
                        F.conv2d(
                            matrix_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                            weight=reversed_input_single_batch,
                            bias=None,
                            stride=self.stride,
                            padding=self.dw_padding,
                            dilation=self.dilation,
                            groups=self.groups,
                        )
                        .reshape(c2, *matrix_single_batch.shape[4:], c1, kernel_h, kernel_w)
                        .movedim((-3, -2, -1), (1, 2, 3))
                    )

                    # reshape as a (num of weights)x(num of column) matrix
                    matrix_J_single_batch = matrix_J_single_batch.reshape(c2 * c1 * kernel_h * kernel_w, num_of_rows)
                    matrix_J[i, :, :] = matrix_J_single_batch

            # transpose
            matrix_J = matrix_J.movedim(-1, -2)

            if self.bias is None:
                return matrix_J
            else:
                b_term = torch.einsum("bvchw->bvc", matrix.reshape(b, -1, c2, h2, w2))
                return torch.cat([matrix_J, b_term], dim=2)

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
                matrix = self.mjp(x, val, matrix, wrt="input")
                matrix = matrix.movedim(-2, -1)
                matrix = self.mjp(x, val, matrix, wrt="input")
                matrix = matrix.movedim(-2, -1)
                return matrix
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
                b, c1, h1, w1 = x.shape
                _, c2, h2, w2 = val.shape

                matrix = matrix.reshape(b, c2, h2, w2)

                Jt_matrix_J = (
                    F.conv_transpose2d(
                        matrix.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c2, h2, w2),
                        weight=self.weight**2,
                        bias=None,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups,
                        output_padding=0,
                    )
                    .reshape(b, *matrix.shape[4:], c1, h1, w1)
                    .movedim((-3, -2, -1), (1, 2, 3))
                ).reshape(b, c1 * h1 * w1)

                return Jt_matrix_J
        elif wrt == "weight":
            if not from_diag and not to_diag:
                # full -> full
                b, c1, h1, w1 = x.shape
                _, c2, h2, w2 = val.shape
                kernel_h, kernel_w = self.kernel_size
                assert list(matrix.shape) == [b, c2 * h2 * w2, c2 * h2 * w2]
                num_of_rows = c2 * h2 * w2

                if self.use_vmap:

                    def single_batch_fun(input_single_batch, matrix_single_batch):
                        # set the weight to the convolution
                        reversed_input_single_batch = input_single_batch.unsqueeze(0).flip([-2, -1]).movedim(0, 1)
                        # reshape, transpose and reverse the matrix
                        matrix_single_batch = (
                            matrix_single_batch.unsqueeze(0)
                            .movedim(-1, -2)
                            .reshape(1, c2, h2, w2, num_of_rows)
                            .flip([-3, -2])
                            .movedim(0, 1)
                        )
                        # convolve each column
                        matrix_J_single_batch = (
                            F.conv2d(
                                matrix_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                                weight=reversed_input_single_batch,
                                bias=None,
                                stride=self.stride,
                                padding=self.dw_padding,
                                dilation=self.dilation,
                                groups=self.groups,
                            )
                            .reshape(c2, num_of_rows, c1, kernel_h, kernel_w)
                            .movedim((-3, -2, -1), (1, 2, 3))
                        )
                        # reshape as a (num of weights)x(num of column) matrix
                        matrix_J_single_batch = matrix_J_single_batch.reshape(
                            c2 * c1 * kernel_h * kernel_w, num_of_rows
                        )
                        if self.bias is not None:
                            b_term = torch.einsum("chwv->cv", matrix_single_batch.reshape(c2, h2, w2, -1))
                            matrix_J_single_batch = torch.cat([matrix_J_single_batch, b_term], dim=0)

                        matrix_J_single_batch = (
                            matrix_J_single_batch.movedim(-1, -2)
                            .reshape(1, c2, h2, w2, self._n_params)
                            .flip([-3, -2])
                            .movedim(0, 1)
                        )
                        # convolve each column
                        Jt_matrix_J_single_batch = (
                            F.conv2d(
                                matrix_J_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                                weight=reversed_input_single_batch,
                                bias=None,
                                stride=self.stride,
                                padding=self.dw_padding,
                                dilation=self.dilation,
                                groups=self.groups,
                            )
                            .reshape(c2, self._n_params, c1, kernel_h, kernel_w)
                            .movedim((-3, -2, -1), (1, 2, 3))
                        )
                        Jt_matrix_J_single_batch = Jt_matrix_J_single_batch.reshape(
                            c2 * c1 * kernel_h * kernel_w, self._n_params
                        )
                        if self.bias is not None:
                            b_term = torch.einsum("chwv->cv", matrix_J_single_batch.reshape(c2, h2, w2, -1))
                            Jt_matrix_J_single_batch = torch.cat([Jt_matrix_J_single_batch, b_term], dim=0)

                        return Jt_matrix_J_single_batch

                    Jt_matrix_J = vmap(single_batch_fun)(x, matrix)

                else:
                    Jt_matrix_J = torch.zeros(b, self._n_params, self._n_params, device=x.device)
                    for i in range(b):
                        # set the weight to the convolution
                        input_single_batch = x[i : i + 1, :, :, :]
                        reversed_input_single_batch = torch.flip(input_single_batch, [-2, -1]).movedim(0, 1)

                        matrix_single_batch = matrix[i : i + 1, :, :]
                        matrix_single_batch = (
                            matrix_single_batch.movedim(-1, -2)
                            .reshape(1, c2, h2, w2, num_of_rows)
                            .flip([-3, -2])
                            .movedim(0, 1)
                        )
                        # convolve each column
                        matrix_J_single_batch = (
                            F.conv2d(
                                matrix_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                                weight=reversed_input_single_batch,
                                bias=None,
                                stride=self.stride,
                                padding=self.dw_padding,
                                dilation=self.dilation,
                                groups=self.groups,
                            )
                            .reshape(c2, num_of_rows, c1, kernel_h, kernel_w)
                            .movedim((-3, -2, -1), (1, 2, 3))
                        )
                        # reshape as a (num of weights)x(num of column) matrix
                        matrix_J_single_batch = matrix_J_single_batch.reshape(
                            c2 * c1 * kernel_h * kernel_w, num_of_rows
                        )
                        if self.bias is not None:
                            b_term = torch.einsum("chwv->cv", matrix_single_batch.reshape(c2, h2, w2, -1))
                            matrix_J_single_batch = torch.cat([matrix_J_single_batch, b_term], dim=0)

                        matrix_J_single_batch = (
                            matrix_J_single_batch.movedim(-1, -2)
                            .reshape(1, c2, h2, w2, self._n_params)
                            .flip([-3, -2])
                            .movedim(0, 1)
                        )
                        # convolve each column
                        Jt_matrix_J_single_batch = (
                            F.conv2d(
                                matrix_J_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                                weight=reversed_input_single_batch,
                                bias=None,
                                stride=self.stride,
                                padding=self.dw_padding,
                                dilation=self.dilation,
                                groups=self.groups,
                            )
                            .reshape(c2, self._n_params, c1, kernel_h, kernel_w)
                            .movedim((-3, -2, -1), (1, 2, 3))
                        )
                        Jt_matrix_J_single_batch = Jt_matrix_J_single_batch.reshape(
                            c2 * c1 * kernel_h * kernel_w, self._n_params
                        )
                        if self.bias is not None:
                            b_term = torch.einsum("chwv->cv", matrix_J_single_batch.reshape(c2, h2, w2, -1))
                            Jt_matrix_J_single_batch = torch.cat([Jt_matrix_J_single_batch, b_term], dim=0)

                        Jt_matrix_J[i, :, :] = Jt_matrix_J_single_batch
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
                b, c1, h1, w1 = x.shape
                _, c2, h2, w2 = val.shape
                kernel_h, kernel_w = self.kernel_size

                if self.bias is not None:
                    matrix = matrix.reshape(b, c2, h2 * w2)
                    bias_term = torch.sum(matrix, 2)

                matrix = matrix.reshape(b, c2, h2, w2)
                # transpose the images in (output height)x(output width)
                matrix = torch.flip(matrix, [-3, -2, -1])
                flip_squared_input = torch.flip(x, [-3, -2, -1]) ** 2

                if self.use_vmap:

                    def single_batch_fun(input_single_batch, matrix_single_batch):
                        weigth_sq = input_single_batch.unsqueeze(0).movedim(0, 1)
                        matrix_single_batch = matrix_single_batch.unsqueeze(0).movedim(0, 1)
                        Jt_matrix_J_single_batch = (
                            F.conv2d(
                                matrix_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                                weight=weigth_sq,
                                bias=None,
                                stride=self.stride,
                                padding=self.dw_padding,
                                dilation=self.dilation,
                                groups=self.groups,
                            )
                            .reshape(c2, *matrix_single_batch.shape[4:], c1, kernel_h, kernel_w)
                            .movedim((-3, -2, -1), (1, 2, 3))
                        )

                        Jt_matrix_J_single_batch = torch.flip(Jt_matrix_J_single_batch, [-4, -3])
                        # reshape as a (num of weights)x(num of column) matrix
                        Jt_matrix_J_single_batch = Jt_matrix_J_single_batch.reshape(c2 * c1 * kernel_h * kernel_w)
                        return Jt_matrix_J_single_batch

                    Jt_matrix_J = vmap(single_batch_fun)(flip_squared_input, matrix)
                else:
                    # switch batch size and output channel
                    matrix = matrix.movedim(0, 1)
                    Jt_matrix_J = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, device=x.device)
                    flip_squared_input = flip_squared_input.movedim(0, 1)

                    for i in range(b):
                        # set the weight to the convolution
                        weigth_sq = flip_squared_input[:, i : i + 1, :, :]
                        matrix_single_batch = matrix[:, i : i + 1, :, :]

                        Jt_matrix_J_single_batch = (
                            F.conv2d(
                                matrix_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                                weight=weigth_sq,
                                bias=None,
                                stride=self.stride,
                                padding=self.dw_padding,
                                dilation=self.dilation,
                                groups=self.groups,
                            )
                            .reshape(c2, *matrix_single_batch.shape[4:], c1, kernel_h, kernel_w)
                            .movedim((-3, -2, -1), (1, 2, 3))
                        )

                        Jt_matrix_J_single_batch = torch.flip(Jt_matrix_J_single_batch, [-4, -3])
                        # reshape as a (num of weights)x(num of column) matrix
                        Jt_matrix_J_single_batch = Jt_matrix_J_single_batch.reshape(c2 * c1 * kernel_h * kernel_w)
                        Jt_matrix_J[i, :] = Jt_matrix_J_single_batch

                if self.bias is not None:
                    Jt_matrix_J = torch.cat([Jt_matrix_J, bias_term], dim=1)

                return Jt_matrix_J
