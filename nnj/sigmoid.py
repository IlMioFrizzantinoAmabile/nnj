from typing import Literal, Union

import torch
from torch import nn, Tensor

from nnj.abstract_diagonal_jacobian import AbstractDiagonalJacobian


class Sigmoid(AbstractDiagonalJacobian, nn.Sigmoid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_params = 0

    def jacobian(
        self,
        x: Tensor,
        val: Union[Tensor, None] = None,
        wrt: Literal["input", "weight"] = "input",
        diag: bool = False,
    ) -> Union[Tensor, None]:
        """Returns the Jacobian matrix"""
        if wrt == "input":
            if val is None:
                val = self.forward(x)
            diag_jacobian = (val * (1.0 - val)).reshape(val.shape[0], -1)
            if diag:
                return diag_jacobian
            else:
                return torch.diag_embed(diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None
