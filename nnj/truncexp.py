from typing import Literal, Union

import torch
from torch import nn, Tensor
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from nnj.abstract_diagonal_jacobian import AbstractDiagonalJacobian


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


class TruncExp(AbstractDiagonalJacobian, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_params = 0

        self._trunc_exp = _trunc_exp.apply

    def forward(self, x):
        return self._trunc_exp(x)

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
            if val is None:
                val = self.forward(x)
            diag_jacobian = torch.exp(x.clamp(-15, 15)).reshape(val.shape[0], -1)
            if diag:
                return diag_jacobian
            else:
                return torch.diag_embed(diag_jacobian)
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None
