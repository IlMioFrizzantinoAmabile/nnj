from typing import Literal, Union

import torch
from torch import nn, Tensor

from nnj.abstract_jacobian import AbstractJacobian


class L2Norm(AbstractJacobian, nn.Module):
    """L2 normalization layer"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._n_params = 0
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x.reshape(x.shape[0], -1), p=2, dim=1)
        normalized_x = torch.einsum("b,b...->b...", 1. / (norm + self.eps), x)
        return normalized_x

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
            if diag==True:
                raise NotImplementedError
            if val is None:
                val = self.forward(x)
            x = x.reshape(x.shape[0], -1)
            b, d = x.shape
            norm = torch.norm(x, p=2, dim=1)
            normalized_x = torch.einsum("b,bi->bi", 1. / (norm + self.eps), x)
            jacobian = torch.einsum("bi,bj->bij", normalized_x, normalized_x)
            jacobian = torch.diag(torch.ones(d, device=x.device)).expand(b, d, d) - jacobian
            jacobian = torch.einsum("b,bij->bij", 1. / (norm + self.eps), jacobian)
            return jacobian
        elif wrt == "weight":
            # non parametric layer has no jacobian with respect to weight
            return None