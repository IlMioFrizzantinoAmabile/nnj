from typing import Literal, Tuple, Union

import torch
from torch import nn, Tensor

class LogGaussian(nn.Module):
    def __init__(self, model, aggregate_batch=None): 
        super().__init__()
        self.model = model
        assert aggregate_batch in (None, "sum", "average")
        self.aggregate_batch = aggregate_batch

    @torch.no_grad()
    def forward(self, x: Tensor, target=None):
        b = x.shape[0]
        val = self.model(x)
        log_gaussian = - 0.5 * ((val - target) ** 2)
        log_gaussian = torch.sum(log_gaussian.reshape(b, -1), dim=1)
        if self.aggregate_batch == "sum":
            return torch.sum(log_gaussian)
        if self.aggregate_batch == "sum":
            return torch.avg(log_gaussian)
        return log_gaussian

    @torch.no_grad()
    def gradient(self, x: Tensor, target=None, wrt="weight"):
        b = x.shape[0]
        val = self.model(x)
        residual = - (val - target).reshape(b, -1)
        gradient = self.model.vjp(x, val, residual, wrt=wrt)
        if self.aggregate_batch == "sum":
            return torch.sum(gradient, dim=0)
        if self.aggregate_batch == "sum":
            return torch.avg(gradient, dim=0)
        return gradient
    
    @torch.no_grad()
    def ggn(self, x: Tensor, target=None, wrt="weight", to_diag=True, diag_backprop=False):
        b = x.shape[0]
        val = self.model(x)
        log_gaussian_hessian = - torch.ones_like(val).reshape(b, -1)
        JtHJ = self.model.jTmjp(x, val, log_gaussian_hessian, wrt=wrt, from_diag=True, to_diag=to_diag, diag_backprop=diag_backprop)
        if self.aggregate_batch == "sum":
            return torch.sum(JtHJ, dim=0)
        if self.aggregate_batch == "sum":
            return torch.avg(JtHJ, dim=0)
        return JtHJ
