from typing import Literal, Optional, Tuple, Union

import torch
from torch import nn, Tensor

from nnj.abstract_jacobian import AbstractJacobian


class LogLikelihood(nn.Module):
    def __init__(
        self,
        aggregate_batch: Literal["return all", "sum", "average"] = "return all",
        metric_shape: Literal["full", "diag"] = "diag",
        metric_backprop: Literal["exact", "approx"] = "exact",
    ):
        super().__init__()

        assert aggregate_batch in (None, "sum", "average")
        assert metric_shape in ("full", "diag")
        assert metric_backprop in ("exact", "approx")

        self.aggregate_batch = aggregate_batch
        self.metric_shape = metric_shape
        self.metric_backprop = metric_backprop


class LogGaussian(LogLikelihood):
    @torch.no_grad()
    def forward(self, x: torch.Tensor, model: AbstractJacobian, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = x.shape[0]
        val = model(x)
        assert val.shape == target.shape

        log_gaussian = -0.5 * ((val - target) ** 2)
        log_gaussian = torch.sum(log_gaussian.reshape(b, -1), dim=1)

        if self.aggregate_batch == "return all":
            return log_gaussian
        elif self.aggregate_batch == "sum":
            return torch.sum(log_gaussian)
        elif self.aggregate_batch == "average":
            return torch.mean(log_gaussian)

    @torch.no_grad()
    def gradient(
        self,
        x: torch.Tensor,
        model: AbstractJacobian,
        target: Optional[torch.Tensor] = None,
        wrt: Literal["input", "weight"] = "weight",
    ) -> torch.Tensor:
        b = x.shape[0]
        val = model(x)
        assert val.shape == target.shape

        residual = (target - val).reshape(b, -1)
        gradient = model.vjp(x, val, residual, wrt=wrt)

        if self.aggregate_batch == "return all":
            return gradient
        elif self.aggregate_batch == "sum":
            return torch.sum(gradient, dim=0)
        elif self.aggregate_batch == "average":
            return torch.mean(gradient, dim=0)

    @torch.no_grad()
    def ggn(
        self,
        x: torch.Tensor,
        model: AbstractJacobian,
        target: Optional[torch.Tensor] = None,
        wrt: Literal["input", "weight"] = "weight",
    ) -> torch.Tensor:
        b = x.shape[0]
        val = model(x)

        log_gaussian_hessian = -torch.ones_like(val).reshape(b, -1)
        JtHJ = model.jTmjp(
            x,
            val,
            log_gaussian_hessian,
            wrt=wrt,
            from_diag=True,
            to_diag=self.metric_shape == "diag",
            diag_backprop=self.metric_backprop == "approx",
        )

        if self.aggregate_batch == "return all":
            return JtHJ
        elif self.aggregate_batch == "sum":
            return torch.sum(JtHJ, dim=0)
        elif self.aggregate_batch == "average":
            return torch.mean(JtHJ, dim=0)


class LogBinaryBernoulli(LogLikelihood):
    @torch.no_grad()
    def forward(self, x: torch.Tensor, model: AbstractJacobian, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = x.shape[0]
        val = model(x)
        assert val.shape == target.shape

        log_normalization = torch.logsumexp(torch.stack([val, torch.zeros_like(val)], dim=0), dim=0)
        log_prob_class0 = 0.0 - log_normalization
        log_prob_class1 = val - log_normalization
        log_bernoulli = target * log_prob_class1 + (1 - target) * log_prob_class0
        log_bernoulli = torch.sum(log_bernoulli.reshape(b, -1), dim=1)

        if self.aggregate_batch == "return all":
            return log_bernoulli
        elif self.aggregate_batch == "sum":
            return torch.sum(log_bernoulli)
        elif self.aggregate_batch == "average":
            return torch.mean(log_bernoulli)

    @torch.no_grad()
    def gradient(
        self,
        x: torch.Tensor,
        model: AbstractJacobian,
        target: Optional[torch.Tensor] = None,
        wrt: Literal["input", "weight"] = "weight",
    ) -> torch.Tensor:
        b = x.shape[0]
        val = model(x)
        assert val.shape == target.shape

        binary_bernoulli_prob = torch.exp(val) / (1.0 + torch.exp(val))
        residual = (target - binary_bernoulli_prob).reshape(b, -1)
        gradient = model.vjp(x, val, residual, wrt=wrt)

        if self.aggregate_batch == "return all":
            return gradient
        elif self.aggregate_batch == "sum":
            return torch.sum(gradient, dim=0)
        elif self.aggregate_batch == "average":
            return torch.mean(gradient, dim=0)

    @torch.no_grad()
    def ggn(
        self,
        x: torch.Tensor,
        model: AbstractJacobian,
        target: Optional[torch.Tensor] = None,
        wrt: Literal["input", "weight"] = "weight",
    ) -> torch.Tensor:
        b = x.shape[0]
        val = model(x)

        bernoulli_prob = torch.exp(val) / (1.0 + torch.exp(val))
        log_bernoulli_hessian = bernoulli_prob**2 - bernoulli_prob
        log_bernoulli_hessian = log_bernoulli_hessian.reshape(b, -1)

        JtHJ = model.jTmjp(
            x,
            val,
            log_bernoulli_hessian,
            wrt=wrt,
            from_diag=True,
            to_diag=self.metric_shape == "diag",
            diag_backprop=self.metric_backprop == "approx",
        )

        if self.aggregate_batch == "return all":
            return JtHJ
        elif self.aggregate_batch == "sum":
            return torch.sum(JtHJ, dim=0)
        elif self.aggregate_batch == "average":
            return torch.mean(JtHJ, dim=0)


class LogBernoulli(LogLikelihood):
    @torch.no_grad()
    def forward(self, x: torch.Tensor, model: AbstractJacobian, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = x.shape[0]
        val = model(x)
        assert val.shape == target.shape

        log_normalization = torch.logsumexp(val, dim=-1).unsqueeze(-1).expand(val.shape)
        log_prob_classes = val - log_normalization
        log_bernoulli = target * log_prob_classes
        log_bernoulli = torch.sum(log_bernoulli.reshape(b, -1), dim=1)

        if self.aggregate_batch == "return all":
            return log_bernoulli
        elif self.aggregate_batch == "sum":
            return torch.sum(log_bernoulli)
        elif self.aggregate_batch == "average":
            return torch.mean(log_bernoulli)

    @torch.no_grad()
    def gradient(
        self,
        x: torch.Tensor,
        model: AbstractJacobian,
        target: Optional[torch.Tensor] = None,
        wrt: Literal["input", "weight"] = "weight",
    ) -> torch.Tensor:
        b = x.shape[0]
        val = model(x)
        assert val.shape == target.shape

        exp_val = torch.exp(val)
        normalization = torch.sum(exp_val, dim=-1).unsqueeze(-1).expand(val.shape)
        bernoulli_prob = exp_val / normalization

        residual = (target - bernoulli_prob).reshape(b, -1)
        gradient = model.vjp(x, val, residual, wrt=wrt)

        if self.aggregate_batch == "return all":
            return gradient
        elif self.aggregate_batch == "sum":
            return torch.sum(gradient, dim=0)
        elif self.aggregate_batch == "average":
            return torch.mean(gradient, dim=0)

    @torch.no_grad()
    def ggn(
        self,
        x: torch.Tensor,
        model: AbstractJacobian,
        target: Optional[torch.Tensor] = None,
        wrt: Literal["input", "weight"] = "weight",
    ) -> torch.Tensor:
        b = x.shape[0]
        val = model(x)
        c = val.shape[-1]
        l = val[0].numel() / c

        exp_val = torch.exp(val)
        normalization = torch.sum(exp_val, dim=-1).unsqueeze(-1).expand(val.shape)
        bernoulli_prob = exp_val / normalization
        diag = torch.diag_embed(bernoulli_prob)
        outer = torch.einsum("...i,...j->...ij", bernoulli_prob, bernoulli_prob)
        log_bernoulli_hessian = outer - diag
        if l == 1:
            log_bernoulli_hessian = log_bernoulli_hessian.reshape(b, c, c)
        else:
            log_bernoulli_hessian = (
                log_bernoulli_hessian.movedim((-2, -1), (1, 2))
                .reshape(b, c, c, l)
                .diag_embed()
                .movedim((-2, -1), (1, 3))
                .reshape(b, l * c, l * c)
            )
        JtHJ = model.jTmjp(
            x,
            val,
            log_bernoulli_hessian,
            wrt=wrt,
            from_diag=False,
            to_diag=self.metric_shape == "diag",
            diag_backprop=self.metric_backprop == "approx",
        )

        if self.aggregate_batch == "return all":
            return JtHJ
        elif self.aggregate_batch == "sum":
            return torch.sum(JtHJ, dim=0)
        elif self.aggregate_batch == "average":
            return torch.mean(JtHJ, dim=0)
