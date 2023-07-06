import torch
import torch.nn as nn
from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, DiagGGNMC

from benchmarks.models import linear_model
from benchmarks.timer import Timer


class GGNExact():
    def __init__(self, model: nn.Sequential, lossfunc: nn.Module):

        self.lossfunc = extend(lossfunc)
        self.model = extend(model)

    def ggn(self, X: torch.tensor, y: torch.Tensor) -> torch.Tensor:

        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(DiagGGNExact()):
            loss.backward()
        dggn = torch.cat([p.diag_ggn_exact.data.flatten() for p in self.model.parameters()])

        return dggn


class GGNStochastic():
    def __init__(self, model: nn.Sequential, lossfunc: nn.Module):

        self.lossfunc = extend(lossfunc)
        self.model = extend(model)

    def ggn(self, X: torch.tensor, y: torch.Tensor) -> torch.Tensor:

        f = self.model(X)
        loss = self.lossfunc(f, y)
        with backpack(DiagGGNMC()):
            loss.backward()
        dggn = torch.cat([p.diag_ggn_mc.data.flatten() for p in model.parameters()])

        return dggn


if __name__ == "__main__":

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    X = torch.randn(10, 100, device="cuda")
    y = torch.randn(10, 100, device="cuda")
    N = 300

    model = linear_model.to("cuda")

    ggn_exact = GGNExact(model, nn.MSELoss())
    ggn_stochastic = GGNStochastic(model, nn.MSELoss())

    timer = Timer()
    mu, std = timer.time(ggn_exact, X, y, repetitions=N)
    print(f"DiagGGNExact: {mu:.2f} ± {std:.2f} ms")

    mu, std = timer.time(ggn_stochastic, X, y, repetitions=N)
    print(f"DiagGGNMC: {mu:.2f} ± {std:.2f} ms")
