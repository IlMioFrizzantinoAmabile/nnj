import torch

import nnj
from benchmarks.models import linear_model
from benchmarks.timer import Timer


class GGN:
    def __init__(self, model: nnj.Sequential):
        self.model = nnj.utils.convert_to_nnj(model)

    def ggn(self, X: torch.tensor, y: torch.Tensor) -> torch.Tensor:
        # backpropagate through the network
        with torch.no_grad():
            Jt_J = self.model.jTmjp(
                X,
                None,
                None,
                wrt="weight",  # computes the jacobian wrt weights or inputs
                to_diag=True,  # computes the diagonal elements only
                from_diag=False,
                diag_backprop=False,  # approximates the diagonal elements of the Hessian
            )

            # average along batch size
            Jt_J = torch.mean(Jt_J, dim=0)

        return Jt_J


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    X = torch.randn(10, 100, device="cuda:0")
    y = torch.randn(10, 100, device="cuda:0")
    N = 300

    model = linear_model
    ggn = GGN(model)
    ggn.model = ggn.model.to("cuda:0")

    timer = Timer()
    mu, std = timer.time(ggn.ggn, X, y, repetitions=N)
    print(f"NNJ: {mu:.2f} ± {std:.2f} ms")
