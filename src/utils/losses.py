import torch
import torch.nn as nn
import torch.nn.functional as F


class BellLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_p = torch.pow((y - p), 2)
        y_p_div = -1.0 * torch.div(y_p, 162.0)
        exp_y_p = torch.exp(y_p_div)
        loss = 300 * (1.0 - exp_y_p)
        return torch.mean(loss)


class LogCosh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.log(torch.cosh(p - y))
        return torch.mean(loss)


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(p, y))


class GL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.sigma = sigma

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gl = self.eps / (self.lam ** 2) * (1 - torch.exp(-1 * ((y - p) ** 2) / (self.sigma ** 2)))
        return gl.mean()


class RMBell(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y)


class RMLCosh(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.logcosh(p, y)


class RMGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.rmse = RMSE()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.gl(p, y)


class RMBellLCosh(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y) + self.logcosh(p, y)


class RMBellGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y) + self.gl(p, y)


class BellLCosh(nn.Module):
    def __init__(self):
        super().__init__()
        self.bell = BellLoss()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.logcosh(p, y)


class BellGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.bell = BellLoss()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.gl(p, y)


class BellLCoshGL(nn.Module):
    def __init__(self):
        super().__init__()
        self.bell = BellLoss()
        self.logcosh = LogCosh()
        self.gl = GL()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.logcosh(p, y) + self.gl(p, y)


class LogCoshGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.logcosh = LogCosh()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.logcosh(p, y) + self.gl(p, y)


class MAELoss(nn.Module):
    """Mean Absolute Error loss"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(x - y))


class MSELoss(nn.Module):
    """Mean Squared Error loss"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.pow(x - y, 2))


class CCCLoss(nn.Module):
    """Lin's Concordance Correlation Coefficient."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps
        )
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1 - ccc


