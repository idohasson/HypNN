import torch


@torch.jit.script
def projection(x: torch.Tensor):
    return torch.renorm(x, 2, 0, 1 - 9e-16)


@torch.jit.script
def mobius_add(x: torch.Tensor, y: torch.Tensor):
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2
    result = num / denom
    return result


@torch.jit.script
def mobius_mm(x: torch.Tensor, m: torch.Tensor):
    mx = x @ m
    x_norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(9e-16, 1 - 9e-16)
    mx_norm = torch.linalg.norm(mx, dim=-1, keepdim=True).clamp(9e-16, 19.0)
    result = torch.tanh(mx_norm / x_norm * torch.atanh(x_norm)) * mx / mx_norm
    return result


@torch.jit.script
def mobius_prod(x: torch.Tensor, m: torch.Tensor):
    EPS = 9e-16
    mx = m * x
    x_norm = torch.norm(x, dim=-1, keepdim=True, p=2).clamp(EPS, 1 - EPS)
    mx_norm = torch.norm(mx, dim=-1, keepdim=True, p=2).clamp(EPS, 19.0)
    result = torch.tanh(mx_norm / x_norm * torch.atanh(x_norm)) * mx / mx_norm
    return result


@torch.jit.script
def logmap0(x: torch.Tensor):
    x_norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(9e-16, 1 - 9e-16)
    return torch.atanh(x_norm) * x / x_norm


@torch.jit.script
def expmap0(x: torch.Tensor):
    norm_x = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(9e-16, 19)
    return torch.tanh(norm_x) * x / norm_x


@torch.jit.script
def hyp_distance(x: torch.Tensor, y: torch.Tensor):
    xx = torch.sum(x ** 2, dim=-1)
    yy = torch.sum(y ** 2, dim=-1)
    xy = torch.sum((x - y) ** 2, dim=-1)
    norm_xy = 1 + 2 * xy / ((1 - xx) * (1 - yy))
    return torch.log(norm_xy + torch.sqrt(norm_xy ** 2 - 1))
