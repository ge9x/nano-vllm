import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    # [Backend Analogy]: 非线性激活函数。可以理解为业务逻辑中的“非线性判断器 (if-else)”。
    # 让神经网络能够拟合复杂的非线性关系，而不仅仅是线性加法。
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
