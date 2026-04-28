"""本文件为面向无人机车辆小目标检测任务的二次开发模块。
其中基础检测框架依赖 Ultralytics YOLO，软著权利主张范围不包括第三方开源框架原始代码。
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Module):
    """轻量卷积单元，用于构造 Ghost 模块。"""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, groups: int = 1, act: bool = True):
        super().__init__()
        padding = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class GhostConv(nn.Module):
    """GhostConv 用于以更低的参数量生成更多特征图。"""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, act: bool = True):
        super().__init__()
        hidden = max(c2 // 2, 1)
        self.primary = ConvBNAct(c1, hidden, k, s, act=act)
        self.cheap = ConvBNAct(hidden, hidden, 5, 1, groups=hidden, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.primary(x)
        return torch.cat((y, self.cheap(y)), dim=1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck 适合替换较重的瓶颈块以降低计算量。"""

    def __init__(self, c1: int, c2: int, stride: int = 1):
        super().__init__()
        hidden = max(c2 // 2, 1)
        self.conv = nn.Sequential(
            GhostConv(c1, hidden, 1, 1),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False)
            if stride > 1
            else nn.Identity(),
            nn.BatchNorm2d(hidden) if stride > 1 else nn.Identity(),
            nn.SiLU(),
            GhostConv(hidden, c2, 1, 1, act=False),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(c1, c1, 3, stride=stride, padding=1, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
            )
            if stride > 1 or c1 != c2
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class C3Ghost(nn.Module):
    """C3Ghost 适合用于轻量化特征提取和场景化替换。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        hidden = max(int(c2 * 0.5), 1)
        self.cv1 = ConvBNAct(c1, hidden, 1, 1)
        self.cv2 = ConvBNAct(c1, hidden, 1, 1)
        self.blocks = nn.Sequential(*(GhostBottleneck(hidden, hidden) for _ in range(max(n, 1))))
        self.cv3 = ConvBNAct(hidden * 2, c2, 1, 1)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.blocks(self.cv1(x))
        y2 = self.cv2(x)
        out = self.cv3(torch.cat((y1, y2), dim=1))
        return out

