"""本文件为面向无人机车辆小目标检测任务的二次开发模块。
其中基础检测框架依赖 Ultralytics YOLO，软著权利主张范围不包括第三方开源框架原始代码。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


class ASFF_Concat(nn.Module):
    """自适应空间特征融合后再拼接。

    该模块保留多路特征图并输出拼接结果，同时通过轻量级空间权重抑制冗余分支，
    适合无人机视角中对小目标细节更敏感的特征融合场景。
    """

    def __init__(self, channels: Sequence[int] | int):
        super().__init__()
        if isinstance(channels, int):
            channels = [channels]
        self.channels = list(channels)
        self.attention = nn.Sequential(
            nn.Conv2d(sum(self.channels), len(self.channels), kernel_size=1, bias=False),
            nn.BatchNorm2d(len(self.channels)),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs: Iterable[torch.Tensor]) -> torch.Tensor:
        inputs = list(inputs)
        if len(inputs) != len(self.channels):
            raise ValueError(f"Expected {len(self.channels)} feature maps, got {len(inputs)}.")
        concat_x = torch.cat(inputs, dim=1)
        weights = self.attention(concat_x)
        weighted = [inputs[idx] * weights[:, idx : idx + 1] for idx in range(len(inputs))]
        return torch.cat(weighted, dim=1)

