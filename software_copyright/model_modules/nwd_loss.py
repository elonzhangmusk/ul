"""本文件为面向无人机车辆小目标检测任务的二次开发模块。
其中基础检测框架依赖 Ultralytics YOLO，软著权利主张范围不包括第三方开源框架原始代码。
"""

from __future__ import annotations

import torch


def _to_xyxy(boxes: torch.Tensor, xywh: bool) -> torch.Tensor:
    if xywh:
        x_c, y_c, w, h = boxes.unbind(dim=-1)
        half_w = w / 2.0
        half_h = h / 2.0
        return torch.stack((x_c - half_w, y_c - half_h, x_c + half_w, y_c + half_h), dim=-1)
    return boxes


def wasserstein_distance(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True) -> torch.Tensor:
    """计算两个边界框之间的 Wasserstein 距离近似形式。

    NWD 对小目标更友好的原因在于：当目标本身像素尺寸较小的时候，轻微的位移会导致 IoU 波动很大，
    而基于中心点与尺度差异的距离度量更平滑，训练时提供的回归信号更稳定。
    """
    box1 = _to_xyxy(box1, xywh)
    box2 = _to_xyxy(box2, xywh)
    box1 = box1.reshape(-1, 4)
    box2 = box2.reshape(-1, 4)
    center1 = (box1[:, :2] + box1[:, 2:]) / 2.0
    center2 = (box2[:, :2] + box2[:, 2:]) / 2.0
    size1 = box1[:, 2:] - box1[:, :2]
    size2 = box2[:, 2:] - box2[:, :2]
    center_term = ((center1 - center2) ** 2).sum(dim=-1)
    size_term = (((size1 - size2) / 2.0) ** 2).sum(dim=-1)
    return center_term + size_term


def normalized_wasserstein_score(distance: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """将 Wasserstein 距离转换为 [0, 1] 区间的相似度分数。"""
    distance = torch.as_tensor(distance, dtype=torch.float32)
    return torch.exp(-torch.sqrt(torch.clamp(distance, min=0.0)) / tau)


def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    box1 = _to_xyxy(box1, xywh=False).reshape(-1, 4)
    box2 = _to_xyxy(box2, xywh=False).reshape(-1, 4)
    inter_x1 = torch.maximum(box1[:, 0], box2[:, 0])
    inter_y1 = torch.maximum(box1[:, 1], box2[:, 1])
    inter_x2 = torch.minimum(box1[:, 2], box2[:, 2])
    inter_y2 = torch.minimum(box1[:, 3], box2[:, 3])
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h
    area1 = torch.clamp(box1[:, 2] - box1[:, 0], min=0.0) * torch.clamp(box1[:, 3] - box1[:, 1], min=0.0)
    area2 = torch.clamp(box2[:, 2] - box2[:, 0], min=0.0) * torch.clamp(box2[:, 3] - box2[:, 1], min=0.0)
    union = torch.clamp(area1 + area2 - inter_area, min=1e-7)
    return inter_area / union


def mixed_iou_nwd_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    iou_weight: float = 0.5,
    nwd_weight: float = 0.5,
    xywh: bool = False,
) -> torch.Tensor:
    """融合 IoU 和 NWD 的回归损失。

    该实现保留了小目标场景下的平滑距离监督，同时仍然兼顾 IoU 的区域重叠约束。
    """
    pred_boxes = torch.as_tensor(pred_boxes, dtype=torch.float32)
    target_boxes = torch.as_tensor(target_boxes, dtype=torch.float32)
    pred_boxes = _to_xyxy(pred_boxes, xywh)
    target_boxes = _to_xyxy(target_boxes, xywh)
    iou = _box_iou(pred_boxes, target_boxes)
    distance = wasserstein_distance(pred_boxes, target_boxes, xywh=False)
    nwd = normalized_wasserstein_score(distance)
    return iou_weight * (1.0 - iou).mean() + nwd_weight * (1.0 - nwd).mean()

