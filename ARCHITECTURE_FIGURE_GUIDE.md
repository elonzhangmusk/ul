# 模型架构图撰写说明

本文档用于帮助你绘制论文中的模型架构图，内容基于原始 YOLO12 结构和你的自定义轻量化版本 `yolo12_yange.yaml`。目标是把“原始骨架是什么、你改了哪里、图里该如何突出核心贡献”讲清楚。

## 1. 论文里这张架构图应该表达什么？

如果你的论文主线聚焦在 `yange4` 这版模型，那么这张图最核心要表达的不是整个 YOLO12 的所有实现细节，而是下面这句话：

> 在保持 YOLO12 原始多尺度检测框架不变的前提下，本文仅对 backbone 最深层的重型特征提取模块进行轻量化替换，将原始 `A2C2f` 替换为 `C3Ghost`，从而实现更低的参数量、更小的计算量和更快的推理速度。

所以这张图不是“把所有层都画满”，而是要突出：

- 原始 YOLO12 的基本拓扑
- 你的模型沿用了哪些部分
- 你的核心改动发生在哪个位置
- 改动后信息是如何流向检测头的

## 2. 你的模型可以分成哪四部分来画？

建议你整张图按四个区域来画：

1. Input
2. Backbone
3. Neck
4. Detect Head

如果要画得更像论文图，我建议采用 **左到右** 或 **上到下** 的方式，保持三个尺度特征流清楚可见。

## 3. 原始 YOLO12 的核心骨架怎么概括？

根据 `ultralytics/cfg/models/12/yolo12.yaml`，原始 YOLO12n 可以概括为：

### 3.1 Backbone

- `Conv` 下采样到 P1/2
- `Conv` 下采样到 P2/4
- `C3k2` 做浅层特征提取
- `Conv` 下采样到 P3/8
- `C3k2` 做中浅层特征提取
- `Conv` 下采样到 P4/16
- `A2C2f` 做高级语义建模
- `Conv` 下采样到 P5/32
- `A2C2f` 做最深层特征提取

### 3.2 Neck

- 上采样 + `Concat` 与 P4 融合
- `A2C2f` 融合特征
- 再上采样 + `Concat` 与 P3 融合
- `A2C2f` 输出小目标分支特征
- 再通过下采样回流到 P4
- 再通过下采样回流到 P5

### 3.3 Head

- 三个尺度检测头：`P3`、`P4`、`P5`
- 最终由 `Detect` 模块输出分类和边框回归结果

## 4. 你的 yange 架构和原始 YOLO12 的区别是什么？

根据 `ultralytics/cfg/models/12/yolo12_yange.yaml`，你的版本基本保留了原始 YOLO12n 的主体结构，唯一需要在架构图上高亮的改动是：

### 4.1 核心改动点

原始 YOLO12：

```yaml
- [-1, 4, A2C2f, [1024, True, 1]]
```

你的版本：

```yaml
- [-1, 4, C3Ghost, [1024, True]]
```

也就是说，**你仅将 backbone 最深层 P5/32 的重型 `A2C2f` 模块替换为轻量化 `C3Ghost` 模块**。

### 4.2 这个改动为什么是“核心改动”？

因为它同时满足三点：

- 改动很小，不破坏原始多尺度检测拓扑
- 改动位置足够关键，位于最深层高语义特征提取位置
- 改动目标非常明确，就是做轻量化

这意味着你的架构图里最应该被框出来、标红或加虚线注释的，就是这个模块替换位置。

## 5. 图里每个模块建议怎么命名？

为了让图既专业又不显得太乱，建议你统一使用下面这些名称：

### 5.1 输入层

- `Input Image`
- 可在旁边标注：`1024 x 1024`

### 5.2 Backbone

- `Conv`
- `C3k2 Block`
- `A2C2f Block`
- `C3Ghost Block`

### 5.3 Neck

- `Upsample`
- `Concat`
- `A2C2f Fusion Block`
- `Conv Downsample`

### 5.4 检测头

- `P3 Detection Head`
- `P4 Detection Head`
- `P5 Detection Head`
- `Detect`

### 5.5 你的改动标注

你可以专门加一个红框或注释框：

- `Lightweight Replacement`
- `A2C2f -> C3Ghost`
- `Only modification in the backbone`

## 6. 论文图建议怎么布局？

我建议你画成下面这种层次关系。

```text
Input
  -> Conv (P1/2)
  -> Conv (P2/4)
  -> C3k2
  -> Conv (P3/8)
  -> C3k2
  -> Conv (P4/16)
  -> A2C2f
  -> Conv (P5/32)
  -> [C3Ghost]  <-- 核心改动，用高亮框起来

P5 -> Upsample -> Concat(P4) -> A2C2f
        -> Upsample -> Concat(P3) -> A2C2f -> P3 Head
        -> Downsample -> Concat -> A2C2f -> P4 Head
        -> Downsample -> Concat -> C3k2 -> P5 Head

P3/P4/P5 -> Detect
```

如果你画图软件支持颜色，我建议：

- 原始 YOLO12 公共模块用灰色或蓝色
- 你的改动模块 `C3Ghost` 用红色或橙色
- 三个输出头用绿色或紫色区分

## 7. 图注该怎么写？

你可以直接用下面这个版本作为论文图注基础：

> Figure X. Overall architecture of the proposed lightweight YOLO12-based detector. The model preserves the original three-scale detection pipeline of YOLO12, including the P3, P4, and P5 prediction branches. The key modification is a lightweight replacement in the deepest backbone stage, where the original A2C2f block is replaced with a C3Ghost block to reduce parameters and computational cost while maintaining the multi-scale detection framework.

如果你要中文图注，也可以写成：

> 图X 本文提出的轻量化 YOLO12 检测框架。该模型保持了原始 YOLO12 的三尺度检测结构，包括 P3、P4 和 P5 三个检测分支。与原始模型相比，本文仅在 backbone 最深层将重型 A2C2f 模块替换为轻量化 C3Ghost 模块，从而在保持整体检测框架不变的前提下降低模型参数量和计算复杂度。

## 8. 正文里“模型架构”这一段怎么写？

你可以直接把方法部分写成下面这种风格：

### 中文版

本文模型整体沿用 YOLO12 的多尺度检测框架，由 Backbone、Neck 和 Detect Head 三部分组成。Backbone 负责逐步提取从浅层纹理信息到深层语义信息的多级特征，Neck 通过上采样与横向连接实现多尺度特征融合，最终由 Detect Head 在 P3、P4 和 P5 三个尺度上完成目标分类与边框回归。

与原始 YOLO12 不同，本文并未对整体检测拓扑进行大规模重构，而是采用一种更克制的轻量化策略：仅在 Backbone 最深层的高语义特征提取阶段，将原始 A2C2f 模块替换为 C3Ghost 模块。该设计的出发点在于，浅层与中层特征对小目标检测更为敏感，因此尽量保持原始结构不变；而最深层特征提取模块计算开销较大，更适合作为轻量化改造的位置。通过这种单点替换，模型在保持原始三尺度检测框架的同时，有效降低了参数量与计算量，并提升了推理速度。

### 英文版

The proposed model follows the original YOLO12 multi-scale detection paradigm, consisting of a backbone, a neck, and a detection head. The backbone progressively extracts hierarchical features from shallow texture information to deep semantic representations. The neck performs multi-scale feature fusion through upsampling and lateral concatenation, and the detection head produces object classification and bounding box regression outputs on three prediction scales, namely P3, P4, and P5.

Unlike the original YOLO12, the proposed method does not redesign the entire detection topology. Instead, a conservative lightweight strategy is adopted by replacing only the deepest A2C2f block in the backbone with a C3Ghost block. This design is motivated by the observation that shallow and intermediate features are more critical for preserving small-object representations, while the deepest semantic stage contributes more heavily to computational overhead. Therefore, the proposed modification reduces model complexity and inference cost while retaining the original three-scale detection framework.

## 9. 如果你要单独画“改进模块局部放大图”，应该怎么画？

如果论文版面允许，我建议你除了整体架构图，再加一个右侧小图，专门放大这个替换模块：

### 小图标题建议

- `Detailed structure of the lightweight replacement block`
- `Replacement of A2C2f with C3Ghost in the deepest backbone stage`

### 小图内容建议

左边画原模块：

- `A2C2f`
- 标注：`heavy semantic extraction`

右边画改进模块：

- `C3Ghost`
- 内部可简化表示为多个 `GhostBottleneck`
- 标注：`lightweight semantic extraction`

箭头中间写：

- `Replace`
- 或 `Lightweight substitution`

## 10. 你这张图最应该强调的关键词

最后总结一下，你这张论文架构图最应该反复强调的关键词是：

- `YOLO12-based`
- `Lightweight`
- `Single-point replacement`
- `A2C2f -> C3Ghost`
- `Preserved multi-scale detection framework`
- `P3/P4/P5 detection`

## 11. 一句话告诉画图的人怎么画

如果让别人代你画图，你可以直接把这句话发给他：

> 请基于原始 YOLO12 三尺度检测框架画图，保留 Backbone-Neck-Head 主体拓扑不变，只在 Backbone 最深层 P5/32 位置把原始 A2C2f 模块替换成高亮的 C3Ghost 模块，并在图中明确标出这是本文唯一且核心的轻量化改动。
