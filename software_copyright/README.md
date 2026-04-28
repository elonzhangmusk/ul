# 无人机视角车辆小目标检测与评估分析软件 V1.0

## 软件概述
本软件用于无人机视角车辆小目标检测的数据处理、训练推理、评估统计和结果导出。软件围绕 EVD4UAV 数据集进行设计，支持原始标注整理、图像切片、类别修复、YOLO/COCO 格式转换、多模型训练与推理、多指标评估以及实验结果归档。

本软件的基础检测能力依赖 Ultralytics YOLO 开源框架，申请人基于该框架完成了面向无人机车辆小目标检测场景的二次开发。软著权利主张范围主要包括数据处理流程、训练推理封装、评估统计、结果导出、模型配置与场景化改进模块，不包括第三方开源框架原始代码。

## 功能模块
- 数据集配置模块：提供 EVD4UAV 数据集 YAML 配置与目录约定。
- 标注转换模块：支持原始标注转换为 YOLO 标注格式。
- 图像切片模块：支持高分辨率图像切片与标签同步处理。
- 模型训练模块：封装 YOLO 训练流程与训练参数。
- 目标检测推理模块：封装单图像和文件夹批量推理流程。
- 检测结果评估模块：基于 COCOeval 统计 AP_S、AR_S、mAP、AP50、AP75 等指标。
- 指标统计与导出模块：统计 Params、GFLOPs、Latency、FPS，并导出 CSV/JSON。
- 场景化模型改进模块：包含 NWD、ASFF、GhostConv/C3Ghost、P2 分支等改进配置。

## 第三方开源组件边界
本软件在目标检测基础功能中参考并集成了 Ultralytics YOLO 开源目标检测框架。申请人基于该框架完成了面向无人机视角车辆小目标检测场景的数据处理、标签转换、模型训练调度、推理封装、检测结果解析、COCO 格式评估、指标统计与结果导出等功能开发，并针对小目标检测任务进行了模型结构和损失函数的场景化改进。

本次软件著作权登记所主张的原创内容为申请人独立完成的上述业务流程、数据处理、训练评估、模型配置和改进模块代码，不包括第三方开源框架原始代码。第三方开源组件仍遵循其原始开源许可协议。

## 目录结构
```text
software_copyright/
├── README.md
├── THIRD_PARTY_NOTICE.md
├── docs/
│   └── software_description_outline.md
├── dataset_tools/
├── services/
├── model_modules/
├── configs/
└── scripts/
```

## 运行示例
### 1. 转换标签
```bash
python dataset_tools/label_convert.py ^
  --input-dir path/to/raw_labels ^
  --output-dir path/to/yolo_labels ^
  --img-width 1920 ^
  --img-height 1080 ^
  --class-map car=0
```

### 2. 切图
```bash
python dataset_tools/crop_dataset.py ^
  --image-dir path/to/images ^
  --label-dir path/to/labels ^
  --output-dir path/to/output ^
  --crop-size 640 ^
  --overlap 0.2 ^
  --min-area-ratio 0.4
```

### 3. 训练
```bash
python scripts/run_train.py --model configs/yolo12_yange.yaml --data configs/evd4uav.yaml --epochs 100 --imgsz 1024 --device 0
```

### 4. 推理
```bash
python scripts/run_predict.py --weights path/to/best.pt --source path/to/images --imgsz 1024 --device 0
```

### 5. 评估
```bash
python scripts/run_evaluate.py --weights path/to/a.pt path/to/b.pt --img-dir path/to/images/val --label-dir path/to/labels/val --out results.csv
```

## 说明
本目录用于软著申请材料整理，不包含训练数据、权重文件、运行缓存或大规模实验结果。实际使用时，用户应根据自己的数据集路径和实验环境调整命令行参数。

