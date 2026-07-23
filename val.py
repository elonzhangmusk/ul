# 直接对你的 1920x1080 的 Val（验证集）原图进行完整的切片推理和映射。代码会自动完成“切图 -> 预测 -> 坐标映射回原图 -> NMS 去重”的全过程：

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 1. 加载你用 640 切图训练出来的权重 (假设你已经训练完了)
# 注意这里的 model_type 是 'ultralytics'，完美兼容你的 YOLO 架构
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="/home/zhanghangning/ultralytics/runs/train/yolo26_evd4uav/weights/best.pt",
    confidence_threshold=0.3,  # 置信度阈值
    device="cuda:0",  # 使用 0 号 GPU
)

# 2. 读取一张 1920x1080 的验证集原图
image_path = "/home/zhanghangning/ultralytics/EVD4UAV/rotated_bb/your_val_image.jpg"

# 3. 执行切片推理与坐标映射 (这就是你要的核心功能)
result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=640,  # 推理时的切片高度
    slice_width=640,  # 推理时的切片宽度
    overlap_height_ratio=0.2,  # 高度方向 20% 重叠 (防止目标被腰斩)
    overlap_width_ratio=0.2,  # 宽度方向 20% 重叠
    perform_standard_pred=False,  # 是否同时对 1920 缩放后的整图做一次推理并合并 (轻量化建议选 False)
)

# 4. 提取映射回 1920x1080 原图后的真实坐标
for object_prediction in result.object_prediction_list:
    # 这里的 bbox 就是已经加上了 offset 并做完 NMS 去重后的全局绝对坐标
    bbox = object_prediction.bbox
    x_min = int(bbox.minx)
    y_min = int(bbox.miny)
    x_max = int(bbox.maxx)
    y_max = int(bbox.maxy)

    # 获取类别名称和置信度
    category_name = object_prediction.category.name
    score = object_prediction.score.value

    print(f"检测到 {category_name} 置信度: {score:.2f}, 原图坐标: [{x_min}, {y_min}, {x_max}, {y_max}]")

# 5. 可视化并保存结果，让你直观看到映射后的框准不准
result.export_visuals(export_dir="/home/zhanghangning/ultralytics/runs/val_sahi_output/")
print("映射与可视化完成，请去对应目录查看结果图！")
