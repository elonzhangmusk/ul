import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 使用你刚写的带 P2 头的 yaml 文件初始化一个全新的模型结构
    model = YOLO("//home/zhanghangning/ultralytics/ultralytics/cfg/models/12/yolo12_yange.yaml")

    model.load("/home/zhanghangning/ultralytics/runs/train/yolo12_yange/weights/best.pt")
    model.train(
        # --- 你的专属数据路径 ---
        data="/home/zhanghangning/ultralytics/EVD4UAV/yolo_dataset/data.yaml",
        # --- 训练核心参数 ---
        imgsz=1024,  # 针对无人机小目标放大的图像尺寸
        epochs=100,  # 训练总轮数
        batch=16,  # 防爆显存的批次大小 (如果显存报警报错 OOM，把它改成 8)
        workers=8,  # 8个线程加载数据，提升读取速度
        device="1",  # 使用 1 号 GPU
        # --- 优化策略 ---
        optimizer="SGD",  # 使用 SGD 优化器
        close_mosaic=10,  # YOLOv8 特性：最后10轮关闭马赛克数据增强，提升精度
        resume=False,  # 全新开始，不继续之前的训练
        # --- 结果保存 ---
        project="/home/zhanghangning/ultralytics/runs/train",  # 训练产生的所有图表和新权重会存在这里
        name="yolo12_yange",  # 给这s次训练起个名字，你的结果会存在 runs/train/yolo12_yange 目录下
        single_cls=False,  # 数据集配了 nc:1 和 names:['car']，不需要单类强制覆盖
        cache=False,  # 不把数据全塞进内存，防止服务器内存爆满
    )
