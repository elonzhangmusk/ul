import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 使用单层替换 C3Ghost 的 yaml 文件初始化模型
    model = YOLO('/home/zhanghangning/ultralytics/ultralytics/cfg/models/12/yolo12_yange.yaml') 

    # 2. 加载预训练权重 (有助于加速收敛)
    model.load('/home/zhanghangning/ultralytics/runs/train/yolo12_yange_aug2/weights/best.pt')
    
    # 3. 开始训练
    model.train(
        # --- 数据集与硬件配置 ---
        data='/home/zhanghangning/ultralytics/EVD4UAV/yolo_dataset/data.yaml', 
        imgsz=1024,          # 针对无人机小目标放大的图像尺寸
        epochs=100,          # 建议先跑 100 轮看趋势
        batch=16,            # 防爆显存，如果 OOM 改成 8
        workers=8,           # 提升数据读取速度
        device='0',          # 使用 1 号 GPU
        
        # --- 训练与优化策略 ---
        optimizer='SGD',     
        close_mosaic=10,     # 最后10轮关闭马赛克，提升最终精度
        resume=False,        
        
        # 👇👇👇 --- 新增：UAV 无人机专属“数据增强外挂” --- 👇👇👇
        copy_paste=0.3,      # 开启 30% 概率将小目标复制粘贴到空白背景，极大增加正样本
        scale=0.2,           # 限制缩放幅度(原默认0.5)，防止小目标被缩放成噪点
        erasing=0.0,         # 彻底关闭随机遮挡(打马赛克)，防止小目标被完全盖住
        mixup=0.1,           # 开启 10% 概率图像混合，增强抗干扰能力
        degrees=10.0,        # 增加轻微随机旋转，模拟无人机偏航视角
        # 👆👆👆 ------------------------------------------- 👆👆👆
        
        # --- 结果保存 ---
        project='/home/zhanghangning/ultralytics/runs/train', 
        name='yolo12_yange_aug', # 改个名字，标记这是加了数据增强的版本
        
        single_cls=False,    
        cache=False,         
    )