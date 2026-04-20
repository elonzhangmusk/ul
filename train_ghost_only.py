import os
import warnings

warnings.filterwarnings('ignore')

# Disable NWD and keep pure Ghost lightweight ablation.
os.environ['YOLO_USE_NWD'] = '0'

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/zhanghangning/ultralytics/ultralytics/cfg/models/12/yolo12_yange.yaml')
    model.load('/home/zhanghangning/ultralytics/yolo12n.pt')

    model.train(
        data='/home/zhanghangning/ultralytics/EVD4UAV/yolo_dataset/data.yaml',
        imgsz=1024,
        epochs=100,
        batch=16,
        workers=8,
        device='0',
        optimizer='SGD',
        close_mosaic=10,
        resume=False,
        project='/home/zhanghangning/ultralytics/runs/train',
        name='yolo12_yange_ghost_only',
        single_cls=False,
        cache=False,
    )
