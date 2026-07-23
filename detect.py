from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO(model="/home/zhanghangning/ultralytics/yolo26n.pt")
    model.predict(
        source="/home/zhanghangning/ultralytics/EVD4UAV/images/DJI_0157.jpg",
        save=True,
        show=False,
    )
