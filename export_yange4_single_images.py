from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = Path('/home/zhanghangning/ultralytics/runs/train/yolo12_yange4/weights/best.pt')
IMAGE_DIR = Path('/home/zhanghangning/ultralytics/EVD4UAV/yolo_dataset/images/val')
OUTPUT_DIR = Path('/home/zhanghangning/ultralytics/runs/visualize/yange4_single')
IMAGE_NAMES = [
    'DJI_0159.jpg',
    'DJI_0162.jpg',
    'DJI_0163.jpg',
    'DJI_0165.jpg',
]


def draw_predictions(image, result):
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = result.names.get(cls_id, str(cls_id))
        text = f'{label} {conf:.2f}'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 80), 2)
        (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
        text_y = max(24, y1 - 8)
        cv2.rectangle(annotated, (x1, text_y - th - 8), (x1 + tw + 8, text_y + 4), (0, 220, 80), -1)
        cv2.putText(annotated, text, (x1 + 4, text_y), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return annotated


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    for image_name in IMAGE_NAMES:
        image_path = IMAGE_DIR / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {image_path}')
        result = model.predict(source=str(image_path), conf=0.25, iou=0.65, verbose=False)[0]
        annotated = draw_predictions(image, result)
        out_path = OUTPUT_DIR / image_name
        cv2.imwrite(str(out_path), annotated)
        print(out_path)


if __name__ == '__main__':
    main()
