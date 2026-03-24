import os
import cv2

# --- 1. 路径配置 ---
# 你的原始数据集根目录
BASE_DIR = '/home/zhanghangning/ultralytics/EVD4UAV/yolo_dataset'
# 生成的新数据集根目录 (全切成 640x640 后的数据集)
OUT_DIR = '/home/zhanghangning/ultralytics/EVD4UAV/yolo_dataset_cropped'

# --- 2. 切图参数配置 ---
CROP_SIZE = 640
OVERLAP = 0.2
STRIDE = int(CROP_SIZE * (1 - OVERLAP))

def yolo_to_pixel(yolo_box, img_w, img_h):
    cls_id, x_c, y_c, bw, bh = float(yolo_box[0]), float(yolo_box[1]), float(yolo_box[2]), float(yolo_box[3]), float(yolo_box[4])
    x_min = int((x_c - bw / 2) * img_w)
    y_min = int((y_c - bh / 2) * img_h)
    x_max = int((x_c + bw / 2) * img_w)
    y_max = int((y_c + bh / 2) * img_h)
    return int(cls_id), x_min, y_min, x_max, y_max

def pixel_to_yolo(cls_id, x_min, y_min, x_max, y_max, crop_w, crop_h):
    x_c = ((x_min + x_max) / 2) / crop_w
    y_c = ((y_min + y_max) / 2) / crop_h
    bw = (x_max - x_min) / crop_w
    bh = (y_max - y_min) / crop_h
    return f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"

def process_split(split_name):
    """分别处理 train 或 val 文件夹"""
    print(f"\n开始处理 {split_name} 数据集...")
    img_dir = os.path.join(BASE_DIR, 'images', split_name)
    label_dir = os.path.join(BASE_DIR, 'labels', split_name)
    
    out_img_dir = os.path.join(OUT_DIR, 'images', split_name)
    out_label_dir = os.path.join(OUT_DIR, 'labels', split_name)
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    
    for idx, img_name in enumerate(images):
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) >= 5: # 确保是正常的检测框格式
                        boxes.append(yolo_to_pixel(data, img_w, img_h))

        for y in range(0, img_h, STRIDE):
            for x in range(0, img_w, STRIDE):
                x_start = min(x, img_w - CROP_SIZE)
                y_start = min(y, img_h - CROP_SIZE)
                x_end = x_start + CROP_SIZE
                y_end = y_start + CROP_SIZE

                crop_img = img[y_start:y_end, x_start:x_end]
                crop_boxes = []

                for (cls_id, bx_min, by_min, bx_max, by_max) in boxes:
                    inter_x_min = max(x_start, bx_min)
                    inter_y_min = max(y_start, by_min)
                    inter_x_max = min(x_end, bx_max)
                    inter_y_max = min(y_end, by_max)

                    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
                        orig_area = (bx_max - bx_min) * (by_max - by_min)
                        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                        
                        if inter_area / orig_area > 0.4:
                            new_x_min = inter_x_min - x_start
                            new_y_min = inter_y_min - y_start
                            new_x_max = inter_x_max - x_start
                            new_y_max = inter_y_max - y_start
                            
                            yolo_str = pixel_to_yolo(cls_id, new_x_min, new_y_min, new_x_max, new_y_max, CROP_SIZE, CROP_SIZE)
                            crop_boxes.append(yolo_str)

                # 如果切出来的图里有目标，才保存（避免产生大量纯背景图拖慢训练）
                # 如果你想保留背景图做 Negative sample，可以去掉这个 if 判断
                if crop_boxes:
                    base_name = os.path.splitext(img_name)[0]
                    crop_name = f"{base_name}_{x_start}_{y_start}"
                    
                    cv2.imwrite(os.path.join(out_img_dir, f"{crop_name}.jpg"), crop_img)
                    with open(os.path.join(out_label_dir, f"{crop_name}.txt"), 'w') as f:
                        f.write('\n'.join(crop_boxes))
                        
        if idx % 100 == 0 and idx > 0:
            print(f"[{split_name}] 已处理 {idx}/{len(images)} 张图片...")

if __name__ == '__main__':
    # 依次处理训练集和验证集
    for split in ['train', 'val']:
        if os.path.exists(os.path.join(BASE_DIR, 'images', split)):
            process_split(split)
    print("\n所有切图与标签转换完成！新数据集位于:", OUT_DIR)