#转换数据变成yolko数据格式，适用于无人机图片
import os

# ================= 配置路径与图片尺寸 =================
# 你存放这种 x_min y_min x_max y_max 格式 txt 文件的文件夹
input_labels_dir = "/home/zhanghangning/ultralytics/EVD4UAV/bb" 
# 转换后，存放标准 YOLO 格式 txt 的新文件夹
output_labels_dir = "/home/zhanghangning/ultralytics/EVD4UAV/yolo_labels_normalized" 

# 【关键】DJI Mini 3 Pro 的图片分辨率。请根据你实际的图片大小修改！
# 通常是 3840x2160 (4K) 或者 4000x3000
IMG_WIDTH = 1920  
IMG_HEIGHT = 1080
# ====================================================

os.makedirs(output_labels_dir, exist_ok=True)

converted_count = 0

for txt_name in os.listdir(input_labels_dir):
    if not txt_name.endswith('.txt'):
        continue
        
    input_path = os.path.join(input_labels_dir, txt_name)
    output_path = os.path.join(output_labels_dir, txt_name)
    
    yolo_lines = []
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = parts[0]
                x_min = float(parts[1])
                y_min = float(parts[2])
                x_max = float(parts[3])
                y_max = float(parts[4])
                
                # 1. 计算中心点和宽高
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                width = x_max - x_min
                height = y_max - y_min
                
                # 2. 归一化 (除以图片的宽和高)
                norm_x = x_center / IMG_WIDTH
                norm_y = y_center / IMG_HEIGHT
                norm_w = width / IMG_WIDTH
                norm_h = height / IMG_HEIGHT
                
                # 防止坐标越界
                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = max(0.0, min(1.0, norm_y))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))
                
                # 3. 组合成 YOLO 格式 (保留 6 位小数)
                yolo_lines.append(f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
    # 保存转换后的文件
    with open(output_path, 'w') as f:
        f.writelines(yolo_lines)
    converted_count += 1

print(f"✅ 成功转换了 {converted_count} 个标签文件！")
print(f"标准 YOLO 标签已保存在: {output_labels_dir}")