import os
import glob

# 指向你的数据集标签根目录 (请确保路径正确，通常包含 train 和 val 文件夹)
labels_dir = '/home/zhanghangning/ultralytics/EVD4UAV/yolo_dataset/labels'

# 递归查找该目录下所有的 .txt 文件
txt_files = glob.glob(os.path.join(labels_dir, '**', '*.txt'), recursive=True)

modified_count = 0

for file_path in txt_files:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    needs_rewrite = False
    
    for line in lines:
        parts = line.strip().split()
        # YOLO 标签格式: class_id x_center y_center width height
        if len(parts) >= 5: 
            if parts[0] != '0':
                parts[0] = '0'  # 强制将第一列（类别ID）改成 0
                needs_rewrite = True
            new_lines.append(' '.join(parts) + '\n')
            
    # 只有当发现非 0 类别时，才重写文件，节省 I/O 时间
    if needs_rewrite:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        modified_count += 1

print(f"清洗完成！共扫描了 {len(txt_files)} 个标签文件。")
print(f"成功修复了 {modified_count} 个包含越界类别的标签文件。")