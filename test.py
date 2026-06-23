import json
import os

import cv2
import pandas as pd
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from thop import profile
from tqdm import tqdm

from ultralytics import YOLO

# ================= 配置区域 =================
# 你的模型权重路径
MODEL_PATHS = [
    "/home/zhanghangning/ultralytics/runs/train/yolo12_n2/weights/best.pt",
    "/home/zhanghangning/ultralytics/runs/train/yolo12_nwd/weights/best.pt",
    "/home/zhanghangning/ultralytics/runs/train/yolo12_yange4/weights/best.pt",
    "/home/zhanghangning/ultralytics/runs/train/yolo12_yange_aug5/weights/best.pt",
]

# 你的数据集路径
IMG_DIR = "/home/zhanghangning/ultralytics/EVD4UAV/yolo_dataset/images/val"
LABEL_DIR = "/home/zhanghangning/ultralytics/EVD4UAV/yolo_dataset/labels/val"

# 测试参数
IMG_SIZE = 640  # Ultralytics API 接受整数即可
CONF_THRES = 0.001  # 评估 mAP 时，置信度阈值通常设得很低 (如0.001)
IOU_THRES = 0.65  # NMS 的 IoU 阈值
OUTPUT_CSV = "uav_models_1.csv"
# ============================================


def convert_yolo_to_coco_gt(img_dir, label_dir, output_json="temp_gt.json"):
    """将 YOLO txt 标签动态转换为 COCO JSON 格式."""
    print("  -> 正在将 YOLO 标签转换为 COCO Ground Truth 格式...")

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "target"}],  # 假设单类别，如果是多类别需扩充
    }

    img_files = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    ann_id = 1

    for img_id, img_name in enumerate(tqdm(img_files)):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        coco_format["images"].append({"id": img_id, "file_name": img_name, "width": w, "height": h})

        txt_name = img_name.rsplit(".", 1)[0] + ".txt"
        txt_path = os.path.join(label_dir, txt_name)

        if os.path.exists(txt_path):
            with open(txt_path) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                x_center, y_center, bbox_w, bbox_h = map(float, parts[1:5])

                # 反归一化：转为 COCO 的绝对坐标 [x_min, y_min, w, h]
                abs_w = bbox_w * w
                abs_h = bbox_h * h
                x_min = (x_center * w) - (abs_w / 2)
                y_min = (y_center * h) - (abs_h / 2)

                coco_format["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)],
                        "area": round(abs_w * abs_h, 2),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_format, f)
    return output_json


def get_model_info(model, img_size=640):
    """使用 PyTorch 和 thop 准确计算 Params 和 GFLOPs."""
    # Ultralytics 的 YOLO 类包装了真实的 PyTorch 模型，底层模型在 model.model 中
    pytorch_model = model.model
    pytorch_model.eval()

    # 1. 计算参数量 (纯 PyTorch 方式)
    params = sum(p.numel() for p in pytorch_model.parameters()) / 1e6

    # 2. 计算 GFLOPs (使用 thop)
    try:
        # 获取模型当前所在的设备，并将 dummy_input 放到同设备
        device = next(pytorch_model.parameters()).device
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

        flops, _ = profile(pytorch_model, inputs=(dummy_input,), verbose=False)
        gflops = (flops * 2) / 1e9  # thop 默认算的是 MACs，乘 2 换算成 FLOPs
    except Exception as e:
        print(f"⚠️ 计算 FLOPs 失败: {e}")
        gflops = 0.0

    return params, gflops


def run_evaluation(pth_path, gt_json_path, model_name):
    # 1. 初始化模型
    print(f"\n加载模型: {model_name} ...")
    model = YOLO(pth_path)

    # 将模型显式移至 GPU 以便后续测速和测算力更准
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device_str)

    # 2. 获取参数量与算力
    params, gflops = get_model_info(model, IMG_SIZE)

    coco_gt = COCO(gt_json_path)
    predictions = []

    total_latency = 0.0
    num_images = 0

    print(f"  -> 正在对验证集进行推理 (设备: {device_str})...")
    for img_id in tqdm(coco_gt.getImgIds()):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(IMG_DIR, img_info["file_name"])

        # 3. 官方推理接口：自动 Letterbox 缩放，不破坏宽高比！自动反归一化！
        results = model.predict(
            source=img_path, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, verbose=False, device=device_str
        )
        result = results[0]

        # 累加真实的单张图推理时间 (ms)
        total_latency += result.speed["inference"]
        num_images += 1

        # 4. 解析结果：坐标已经是原图尺度的绝对坐标
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            w = x_max - x_min
            h = y_max - y_min
            score = box.conf[0].item()
            cls_id = int(box.cls[0].item())

            predictions.append(
                {
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                    "score": round(score, 5),
                }
            )

    # 计算 FPS 和 延迟
    avg_latency = total_latency / num_images if num_images > 0 else 0
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    # 5. COCO 评估
    if not predictions:
        print(f"⚠️ 警告: {model_name} 没有预测出任何有效目标！")
        return 0, 0, 0, 0, 0, params, gflops, avg_latency, fps

    pred_json = f"temp_pred_{model_name}.json"
    with open(pred_json, "w") as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    os.remove(pred_json)
    s = coco_eval.stats

    return s[3], s[9], s[0], s[1], s[2], params, gflops, avg_latency, fps


def main():
    gt_json_path = convert_yolo_to_coco_gt(IMG_DIR, LABEL_DIR)
    results = []

    for pth in MODEL_PATHS:
        if not os.path.exists(pth):
            print(f"找不到权重文件 {pth}，跳过...")
            continue

        # 提取更具辨识度的模型名称 (如 yolo12_ema)
        name = os.path.basename(os.path.dirname(os.path.dirname(pth)))

        ap_s, ar_s, map_all, map50, map75, params, gflops, latency, fps = run_evaluation(pth, gt_json_path, name)

        results.append(
            {
                "Model": name,
                "AP_S (Area<32²)": round(ap_s, 4),
                "AR_S (Area<32²)": round(ar_s, 4),
                "mAP (0.5:0.95)": round(map_all, 4),
                "AP_50": round(map50, 4),
                "AP_75": round(map75, 4),
                "Params (M)": round(params, 2),
                "GFLOPs": round(gflops, 2),
                "Latency (ms)": round(latency, 2),
                "FPS": round(fps, 2),
            }
        )

        torch.cuda.empty_cache()

    # 清理临时文件
    if os.path.exists(gt_json_path):
        os.remove(gt_json_path)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ 评估完成！结果已保存至: {OUTPUT_CSV}")

        # 这里改成了 to_string，防止因为没装 tabulate 报错
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
