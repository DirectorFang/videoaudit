# -*- coding: utf-8 -*-
"""
@Time ： 2026/3/11 18:41
@Auth ： chensonglin
@IDE ：PyCharm
  Describe:

"""
import time
import torch
import os
from ultralytics import YOLO

print(torch.cuda.is_available())  # 必须输出 True
print(torch.cuda.get_device_name(0))  # 显示显卡型号
device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO("yolo26s.pt")
# model = YOLO("yolo26n-pose.pt")  # load an official model
# model = YOLO("yolo26n-seg.pt")  # load an official model
# model = YOLO("yolo26n-cls.pt")  # load an official model
results = model.predict(source="./voldemo_train/picture/运动.jpg",  # 您的图片路径
                        device=device,  # 自动适配 GPU/CPU
                        amp=True,  # 开启半精度加速 (省显存、提速)
                        conf=0.45,  # 提高置信度阈值 (过滤误检，如把腿当人)
                        iou=0.6,  # 调整 NMS 阈值 (合并重叠框)
                        )
# results[0].save(filename="./output_result.jpg")
# print("推理完成，图片已保存为 output_result.jpg")
# for box in results[0].boxes:
#     cls_id = int(box.cls[0])
#     cls_name = model.names[cls_id]
#     conf = float(box.conf[0])
#     print(f"检测到类别 ID: {cls_id},{cls_name}, 置信度：{conf:.2f}")
# for result in results:
#     xy = result.keypoints.xy  # x and y coordinates
#     xyn = result.keypoints.xyn  # normalized
#     kpts = result.keypoints.data  # x, y, visibility (if available)
#     print( result.keypoints.shape)
#     print( result.keypoints)
print(model.names)
# for result in results:
#     pred_idx = result.probs.top1
#     pred_prob = result.probs.top1conf
#     pred_class_name = result.names[int(pred_idx)]
#     print(f"Predicted class: {pred_class_name}, Confidence: {pred_prob:.2f}")