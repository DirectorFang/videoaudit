# -*- coding: utf-8 -*-
"""
@Time ： 2026/3/18 17:42
@Auth ： chensonglin
@IDE ：PyCharm
  Describe:

"""
from ultralytics import YOLO


def video_contains_object(video_path: str, model_path: str, target_class_idx: int, conf_thresh: float = 0.5) -> bool:
    """
    直接判断视频中是否出现指定类别（检测模型）

    video_path: 视频文件路径
    model_path: YOLO 检测模型权重路径
    target_class_idx: 目标类别索引
    conf_thresh: 最低置信度阈值
    """
    result = False
    model = YOLO(model_path)
    i = 0
    # 流式处理视频帧
    for result in model.predict(source=video_path, stream=True):
        for box in result.boxes:
            if int(box.cls) == target_class_idx and float(box.conf) >= conf_thresh:
                return True  # 发现目标，立即返回
    return result  # 全视频未检测到目标


# 示例调用
video_path = "./video/makeshift.mp4"
model_path = "yolo26s.pt"
target_class_idx = 0
# 假设这是你的目标类别

if video_contains_object(video_path, model_path, target_class_idx):
    print("视频包含目标物体！")
else:
    print("视频未出现目标物体。")