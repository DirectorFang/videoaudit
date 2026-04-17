# -*- coding: utf-8 -*-
"""
@Time ： 2026/3/16 15:19
@Auth ： chensonglin
@IDE ：PyCharm
  Describe:

"""
from ultralytics import YOLO

model = YOLO("yolo26s.pt")

model.train(
    data="./data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    verbose=True
)

if __name__ == '__main__':
    import torch
    from ultralytics import YOLO

    print(torch.cuda.is_available())  # 必须输出 True
    print(torch.cuda.get_device_name(0))  # 显示显卡型号
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO("./runs/detect/train/weights/best.pt")
    results = model.predict(source="./picture/水杯2.jpg",  # 您的图片路径
                            device=device,  # 自动适配 GPU/CPU
                            amp=True,  # 开启半精度加速 (省显存、提速)
                            conf=0.45,  # 提高置信度阈值 (过滤误检，如把腿当人)
                            iou=0.6,  # 调整 NMS 阈值 (合并重叠框)
                            )
    results[0].save(filename="./output_result.jpg")
    print("推理完成，图片已保存为 output_result.jpg")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        print(f"检测到类别 ID: {cls_id},{cls_name}, 置信度：{conf:.2f}")