## 项目背景  
```
自动在视频中识别公司产品线定义的产品，快速判断视频是否是带货视频 (Computer Vision Object Detection)
```
## 整体设计思路
```
视频 -> 关键帧抽取 -> 自动标注 (Grounding DINO) -> 数据集生成 -> YOLO训练 -> 模型推理 -> 视频分析

训练公司自有yolo模型，使用yolo模型判断视频中是否有画面出现了我们的产品
模型训练流程：
    简介：将标注好的数据喂给yolo模型
        数据标注目前还有的问题(2026.3.11)：
            质量要求:分辨率720p以上
            数据来源：视频关键帧提取？还是直接拿找商家要满足条件与数量的产品图  
            标注过程： 全交给外包团队，还是先用模型初步标记后让外包标准团队校对
                自动标注流程： Grounding DINO + SAM2 预标注 + 人工校验

模型使用教程：
    输入视频 -> 压缩视频到720p(具体分辨率待定) -> 切分关键帧 ->  使用yolo模型快速判别 -> 返回结果

```

## 技术栈汇总
```
python>3.10
视频处理: FFmpeg (通过 subprocess 调用)
图像处理: OpenCV, NumPy
模型推理: PyTorch, Transformers, Grounding DINO, SAM 2
数据 glue: supervision (必装，省很多事)
训练: Ultralytics (YOLOv8)
```

## 实现细则
### yolo模型选择
> 1.yolo文档官网：https://docs.ultralytics.com/
>   最终选择模型：yolo26s.pt yolo26m.pt 按实际训练效果选择
>   模型后缀说明: n s m l x   n最小 依次x最大， n最小一般用于移动端 l高精度检测 x学术研究 ， s m常规场景

```
import time
import torch
import os
from ultralytics import YOLO

print(torch.cuda.is_available())  # 必须输出 True
print(torch.cuda.get_device_name(0))  # 显示显卡型号
device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO("yolo26s.pt")
results = model.predict(source="1000-236975622-3adba09cbfed832b87e13d4e861fecf9.jpg",  # 您的图片路径
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
```


### 训练注意事项
#### 1.视频(图片)帧率处理 
> 标注数据的分辨率>720p (必要)
> 预测时,yolo不关心输入图片分辨率（会统一压缩到指定分辨率 默认640p），但如果追求走量分析，有必要统一分辨率以降低关键帧提取 yolo Resize 的计算量
```
## 压缩像素代码demo 
import subprocess
import json
import os
from pathlib import Path

def get_video_resolution(video_path):
    """获取视频分辨率"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video':
            return stream['width'], stream['height']
    raise ValueError("无法获取视频分辨率")


def compress_to_720p(input_path, output_path=None):
    """压缩视频到 720p"""
    width, height = get_video_resolution(input_path)

    if width <= 1280 and height <= 720:
        print(f"跳过：{input_path}（{width}x{height}）")
        return False

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_720p{ext}"
    # 压缩视频基础命令
    cmd = [
        'ffmpeg', '-i', input_path,
        '-vf', 'scale=-2:720',
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'medium',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-y',  # 覆盖输出
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"压缩完成：{input_path} -> {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"处理失败：{input_path}")
        print(e.stderr)
        return False


# 使用示例
if __name__ == "__main__":
    compress_to_720p('./video/video1.mp4')
```

####  2.抽帧 
##### 1.训练时抽帧： 
> 如果训练数据也来源于视频，也就是需要从视频中截图得到图片，那么有必要使用chip技术尽量减少关键帧
```
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import shutil

from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

from transformers import CLIPProcessor, CLIPModel

"""
cilp模型国外镜像慢  需配置国内源
1.pycharm运行时 run config 添加HF_ENDPOINT=https://hf-mirror.com
2.或者 
"""

# 1 模糊检测

def is_blurry(frame, threshold=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return score < threshold


# 2 Scene Detection

def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()

    scene_manager.add_detector(ContentDetector(threshold=30))

    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)

    scenes = scene_manager.get_scene_list()

    video_manager.release()

    return scenes


# 3 从scene提取关键帧


def extract_scene_frames(video_path, scenes):
    cap = cv2.VideoCapture(video_path)

    frames = []

    for scene in scenes:

        start, end = scene

        mid_frame = int((start.get_frames() + end.get_frames()) / 2)

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)

        ret, frame = cap.read()

        if ret:

            if not is_blurry(frame):
                frames.append((mid_frame, frame))

    cap.release()

    return frames

import math

def make_contact_sheet(keyframes, output="summary.jpg", cols=5):
    frames = [f for _, f in keyframes]

    h, w, _ = frames[0].shape

    rows = math.ceil(len(frames) / cols)

    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        r = i // cols
        c = i % cols

        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame

    cv2.imwrite(output, canvas)

    print("saved keyframe summary:", output)


# 4 CLIP embedding

class CLIPFilter:

    def __init__(self):

        print("Loading CLIP model...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def get_embedding(self, frame):

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        # 如果不是tensor就取pooler_output
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output

        # L2 normalize
        features = torch.nn.functional.normalize(features, p=2, dim=-1)

        return features[0].cpu()


    def filter_frames(self, frames, similarity_threshold=0.9):

        selected = []
        embeddings = []

        for frame_idx, frame in tqdm(frames):

            emb = self.get_embedding(frame)

            keep = True

            for e in embeddings:

                sim = torch.dot(emb, e)

                if sim > similarity_threshold:
                    keep = False
                    break

            if keep:
                selected.append((frame_idx, frame))
                embeddings.append(emb)

        return selected


# 5 保存帧

def save_frames(frames, output_dir="keyframes", clear_existing=True):
    """
    保存关键帧到指定目录

    :param frames: 图像帧列表 (numpy array 或 PIL Image)
    :param output_dir: 输出目录路径
    :param clear_existing: 是否清空目录下现有文件 (默认 True，覆盖写)
    :return: 保存的文件路径列表
    """
    # 转换为 Path 对象，方便操作
    output_path = Path(output_dir)

    # 创建目录（如果不存在）
    output_path.mkdir(parents=True, exist_ok=True)

    # 清空现有文件（如果启用）
    if clear_existing:
        clear_directory(output_path)

    paths = []
    for i, frame in enumerate(frames):
        # 构建文件路径
        file_path = output_path / f"frame_{i:04d}.jpg"

        # 保存图像
        cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(str(file_path))
        print(f"✅ 已保存{str(file_path)}")

    print(f"✅ 已保存 {len(paths)} 张关键帧到 {output_dir}")
    return paths


def clear_directory(dir_path):
    """
    清空目录下的所有文件和子目录

    :param dir_path: 目录路径 (Path 对象或字符串)
    """
    dir_path = Path(dir_path)

    if not dir_path.exists():
        return

    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} 不是目录")

    deleted_count = 0
    for item in dir_path.iterdir():
        try:
            if item.is_file():
                item.unlink()  # 删除文件
            elif item.is_dir():
                shutil.rmtree(item)  # 删除子目录
            deleted_count += 1
        except PermissionError:
            print(f"⚠️  警告：无法删除 {item}，权限不足")
        except Exception as e:
            print(f"⚠️  警告：删除 {item} 时出错：{e}")

    if deleted_count > 0:
        print(f"🗑️  已清空目录，删除 {deleted_count} 个旧文件/文件夹")


# 6 主流程
def extract_semantic_keyframes(video_path):
    video_name = video_path.split("/")[-1].split(".")[0]

    print("Detecting scenes...")

    scenes = detect_scenes(video_path)

    print("scene count:", len(scenes))

    print("Extracting scene frames...")

    frames = extract_scene_frames(video_path, scenes)

    print("after blur filter:", len(frames))

    print("Running CLIP deduplication...")

    clip_filter = CLIPFilter()

    frames = clip_filter.filter_frames(frames)

    print("after CLIP:", len(frames))

    paths = save_frames([f for _, f in frames],
                        output_dir=f"./output/keyframes/{video_name}")

    return frames


# ==============================
# 7 运行
# ==============================

if __name__ == "__main__":
    # video_path = "./video/test_video.mp4"
    # video_path = "./video/women.mp4"
    # video_path = "./video/men_box.mp4"
    video_path = "./video/makeshift.mp4"
    frames = extract_semantic_keyframes(video_path)

    # print("final frames:")
    #
    # for f in frames:
    #     print(f)
```
##### 2.预测时抽帧： 
> 预测试，应平衡yolo预测用时于抽帧用时，抽帧不应太复杂，至少在最初阶段应该不引入chip这个新模型降低处理速度
> 阶段1：利用yolo可以容忍模糊图片，可在按定时抽与按场景抽二选一  经测试：12min 640p视频，定时抽耗时5s，得到1642帧，按场景抽耗时119，得到119帧 ，需使用后再评估
```aiignore
    [frame_extractor.py](frame_extractor.py)
```

#### 3.标注
##### 标注数据格式 
> 一个文件夹
```
../xxx/
    images/   存储图片信息
        xxx.jpg
        xxx.jpg
        ...
    labels/   存储标注信息(通过文件名同名来映射)
        xxx.txt
        xxx.txt
        ...
    classes.txt  存储类别信息
```
#### 数据量建议 
> 若图片有代表性(无重复，画质清晰，轮廓清晰)，建议每个种类300-500张
> 若图片无代表性(重复，画质模糊，轮廓不清晰)，建议每个种类1000-2000张
  
##### 注意事项
###### 模型遗忘
> 指在二次训练时，投喂的数据会让模型忘记已经之前学习到特征 导致新模型对于之前正确预测的东西变得无法准确预测
> 在yolo中，进行二次训练时，若是模型效果增强训练(没有新增类别，只是增加数据让老类别判断得更准确)，几乎不存在遗忘
> 但若是希望模型能认识新分类，就有必要防止灾难性遗忘，在新增分类时，由于历史数据没有新类型的标识，或者新增数据没有老数据的类别标识，在二次训练时，缺失的部分会沦为背景，相应也就影响最终的预测效果
> 所以在新增分类时，历史标注数据需要补全新分类信息，新分类标注数据也需要补全历史分类信息，然后将新旧数据一起投入模型进行二次训练
>   故而：从数据准备的角度看，二次训练其实是准备全量数据进行重新训练
>        但从模型的角度看，即使二次训练投入的是全量数据，但首次训练得到的模型参数是有效的，所以二次训练会花费的算力会大大降低
> yolo灾难性遗忘的解决方案：
> 1. 采用固定预测目标模型。每个模型只预测固定目标，数据标注时，标注好图片中出现的所有类型。这样即使二次训练，也不会由于部分数据沦为背景导致遗忘。
>    极简模式：单类型预测模型，每个目标一个模型
> 2. 若必须采用多预测目标模型：
>    需维护全量标注数据，每次新增类型时，需对历史全量数据进行新类型补全，相应新数据也需要识别老分类
>        可选择技术实现：
>           a.历史数据使用GroundingDINO + SAM 自动标准新类型后，人工校验 新数据yolo标注老类型后，人工标注
>           b.标注时，标注unknown 类型，在添加新类型时，捞出unknown相关数据，识别新类型，再将所有数据(新数据 老数据 老数据修改数据)投入模型进行训练

#### 4.训练
a.
b.