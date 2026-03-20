# -*- coding: utf-8 -*-
"""
@Time ： 2026/3/12 14:12
@Auth ： chensonglin
@IDE ：PyCharm
  Describe:

"""
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
