import os
import cv2
import shutil

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def get_frame_name(frame_id, video_fps):
    timestamp = int(frame_id / video_fps)

    h = timestamp // 3600
    m = (timestamp % 3600) // 60
    s = timestamp % 60

    return f"{h:02d}_{m:02d}_{s:02d}.jpg"


def extract_frames_fixed_interval(
        video_path,
        fps=1,
        output_dir=None
):
    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frames = []
    frame_id = 0

    # ======================
    # 如果需要落盘
    # ======================
    if output_dir:

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

    # ======================
    # 开始读取视频
    # ======================
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % frame_interval == 0:

            name = get_frame_name(frame_id, video_fps)

            if output_dir:

                frame_path = os.path.join(
                    output_dir,
                    f"{name}.jpg"
                )

                cv2.imwrite(frame_path, frame)

                frames.append(frame_path)

            else:

                # 返回 (时间戳, frame)
                frames.append((name, frame))

        frame_id += 1

    cap.release()

    return frames


def extract_frames_by_scene(
        video_path,
        threshold=30,
        output_dir=None
):
    """
    按场景变化抽帧

    Parameters
    ----------
    video_path : str
        视频路径

    threshold : int
        场景变化敏感度
        越小 -> 场景越多
        推荐 25~35

    output_dir : str
        如果传入则保存图片，否则返回frame

    Returns
    -------
    list
        output_dir=None -> [(timestamp, frame)]
        output_dir!=None -> [frame_path]
    """

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    # 检测场景
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(video_path)

    frames = []

    # ======================
    # 如果需要保存
    # ======================
    if output_dir:

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

    # ======================
    # 每个场景取第一帧
    # ======================
    for scene in scene_list:

        start_frame = scene[0].get_frames()

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        ret, frame = cap.read()

        if not ret:
            continue

        name = get_frame_name(cap.get(cv2.CAP_PROP_POS_MSEC), 1000)

        if output_dir:

            frame_path = os.path.join(
                output_dir,
                f"{name}.jpg"
            )

            cv2.imwrite(frame_path, frame)

            frames.append(frame_path)

        else:

            frames.append((name, frame))

    cap.release()

    return frames
