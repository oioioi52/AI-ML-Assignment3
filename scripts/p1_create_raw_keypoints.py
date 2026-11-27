# scripts/download_and_extract.py

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pytubefix import YouTube
import mediapipe as mp
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "raw_videos")
RAW_KEYPOINT_DIR = os.path.join(ROOT, "data", "raw_keypoints")
RAW_TEST_DIR = os.path.join(ROOT, "data", "test_keypoints")

os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
os.makedirs(RAW_KEYPOINT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
MODEL_COMPLEXITY = 2   # [수정] 값을 인자로 바꿀 수 있게 상수로 분리 (기본값은 원본과 같은 2)

def make_pose_model():
    # [수정] 영상마다 새 Pose 객체를 만들기 위해 함수로 감쌈 (앞 영상의 tracking 상태가 섞이지 않게)
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def download_youtube(url):
    yt = YouTube(url)
    name = yt.title.replace(" ", "_").replace("/", "_")
    filepath = f"{RAW_VIDEO_DIR}/{name}.mp4"
    yt.streams.filter(file_extension='mp4').first().download(
        output_path=RAW_VIDEO_DIR,
        filename=f"{name}.mp4"
    )
    print(f"🎬 Downloaded → {filepath}")
    return filepath, name


def extract_3d_keypoints(video_path, dir_path =RAW_KEYPOINT_DIR, name ="Data", scale=1.0):
    pose_model = make_pose_model()   # [수정]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)                    # [추가] 시간 단위 지표 계산용
    size = [int(cap.get(3)), int(cap.get(4))]          # [추가]
    pose_rows = []
    image_rows, detected = [], []                      # [추가] 2D landmark(overlay용), 프레임별 검출 여부

    pbar = tqdm(total=total_frames, desc="Extracting BlazePose 3D",
                ascii=True,          # unicode 막대 → ASCII 막대로 변경
                dynamic_ncols=False,  # 윈도우 콘솔 버그 방지,
                disable=True
                )
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scale != 1.0:   # [추가] 강건성 실험: 입력 해상도를 낮춰서 추출
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_model.process(rgb)

        if res.pose_world_landmarks:
            for i, lm in enumerate(res.pose_world_landmarks.landmark):
                pose_rows.append({
                    "frame": frame_idx,
                    "landmark": i,
                    "x": lm.x,
                    "y": -lm.y,      # flip for unity-like coords
                    "z": lm.z,
                    "visibility": lm.visibility,
                })
            image_rows.append([[lm.x, lm.y] for lm in res.pose_landmarks.landmark])   # [추가]
            detected.append(True)                                                     # [추가]
        else:                                                                         # [추가]
            image_rows.append(np.full((33, 2), np.nan).tolist())
            detected.append(False)

        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    df = pd.DataFrame(pose_rows)

    # extract local positions from pelvis


    out_path = f"{dir_path}/{name}.npz"
    # [수정] data 는 원본과 같은 형식. fps, 2D landmark 등을 함께 저장
    np.savez(out_path, data=df.to_numpy(), image=np.array(image_rows, dtype=np.float32),
             detected=np.array(detected), fps=fps, size=np.array(size), n_frames=frame_idx,
             model_complexity=MODEL_COMPLEXITY, scale=scale)
    pose_model.close()
    print(f"Saved 3D keypoints -> {out_path}  ({size[0]}x{size[1]}, {fps:.2f}fps, frames={frame_idx}, detected={int(np.sum(detected))})")

def main():
    # [수정] YouTube 다운로드 대신 data/raw_videos 에 있는 로컬 영상을 사용 (download_youtube 함수는 그대로 둠)
    import glob, argparse
    global MODEL_COMPLEXITY
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_complexity", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out_dir", default=RAW_KEYPOINT_DIR)
    a = ap.parse_args()
    MODEL_COMPLEXITY = a.model_complexity
    os.makedirs(a.out_dir, exist_ok=True)
    for video_path in sorted(glob.glob(os.path.join(RAW_VIDEO_DIR, "*.mp4"))):
        name = os.path.splitext(os.path.basename(video_path))[0]
        if a.only and name not in a.only:
            continue
        extract_3d_keypoints(video_path, a.out_dir, name, a.scale)

if __name__ == "__main__":
    main()
