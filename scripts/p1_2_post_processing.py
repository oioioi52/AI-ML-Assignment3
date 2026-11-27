import os
import numpy as np
import pandas as pd
from glob import glob
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.viser_test import PoseViser

RAW_KEYPOINTS_DIR = os.path.join(ROOT, "data", "raw_keypoints")
OUT_DIR = os.path.join(ROOT, "data", "processed")

os.makedirs(OUT_DIR, exist_ok=True)


def load_raw_keypoints(npz_path):
    """Load BlazePose raw data (T,33,3)."""
    data = np.load(npz_path, allow_pickle=True)["data"]
    df = pd.DataFrame(data, columns=["frame","landmark","x","y","z","visibility"])

    pts_seq = []

    for f, group in df.groupby("frame"):
        group = group.sort_values("landmark")
        pts = group[["x","y","z"]].values  # (33,3)
        pts_seq.append(pts)

    return np.array(pts_seq)   # (T,33,3)


def create_height_corrected_target(raw_seq):
    """
    raw_seq: (T,33,3)
    Return target height corrected (T,33,3)
    """

    seq = raw_seq.copy()

    # BlazePose foot joints
    LEFT_FOOT = 31
    RIGHT_FOOT = 32

    # 모든 프레임에서 발의 y값 중 최소값 → ground level
    foot_y = np.minimum(seq[:, LEFT_FOOT, 1], seq[:, RIGHT_FOOT, 1])
    
    # reshape to (T,1) → broadcast to (T,33)
    frame_ground = foot_y[:, None]

    # 전체 시퀀스를 ground=0으로 맞춤
    seq[:, :, 1] -= frame_ground

    # 땅 아래(-)로 떨어지지 않게 clamp
    seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)

    return seq

# ======================================================================
# [추가] Bonus: 추가 후처리 (Enhanced).  위의 create_height_corrected_target() 은 교수님 코드 그대로이다.
#   O. spike 이상치 제거 + 보간   : 한 프레임만 관절이 엉뚱한 곳으로 튀는 추적 오류 제거
#   S. visibility 적응형 스무딩   : 잘 보이는 관절은 Savitzky-Golay(동작 보존), 가려진 관절은 Gaussian
#   B. 뼈 길이 고정               : 프레임마다 팔다리 길이가 변하는 문제 제거
#   G. 지면 보정 개선             : 발끝이 아니라 가장 낮은 관절을 지면으로, 지면 곡선 스무딩, clamp 없음
#      (물구나무/구르기처럼 발이 바닥에 없는 자세에서 몸이 눌리는 문제 해결)
# 알고리즘 구현은 scripts/motion_lib.py, 파라미터 근거는 scripts/run_experiments.py 의 실험 결과
# ======================================================================
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import motion_lib as ml

FINAL = dict(spike_k=4.0, spike_floor=0.02, sg_window=9, sg_poly=3, heavy_sigma=2.0,
             ground_k=1, ground_sigma=2.0)


def enhanced_pipeline(raw_seq, vis, fps, use_outlier=True, use_smooth=True, use_bone=True, use_ground=True,
                      adaptive=True, ground_first=False, p=FINAL):
    """raw_seq: (T,33,3), vis: (T,33).  use_* 인자는 ablation 실험용 스위치."""
    seq = raw_seq.copy()

    def ground(s):
        if use_ground:
            return ml.height_adjust(s, mode="lowest", k=p["ground_k"], sigma=p["ground_sigma"], clamp=False)
        return create_height_corrected_target(s)

    if use_outlier:                                   # O
        seq, _ = ml.remove_outliers(seq, p["spike_k"], p["spike_floor"])
    if ground_first:
        seq = ground(seq)
    if use_smooth:                                    # S
        light = ml.smooth_savgol(seq, p["sg_window"], p["sg_poly"])
        heavy = ml.smooth_gauss(seq, p["heavy_sigma"])
        seq = ml.smooth_vis_adaptive(seq, vis, light, heavy) if adaptive is True else (heavy if adaptive == "heavy" else light)
    if use_bone:                                      # B
        seq = ml.stabilize_bones(seq)
    if not ground_first:                              # G
        seq = ground(seq)
    return seq
# ============================ [추가 끝] ================================


def export_unity_csv(motion, output_path):
    """
    motion: (T, 33, 3)
    Export Unity-compatible CSV.
    """

    T, n_landmarks, _ = motion.shape

    rows = []

    for t in range(T):
        for landmark in range(n_landmarks):

            x, y, z = motion[t, landmark]

            rows.append([
                t,
                landmark,
                float(x),
                float(y),
                float(z),
                1.0
            ])

    df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "landmark",
            "x",
            "y",
            "z",
            "visibility"
        ]
    )

    # 문자열 경로 처리
    output_path = os.path.abspath(output_path)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(f"Unity CSV saved: {output_path}")
    print(f"Frames: {T}")
    print(f"Landmarks: {n_landmarks}")
    print(f"Total rows: {len(df)}")

def main():
    paths = glob(f"{RAW_KEYPOINTS_DIR}/*.npz")
    print(f"Found {len(paths)} raw keypoint files")

    for path in paths:
        base = os.path.basename(path).replace(".npz", "")

        # -------------------------------
        # 1) Load raw BlazePose sequence
        # -------------------------------
        raw_seq = load_raw_keypoints(path)      # (T,33,3)
        raw_out_path = os.path.join(OUT_DIR, f"{base}_raw.npy")
        np.save(raw_out_path, raw_seq)
        print(f"Saved raw → {raw_out_path}")

        # ---------------------------------------
        # 2) Create height-corrected target (GT)
        # ---------------------------------------
        target_seq = create_height_corrected_target(raw_seq)
        target_out_path = os.path.join(OUT_DIR, f"{base}_target.npy")
        np.save(target_out_path, target_seq)
        print(f"Saved target → {target_out_path}")

        OUTPUT_PATH = (
                os.path.join(OUT_DIR,
                f"{base}_target.csv")
            )
        export_unity_csv(target_seq,OUTPUT_PATH)

        # ---------------------------------------
        # 3) [추가] Enhanced 후처리 결과 저장
        # ---------------------------------------
        _, vis, fps, _ = ml.load_raw(path)                  # visibility, fps
        enhanced_seq = enhanced_pipeline(raw_seq, vis, fps)
        np.save(os.path.join(OUT_DIR, f"{base}_enhanced.npy"), enhanced_seq)
        export_unity_csv(raw_seq, os.path.join(OUT_DIR, f"{base}_raw.csv"))
        export_unity_csv(enhanced_seq, os.path.join(OUT_DIR, f"{base}_enhanced.csv"))
        print(f"Saved enhanced → {base}_enhanced.npy")

    print("\nDone processing all keypoint files.")

    # IF you want to visualize the output  ([수정] 다른 스크립트에서 반복 실행할 수 있도록 주석 처리.
    #                                         3개 동시 비교는 scripts/viser_compare.py 사용)
    # vis = PoseViser(fps=60)
    # raw = np.load(f"./data/processed/{base}_target.npy")
    # print(raw.shape)
    # vis.play_sequence(raw)


if __name__ == "__main__":
    main()
