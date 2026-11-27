"""bvh_compare.py - DeepMotion BVH(같은 원본 영상에서 만든 별도 추정 결과)와 MediaPipe 결과의 '일치도' 비교.

주의: BVH는 정답(GT)이 아니다. 서로 다른 두 추정기가 얼마나 비슷한 답을 내는지 보는 참고 지표다.
비교 가능한 조건 확인: 프레임 수와 FPS가 같은 영상만 사용 (sidekick_high 103f@24, handstand 161f@24).
정렬: 골반 중심 기준 좌표 -> z축 반전(오른손->왼손 좌표계) -> 시퀀스 전체에 회전+스케일 1개(Procrustes) -> 관절별 거리(cm).
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import motion_lib as ml
from p1_2_post_processing import enhanced_pipeline, create_height_corrected_target

# BVH 관절 -> MediaPipe 번호 (공통으로 존재하는 12개)
MAP = {"l_arm_JNT": 11, "r_arm_JNT": 12, "l_forearm_JNT": 13, "r_forearm_JNT": 14, "l_hand_JNT": 15, "r_hand_JNT": 16,
       "l_upleg_JNT": 23, "r_upleg_JNT": 24, "l_leg_JNT": 25, "r_leg_JNT": 26, "l_foot_JNT": 27, "r_foot_JNT": 28}


def _rot(axis, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    o, z = np.ones_like(c), np.zeros_like(c)
    m = {"X": [[o, z, z], [z, c, -s], [z, s, c]], "Y": [[c, z, s], [z, o, z], [-s, z, c]], "Z": [[c, -s, z], [s, c, z], [z, z, o]]}[axis]
    return np.stack([np.stack(r, -1) for r in m], -2)


def load_bvh(path):
    """BVH -> (관절 이름 목록, 월드 좌표 (T,J,3) [cm], frame_time). Forward Kinematics 직접 계산."""
    lines = open(path).read().split("\n")
    names, parents, offsets, chans, stack = [], [], [], [], []
    i = 0
    while not lines[i].startswith("MOTION"):
        t = lines[i].split()
        if t and t[0] in ("ROOT", "JOINT"):
            names.append(t[1]); parents.append(stack[-1] if stack else -1); stack.append(len(names) - 1)
        elif t and t[0] == "End":
            names.append(None); parents.append(stack[-1]); stack.append(len(names) - 1); chans.append([])
        elif t and t[0] == "OFFSET":
            offsets.append([float(v) for v in t[1:4]])
        elif t and t[0] == "CHANNELS":
            chans.append(t[2:])
        elif t and t[0] == "}":
            stack.pop()
        i += 1
    ft = float(lines[i + 2].split()[-1])
    M = np.array([[float(v) for v in l.split()] for l in lines[i + 3:] if l.strip()])
    T = len(M); J = len(names)
    R = np.zeros((T, J, 3, 3)); P = np.zeros((T, J, 3)); col = 0
    for j in range(J):
        loc = np.tile(np.eye(3), (T, 1, 1)); pos = np.tile(offsets[j], (T, 1))
        for ch in chans[j]:
            v = M[:, col]; col += 1
            if ch.endswith("position"):
                pos[:, "XYZ".index(ch[0])] = v
            else:
                loc = loc @ _rot(ch[0], v)
        if parents[j] < 0:
            R[:, j] = loc; P[:, j] = pos
        else:
            R[:, j] = R[:, parents[j]] @ loc
            P[:, j] = P[:, parents[j]] + np.einsum("tij,tj->ti", R[:, parents[j]], pos)
    return names, P, ft


def procrustes(A, B):
    """B(점집합)를 A에 맞추는 회전 R과 스케일 s (반사 없음). A,B: (N,3), 이미 원점 정렬."""
    U, S, Vt = np.linalg.svd(B.T @ A)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    Rm = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (B ** 2).sum()
    return Rm, s


def main():
    rows, curves = [], {}
    for vid in ("sidekick_high", "handstand"):
        X, V, fps, _ = ml.load_raw(os.path.join(ROOT, "data", "raw_keypoints", f"{vid}.npz"))
        names, P, ft = load_bvh(os.path.join(ROOT, "data", "reference_bvh", f"{vid}.bvh"))
        assert len(P) == len(X) and abs(1 / ft - fps) < 0.1, "프레임 수/FPS 불일치"
        idx = [names.index(k) for k in MAP]; mp_idx = list(MAP.values())
        Bv = P[:, idx] / 100.0                                   # cm -> m
        Bv = Bv - 0.5 * (Bv[:, 6] + Bv[:, 7])[:, None]           # 골반(양 upleg 중점) 기준
        Bv = Bv * np.array([1, 1, -1])                           # 좌표계 손잡이 맞춤: BVH는 오른손, 우리 데이터는 y만 뒤집어 왼손 좌표계
        meths = {"E0 Raw": X, "E1 Basic": create_height_corrected_target(X),
                 "O+S (뼈 고정 없음)": enhanced_pipeline(X, V, fps, use_bone=False), "E6 Enhanced": enhanced_pipeline(X, V, fps)}
        # BVH 자체의 부드러움(참고): 같은 12개 관절의 가속도
        acc = lambda A: np.linalg.norm(np.diff(A, 2, axis=0), axis=2).mean() * fps ** 2
        for name, Y in meths.items():
            A = Y[:, mp_idx] - Y[:, [23, 24]].mean(axis=1, keepdims=True)
            Rm, s = procrustes(A.reshape(-1, 3), Bv.reshape(-1, 3))
            Bal = s * Bv @ Rm
            d = np.linalg.norm(A - Bal, axis=2) * 100
            rows.append({"video": vid, "method": name, "agree_mean_cm": d.mean(), "agree_p95_cm": np.percentile(d, 95),
                         "agree_legs_cm": d[:, 6:].mean(), "agree_arms_cm": d[:, :6].mean(),
                         "acc_12j_m_s2": acc(A), "bvh_acc_12j_m_s2": acc(Bal), "scale": s})
        # 골반 높이 곡선: BVH는 전역 이동이 있으므로 '바닥에서 골반까지 높이'의 독립적인 참고값이 됨
        floor = P[:, :, 1].min()                                 # BVH 바닥 = 시퀀스 전체 최저점(상수)
        hip_b = (P[:, idx[6], 1] + P[:, idx[7], 1]) / 2 - floor
        s_h = np.median(np.linalg.norm(X[:, 23] - X[:, 25], axis=1)) / np.median(np.linalg.norm(P[:, idx[6]] - P[:, idx[8]], axis=1))
        hip_b = hip_b * s_h                                      # 허벅지 길이 비율로 m 단위 환산
        curves[vid] = {"bvh": hip_b}
        for name in ("E1 Basic", "E6 Enhanced"):
            h = meths[name][:, [23, 24], 1].mean(axis=1)
            curves[vid][name] = h
            rows.append({"video": vid, "method": name + " hip-height", "hip_rmse_cm": np.sqrt(((h - hip_b) ** 2).mean()) * 100,
                         "hip_corr": np.corrcoef(h, hip_b)[0, 1]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "bvh_agreement.csv"), index=False)
    np.savez(os.path.join(ROOT, "results", "bvh_hip_curves.npz"), **{f"{v}__{k}": c for v, d in curves.items() for k, c in d.items()})
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
