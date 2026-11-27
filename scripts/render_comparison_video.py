"""render_comparison_video.py - 원본 영상 | Raw | Basic | Enhanced 를 한 화면에 동시 재생하는 비교 영상 생성.

  python scripts/render_comparison_video.py            -> videos/compare_<name>.mp4 (5개) + videos/demo_all.mp4
표시 요소: 지면 grid(y=0), 문제 관절 강조(빨강: 지면 아래 1cm 초과 / 주황 테두리: spike 이상치), 프레임별 지표.
"""
import os, sys, glob, subprocess, shutil
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import motion_lib as ml

W, H = 1280, 720
PW = 300                                   # 3D 패널 폭
COL = {"Raw": (138, 138, 138), "Basic": (0, 123, 224), "Enhanced": (208, 111, 31)}   # BGR
FONT = next((f for f in ("C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc") if os.path.exists(f)), None)
font = lambda s: ImageFont.truetype(FONT, s) if FONT else ImageFont.load_default()
LAB = {"idle": "Idle (일반 동작)", "sidekick_mid": "Kick A (빠른 동작)", "sidekick_high": "Kick B (빠른 동작)", "roll": "Roll (가림·바닥 접촉)", "handstand": "Handstand (거꾸로 선 자세)"}


def project(P, yaw=35, pitch=10, scale=185, cx=PW // 2, cy=455):
    a, b = np.radians(yaw), np.radians(pitch)
    x = P[..., 0] * np.cos(a) + P[..., 2] * np.sin(a); z = -P[..., 0] * np.sin(a) + P[..., 2] * np.cos(a)
    y = P[..., 1] * np.cos(b) - z * np.sin(b)
    return np.stack([cx + x * scale, cy - y * scale], axis=-1).astype(int)


GRID = np.array([[[x, 0, -1], [x, 0, 1]] for x in np.arange(-1, 1.01, 0.25)] + [[[-1, 0, z], [1, 0, z]] for z in np.arange(-1, 1.01, 0.25)])


def panel(P, name, spikes=None):
    im = np.full((H - 110, PW, 3), 250, np.uint8)
    for a, b in project(GRID):
        cv2.line(im, tuple(a), tuple(b), (205, 205, 205), 1, cv2.LINE_AA)
    q = project(P); col = COL[name]
    for i, j in ml.POSE_CONNECTIONS:
        cv2.line(im, tuple(q[i]), tuple(q[j]), col, 3, cv2.LINE_AA)
    below = P[:, 1] < -0.01
    for j in ml.BODY:
        cv2.circle(im, tuple(q[j]), 5 if below[j] else 3, (40, 40, 220) if below[j] else col, -1, cv2.LINE_AA)
    if spikes is not None:
        for j in np.where(spikes)[0]:
            cv2.circle(im, tuple(q[j]), 10, (0, 140, 255), 2, cv2.LINE_AA)
    return im, int(below[ml.BODY].sum())


def render(name):
    d = np.load(os.path.join(ROOT, "data", "raw_keypoints", f"{name}.npz")); I = d["image"]; fps = float(d["fps"])
    seqs = {k: np.load(os.path.join(ROOT, "data", "processed", f"{name}_{t}.npy")) for k, t in (("Raw", "raw"), ("Basic", "target"), ("Enhanced", "enhanced"))}
    spikes = ml.spike_mask(seqs["Raw"]).any(axis=2)
    thigh = lambda P: np.linalg.norm(P[23] - P[25]) * 100
    cap = cv2.VideoCapture(os.path.join(ROOT, "data", "raw_videos", f"{name}.mp4"))
    tmp = os.path.join(ROOT, "videos", f"_tmp_{name}.mp4"); out = os.path.join(ROOT, "videos", f"compare_{name}.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    T = len(seqs["Raw"]); VW = W - 3 * PW - 40
    for t in range(T):
        ok, fr = cap.read()
        if not ok: break
        nz = np.where(fr.max(axis=(0, 2)) > 12)[0]; fr = fr[:, nz.min():nz.max() + 1]; h, w = fr.shape[:2]
        p = (I[t] * [d["size"][0], d["size"][1]]).astype(int) - [nz.min(), 0]
        for a, b in ml.POSE_CONNECTIONS:
            cv2.line(fr, tuple(p[a]), tuple(p[b]), (0, 220, 255), 2, cv2.LINE_AA)
        s = min(VW / w, (H - 110) / h); fr = cv2.resize(fr, (int(w * s), int(h * s)))
        canvas = np.full((H, W, 3), 255, np.uint8)
        y0 = 60 + (H - 110 - fr.shape[0]) // 2; x0 = 10 + (VW - fr.shape[1]) // 2
        canvas[y0:y0 + fr.shape[0], x0:x0 + fr.shape[1]] = fr
        info = []
        for k, key in enumerate(("Raw", "Basic", "Enhanced")):
            im, nb = panel(seqs[key][t], key, spikes[t] if key == "Raw" else None)
            x = VW + 20 + k * (PW + 5); canvas[60:60 + im.shape[0], x:x + PW] = im
            info.append((x, key, nb, thigh(seqs[key][t]), seqs[key][t][[23, 24], 1].mean() * 100))
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(pil)
        dr.text((14, 12), f"{LAB.get(name, name)}   frame {t + 1}/{T}   {t / fps:4.2f}s", font=font(24), fill=(20, 20, 20))
        dr.text((14, H - 40), "원본 영상 + MediaPipe 2D", font=font(17), fill=(60, 60, 60))
        dr.text((W - 470, 18), "● 빨강: 지면 아래 관절   ○ 주황: 이상치(spike)", font=font(16), fill=(90, 90, 90))
        for x, key, nb, th, hip in info:
            c = COL[key][::-1]
            dr.text((x + 8, 64), {"Raw": "Raw (MediaPipe 원본)", "Basic": "Basic (교수님 높이 보정)", "Enhanced": "Enhanced (최종)"}[key], font=font(19), fill=c)
            dr.text((x + 8, H - 46), f"지면 아래 관절 {nb:2d}개 · 골반 높이 {hip:5.1f}cm", font=font(15), fill=(200, 30, 30) if nb else (60, 60, 60))
            dr.text((x + 8, H - 24), f"왼쪽 허벅지 길이 {th:4.1f}cm", font=font(15), fill=(60, 60, 60))
        vw.write(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
    vw.release(); cap.release()
    if shutil.which("ffmpeg"):                      # 재생 호환성을 위해 H.264 로 다시 인코딩
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], check=True); os.remove(tmp)
    else:
        os.replace(tmp, out)
    print("saved", out, f"({T} frames)")
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "videos"), exist_ok=True)
    names = sys.argv[1:] or ["idle", "sidekick_mid", "sidekick_high", "roll", "handstand"]
    outs = [render(n) for n in names]
    if shutil.which("ffmpeg") and len(outs) > 1:
        lst = os.path.join(ROOT, "videos", "_list.txt"); open(lst, "w").write("".join(f"file '{os.path.basename(o)}'\n" for o in outs))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-vf", "fps=30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        os.path.join(ROOT, "videos", "demo_all.mp4")], check=True); os.remove(lst)
        print("saved videos/demo_all.mp4")
