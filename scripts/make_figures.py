"""make_figures.py - 보고서에 쓰는 모든 그래프/이미지를 results/figures/ 에 생성. (run_experiments.py, bvh_compare.py 실행 후)"""
import os, sys, glob
import numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import motion_lib as ml
from p1_2_post_processing import enhanced_pipeline, create_height_corrected_target, FINAL

FIG = os.path.join(ROOT, "results", "figures"); RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)
for f in ("C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(f):
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 220, "savefig.bbox": "tight", "legend.frameon": False})
C = {"raw": "#8a8a8a", "basic": "#e07b00", "enh": "#1f6fd0", "bad": "#d62728", "ref": "#2a9d55"}
VIDS = ["idle", "sidekick_mid", "sidekick_high", "roll", "handstand"]
LAB = {"idle": "Idle", "sidekick_mid": "Kick A", "sidekick_high": "Kick B", "roll": "Roll", "handstand": "Handstand"}


def load(v):
    X, V, fps, _ = ml.load_raw(os.path.join(ROOT, "data", "raw_keypoints", f"{v}.npz"))
    X = ml.fill_missing(X)
    return X, V, fps, create_height_corrected_target(X), enhanced_pipeline(X, V, fps)


def draw_skel(ax, P, color, yaw=0, lw=1.6, hi=None, dx=0.0):
    """정사영 스켈레톤. yaw(도)만큼 y축 회전 후 x-y 평면에 그림."""
    a = np.radians(yaw); x = P[:, 0] * np.cos(a) + P[:, 2] * np.sin(a) + dx; y = P[:, 1]
    for i, j in ml.POSE_CONNECTIONS:
        ax.plot([x[i], x[j]], [y[i], y[j]], color=color, lw=lw, solid_capstyle="round")
    ax.scatter(x[ml.BODY], y[ml.BODY], s=5, color=color, zorder=3)
    if hi is not None and len(hi):
        ax.scatter(x[hi], y[hi], s=26, color=C["bad"], zorder=4)


def frame_overlay(v, t, hi=()):
    d = np.load(os.path.join(ROOT, "data", "raw_keypoints", f"{v}.npz")); I = d["image"]
    cap = cv2.VideoCapture(os.path.join(ROOT, "data", "raw_videos", f"{v}.mp4")); cap.set(1, t); ok, im = cap.read()
    h, w = im.shape[:2]; p = (I[t] * [w, h]).astype(int)
    for a, b in ml.POSE_CONNECTIONS:
        cv2.line(im, tuple(p[a]), tuple(p[b]), (0, 220, 255), 3, cv2.LINE_AA)
    for j in ml.BODY:
        cv2.circle(im, tuple(p[j]), 4, (255, 90, 0), -1, cv2.LINE_AA)
    for j in hi:
        cv2.circle(im, tuple(p[j]), 12, (0, 0, 255), 3, cv2.LINE_AA)
    nz = np.where(im.max(axis=(0, 2)) > 12)[0]                      # 좌우 검은 띠 제거
    return cv2.cvtColor(im[:, nz.min():nz.max() + 1], cv2.COLOR_BGR2RGB)


def fig_acquisition():
    """원본 프레임 + 2D overlay / 같은 프레임의 3D 스켈레톤(정면·비스듬)"""
    picks = [("idle", 150), ("sidekick_mid", 30), ("sidekick_high", 48), ("roll", 128), ("handstand", 80)]
    fig, axs = plt.subplots(2, 5, figsize=(7.4, 3.3), gridspec_kw={"hspace": 0.05, "wspace": 0.04})
    for k, (v, t) in enumerate(picks):
        X = load(v)[0]
        axs[0, k].imshow(frame_overlay(v, t)); axs[0, k].set_title(f"{LAB[v]} (frame {t})", fontsize=8)
        ax = axs[1, k]; draw_skel(ax, X[t], C["raw"], 0, dx=-0.55); draw_skel(ax, X[t], C["enh"], 55, dx=0.55)
        ax.axhline(0, color="k", lw=0.6, ls=":"); ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.1, 1.1)
        ax.text(-0.55, -1.08, "카메라 시점", ha="center", fontsize=6); ax.text(0.55, -1.08, "55° 회전", ha="center", fontsize=6)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
    axs[1, 0].text(-1.28, 0.03, "y=0 (골반)", fontsize=5.5)
    fig.savefig(os.path.join(FIG, "s1_acquisition.png")); plt.close(fig)


def fig_problems():
    data = {v: load(v) for v in VIDS}
    # (a) 지면: raw 최저 관절 높이 / (b) 축별 떨림 / (c) visibility별 떨림
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.05), gridspec_kw={"wspace": 0.38, "width_ratios": [1.25, 1, 1]})
    ax = axs[0]
    for v in VIDS:
        X, V, fps = data[v][:3]; ax.plot(np.arange(len(X)) / fps, X[:, :, 1].min(axis=1) * 100, label=LAB[v], lw=1.1)
    ax.axhline(0, color="k", lw=0.8); ax.text(0.2, 3, "지면 grid (y=0)", fontsize=6.5)
    ax.set_xlabel("시간 (s)"); ax.set_ylabel("가장 낮은 관절의 y (cm)"); ax.set_title("(a) Raw: 전신이 지면 아래 + 높이가 흔들림", fontsize=8)
    ax.legend(fontsize=5.8, ncol=3, loc="upper center", columnspacing=0.8); ax.set_ylim(-115, 55)
    ax = axs[1]; w = 0.26
    for i, c in enumerate("xyz"):
        vals = [np.abs(np.diff(data[v][0][:, ml.BODY, i], 2, axis=0)).mean() * 1000 for v in VIDS]
        ax.bar(np.arange(5) + (i - 1) * w, vals, w, label=f"{c}" + (" (깊이)" if c == "z" else ""), color=["#9ecae1", "#4292c6", "#08306b"][i])
    ax.set_xticks(range(5)); ax.set_xticklabels([LAB[v] for v in VIDS], fontsize=6.5, rotation=20)
    ax.set_ylabel("프레임 간 2차 차분 |Δ²| (mm)"); ax.set_title("(b) 축별 떨림: 깊이(z)가 가장 큼", fontsize=8); ax.legend(fontsize=6)
    ax = axs[2]; bins = [(0, 0.5, "가려짐\n(<0.5)"), (0.5, 0.9, "중간\n(0.5~0.9)"), (0.9, 1.01, "잘 보임\n(≥0.9)")]
    accs, viss = [], []
    for v in VIDS:
        X, V, fps = data[v][:3]
        accs.append((np.linalg.norm(np.diff(X, 2, axis=0), axis=2) * fps ** 2)[:, 11:].ravel()); viss.append(V[1:-1, 11:].ravel())
    accs, viss = np.concatenate(accs), np.concatenate(viss)
    vals = [accs[(viss >= a) & (viss < b)].mean() for a, b, _ in bins]; ns = [((viss >= a) & (viss < b)).mean() * 100 for a, b, _ in bins]
    ax.bar(range(3), vals, color=[C["bad"], "#f4a582", C["raw"]])
    for i, (val, n) in enumerate(zip(vals, ns)): ax.text(i, val + 0.5, f"{val:.1f}\n(n={n:.0f}%)", ha="center", fontsize=6)
    ax.set_xticks(range(3)); ax.set_xticklabels([b[2] for b in bins], fontsize=6.5); ax.set_ylim(0, max(vals) * 1.3)
    ax.set_ylabel("평균 가속도 (m/s²)"); ax.set_title("(c) Occlusion: 가려진 관절일수록 떨림↑", fontsize=8)
    fig.savefig(os.path.join(FIG, "s1_problems_ground_jitter.png")); plt.close(fig)
    pd.DataFrame({"visibility_bin": [b[2].replace("\n", " ") for b in bins], "mean_acc_m_s2": vals, "share_pct": ns}).to_csv(os.path.join(RES, "jitter_by_visibility.csv"), index=False)

    # (d) 뼈 길이 변화 / (e) 이상치 프레임
    fig = plt.figure(figsize=(7.4, 2.05)); gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25], wspace=0.3)
    ax = fig.add_subplot(gs[0]); X, V, fps = data["sidekick_high"][:3]; t = np.arange(len(X))
    for (a, b), n, col in zip([(23, 25), (25, 27), (24, 26), (26, 28)], ["L 허벅지", "L 정강이", "R 허벅지", "R 정강이"], ["#1b9e77", "#66c2a5", "#d95f02", "#fc8d62"]):
        ax.plot(t, np.linalg.norm(X[:, a] - X[:, b], axis=1) * 100, label=n, color=col, lw=1)
    ax.axvline(58, color=C["bad"], ls="--", lw=0.8); ax.text(60, ax.get_ylim()[1] - 1.5, "f58", color=C["bad"], fontsize=6)
    ax.set_xlabel("프레임 (Kick B)"); ax.set_ylabel("뼈 길이 (cm)"); ax.set_title("(d) 뼈 길이가 프레임마다 변함", fontsize=8); ax.legend(fontsize=5.5, ncol=2, loc="lower left")
    ax = fig.add_subplot(gs[1]); cv = []
    for v in VIDS:
        X = data[v][0]; L = np.array([np.linalg.norm(X[:, a] - X[:, b], axis=1) for a, b in ml.BONES]); cv.append(L.std(1) / L.mean(1) * 100)
    cv = np.array(cv); im = ax.imshow(cv, cmap="OrRd", aspect="auto", vmin=0, vmax=20)
    ax.set_xticks(range(12)); ax.set_xticklabels(["L위팔", "L아래팔", "R위팔", "R아래팔", "L허벅지", "L정강이", "R허벅지", "R정강이", "L발", "R발", "어깨폭", "골반폭"], fontsize=5.2, rotation=90); ax.set_yticks(range(5)); ax.set_yticklabels([LAB[v] for v in VIDS], fontsize=6)
    for i in range(5):
        for j in range(12): ax.text(j, i, f"{cv[i, j]:.0f}", ha="center", va="center", fontsize=5, color="w" if cv[i, j] > 12 else "k")
    ax.grid(False); ax.set_title("뼈 길이 변동계수 CV (%)", fontsize=8)
    pd.DataFrame(cv, index=VIDS, columns=ml.BONE_NAMES).round(2).to_csv(os.path.join(RES, "bone_cv_raw.csv"))
    ax = fig.add_subplot(gs[2]); ims = [frame_overlay("sidekick_high", t, hi=(26, 28, 32) if t == 58 else ()) for t in (57, 58, 59)]
    ims = [im[60:, 180:480] for im in ims]; ax.imshow(np.hstack(ims)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for k, t in enumerate((57, 58, 59)): ax.text(k * 300 + 8, 40, f"f{t}", color=C["bad"] if t == 58 else "k", fontsize=7, weight="bold")
    ax.set_title("(e) 이상치: f58 한 프레임만 찬 다리가 지지 다리로 튐", fontsize=8)
    fig.savefig(os.path.join(FIG, "s1_problems_bone_outlier.png")); plt.close(fig)


def fig_basic():
    X, V, fps, B, E = load("handstand"); t = 80
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.1), gridspec_kw={"wspace": 0.3, "width_ratios": [1.5, 1, 1]})
    ax = axs[0]
    for k, (P, col, n) in enumerate([(X, C["raw"], "Raw"), (B, C["basic"], "Basic"), (E, C["enh"], "Enhanced")]):
        draw_skel(ax, P[t], col, 0, dx=k * 1.25); ax.text(k * 1.25, 2.02, n, ha="center", color=col, fontsize=8, weight="bold")
    ax.axhline(0, color="k", lw=0.8); ax.set_aspect("equal"); ax.set_xlim(-0.7, 3.2); ax.set_ylim(-1.15, 2.2); ax.set_xticks([]); ax.set_ylabel("y (m)")
    ax.set_title(f"(a) Handstand f{t}: Basic은 발끝을 0으로 → 몸통이 바닥에 눌림", fontsize=7.5)
    ax = axs[1]; tt = np.arange(len(X)) / fps
    c = np.load(os.path.join(RES, "bvh_hip_curves.npz"))
    ax.plot(tt, B[:, [23, 24], 1].mean(1) * 100, color=C["basic"], label="Basic"); ax.plot(tt, E[:, [23, 24], 1].mean(1) * 100, color=C["enh"], label="Enhanced")
    ax.plot(tt, c["handstand__bvh"] * 100, color=C["ref"], ls="--", label="DeepMotion BVH(참고)")
    ax.set_xlabel("시간 (s)"); ax.set_ylabel("골반 높이 (cm)"); ax.set_title("(b) Handstand 골반 높이", fontsize=8); ax.legend(fontsize=5.5, loc="center")
    ax = axs[2]; X, V, fps, B, E = load("sidekick_high"); tt = np.arange(len(X)) / fps
    for P, col, n in [(B, C["basic"], "Basic"), (E, C["enh"], "Enhanced")]:
        h = P[:, [23, 24], 1].mean(1) * 100; ax.plot(tt, h, color=col, label=n, lw=1)
    ax.set_xlabel("시간 (s)"); ax.set_ylabel("골반 높이 (cm)"); ax.set_title("(c) Kick B: Basic은 발끝 떨림이\n골반(전신) 높이로 번짐", fontsize=7.5); ax.legend(fontsize=6)
    fig.savefig(os.path.join(FIG, "s2_basic.png")); plt.close(fig)


def fig_sweeps():
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.25), gridspec_kw={"wspace": 0.36, "width_ratios": [1.5, 1, 1]})
    ax = axs[0]
    for f, n, col, mk in [("sweep_ma.csv", "Moving Avg", "#7b3294", "o"), ("sweep_gauss.csv", "Gaussian", "#008837", "s"), ("sweep_savgol.csv", "Savitzky-Golay", C["enh"], "^"), ("sweep_one_euro.csv", "One-Euro", "#b8860b", "x")]:
        d = pd.read_csv(os.path.join(RES, f)); d = d[d.method != "E1"]
        if "savgol" in f:
            for p, ls in ((2, "-"), (4, ":")):
                q = d[d.method.str.endswith(f"p={p}")]; ax.plot(q.jerk_m_s3, q.rom_keep_pct, marker=mk, ms=3.5, lw=0.9, ls=ls, color=col, label=f"{n} p={'2,3' if p == 2 else 4}")
                for _, r in q.iterrows(): ax.annotate(r.method.split("w=")[1].split(",")[0], (r.jerk_m_s3, r.rom_keep_pct), fontsize=5, xytext=(2, 3), textcoords="offset points", color=col)
        elif "euro" in f:
            ax.scatter(d.jerk_m_s3, d.rom_keep_pct, marker=mk, s=14, color=col, label=n)
        else:
            ax.plot(d.jerk_m_s3, d.rom_keep_pct, marker=mk, ms=3.5, lw=0.9, color=col, label=n)
            for _, r in d.iterrows(): ax.annotate(r.method.split("=")[1], (r.jerk_m_s3, r.rom_keep_pct), fontsize=5, xytext=(2, -7), textcoords="offset points", color=col)
    ax.axhline(95, color=C["bad"], ls="--", lw=0.8); ax.text(470, 95.3, "선정 기준: 동작 범위 보존 ≥95%", color=C["bad"], fontsize=6, ha="right")
    d = pd.read_csv(os.path.join(RES, "sweep_savgol.csv")); r = d[d.method == "SavGol w=9,p=3"].iloc[0]
    ax.scatter([r.jerk_m_s3], [r.rom_keep_pct], s=90, facecolors="none", edgecolors=C["bad"], lw=1.4, zorder=5)
    ax.annotate("선택: SG(9,3)", (r.jerk_m_s3, r.rom_keep_pct), xytext=(175, 99.2), fontsize=6.5, color=C["bad"], arrowprops=dict(arrowstyle="-", color=C["bad"], lw=0.6))
    ax.set_xlim(0, 480); ax.set_ylim(72, 100.5); ax.set_xlabel("Jerk (m/s³)  ← 낮을수록 부드러움  [E1=687]"); ax.set_ylabel("동작 범위(ROM) 보존율 (%)")
    ax.set_title("(a) 스무딩 필터·파라미터 trade-off (5개 영상 평균)", fontsize=8); ax.legend(fontsize=5.3, loc="lower right")
    ax = axs[1]; d = pd.read_csv(os.path.join(RES, "sweep_spike.csv"))
    for fl, col in ((1.0, "#fdae61"), (2.0, C["bad"]), (4.0, "#7f0000")):
        q = d[d.floor_cm == fl]; ax.plot(q.k, q.n_total, marker="o", ms=3.5, color=col, label=f"floor {fl:.0f}cm")
    ax.scatter([4], [d[(d.k == 4) & (d.floor_cm == 2)].n_total], s=90, facecolors="none", edgecolors="k", lw=1.2, zorder=5)
    ax.set_xlabel("임계값 k (×MAD)"); ax.set_ylabel("검출된 spike 수 (5개 영상 합)"); ax.set_title("(b) 이상치 임계값", fontsize=8); ax.legend(fontsize=6); ax.set_xticks([3, 4, 6, 8])
    ax = axs[2]; d = pd.read_csv(os.path.join(RES, "sweep_ground.csv")); q = d[d.method.str.match(r"lowest1 s=\d$")]
    s = [0] + [int(m.split("=")[1]) for m in q.method]; base = d[d.method == "lowest1/frame"].iloc[0]
    ax.plot(s, [base.root_y_jitter_m_s2] + list(q.root_y_jitter_m_s2), marker="o", ms=3.5, color=C["enh"], label="골반 높이 떨림 (m/s²)")
    ax2 = ax.twinx(); ax2.plot(s, [base.max_depth_cm] + list(q.max_depth_cm), marker="s", ms=3.5, color=C["bad"], label="최대 관통 깊이 (cm)"); ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax.axhline(d[d.method.str.startswith("feet/frame+clamp")].root_y_jitter_m_s2.iloc[0], color=C["basic"], ls="--", lw=0.8); ax.text(1.2, 5.9, "Basic 6.29", color=C["basic"], fontsize=6)
    ax.axvline(2, color="k", lw=0.6, ls=":"); ax.set_xlabel("지면 곡선 스무딩 σ (프레임)"); ax.set_ylabel("골반 높이 떨림 (m/s²)", color=C["enh"]); ax2.set_ylabel("최대 관통 깊이 (cm)", color=C["bad"])
    ax.set_title("(c) 지면 스무딩 σ", fontsize=8); ax.set_xticks(s)
    fig.savefig(os.path.join(FIG, "s2_sweeps.png")); plt.close(fig)


def fig_results():
    d = pd.read_csv(os.path.join(RES, "methods_per_video.csv")); d["k"] = d.method.str[:2]
    mets = [("jerk_m_s3", "Jerk (m/s³)"), ("bone_cv_pct", "뼈 길이 CV (%)"), ("root_y_jitter_m_s2", "골반 높이 떨림 (m/s²)"), ("n_spikes", "남은 spike 수 (개)")]
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 1.9), gridspec_kw={"wspace": 0.42})
    for ax, (m, n) in zip(axs, mets):
        for i, (k, col, lab) in enumerate([("E0", C["raw"], "Raw"), ("E1", C["basic"], "Basic"), ("E6", C["enh"], "Enhanced")]):
            vals = [d[(d.video == v) & (d.k == k)][m].iloc[0] for v in VIDS]; ax.bar(np.arange(5) + (i - 1) * 0.27, vals, 0.27, color=col, label=lab)
        ax.set_xticks(range(5)); ax.set_xticklabels([LAB[v] for v in VIDS], fontsize=5.6, rotation=25); ax.set_title(n, fontsize=8)
    axs[3].legend(fontsize=6, loc="upper left")
    fig.savefig(os.path.join(FIG, "s3_bars.png")); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.0), gridspec_kw={"wspace": 0.3})
    X, V, fps, B, E = load("sidekick_high"); t = np.arange(len(X)); S_only = enhanced_pipeline(X, V, fps, use_outlier=False, use_bone=False, use_ground=False)
    OS = enhanced_pipeline(X, V, fps, use_bone=False, use_ground=False)
    e = X[58, 32] - 0.5 * (X[57, 32] + X[59, 32]); c = int(np.abs(e).argmax()); ax = axs[0]; rel = lambda P: P[:, 32, c] * 100
    ax.plot(t, rel(X), color=C["raw"], lw=1, marker=".", ms=3, label="Raw"); ax.plot(t, rel(S_only), color="#7b3294", lw=1, ls="--", label="스무딩만 (S)")
    ax.plot(t, rel(OS), color=C["enh"], lw=1.3, label="이상치 제거+스무딩 (O+S)")
    ax.axvline(58, color=C["bad"], lw=0.7, ls=":"); ax.set_xlim(48, 68); ax.set_xlabel("프레임 (Kick B)"); ax.set_ylabel(f"R 발끝 {'xyz'[c]} 좌표 (cm)")
    lo = rel(X)[48:68]; ax.set_ylim(lo.min() - 5, lo.max() + 22)
    ax.set_title("(a) 이상치 f58: 스무딩만 하면 흔적이 남음", fontsize=8); ax.legend(fontsize=5.5, loc="upper left")
    pd.DataFrame({"frame": t, "raw_cm": rel(X), "S_only_cm": rel(S_only), "O_S_cm": rel(OS)}).iloc[52:65].to_csv(os.path.join(RES, "outlier_f58_trace.csv"), index=False)
    ax = axs[1]; sp = lambda P: np.linalg.norm(np.diff(P[:, 32] - P[:, [23, 24]].mean(1), axis=0), axis=1) * fps
    ax.plot(t[1:], sp(X), color=C["raw"], lw=0.9, label="Raw"); ax.plot(t[1:], sp(E), color=C["enh"], lw=1.3, label="Enhanced")
    ax.set_xlabel("프레임 (Kick B)"); ax.set_ylabel("R 발끝 속도 (m/s)"); ax.set_title("(b) 떨림은 줄고 발차기 속도 형태는 유지", fontsize=8); ax.legend(fontsize=6)
    ax = axs[2]; r = pd.read_csv(os.path.join(RES, "robustness.csv")); r["k"] = r.method.str[:2]
    for i, (k, col, lab) in enumerate([("E0", C["raw"], "Raw"), ("E1", C["basic"], "Basic"), ("E6", C["enh"], "Enhanced")]):
        q = r[r.k == k].groupby("input_scale", sort=False).jerk_m_s3.mean(); ax.bar(np.arange(3) + (i - 1) * 0.27, q.values, 0.27, color=col, label=lab)
        for x, val in zip(np.arange(3) + (i - 1) * 0.27, q.values): ax.text(x, val + 8, f"{val:.0f}", ha="center", fontsize=5.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(["원본 해상도", "1/2 해상도", "1/4 해상도"], fontsize=6.5); ax.set_ylabel("Jerk (m/s³)"); ax.set_title("(c) 강건성: 저해상도 입력에서도 같은 효과", fontsize=8); ax.legend(fontsize=6)
    fig.savefig(os.path.join(FIG, "s3_timeseries_robust.png")); plt.close(fig)


def fig_strip():
    """Raw / Basic / Enhanced 를 같은 프레임에서 나란히 (Handstand, Roll)"""
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 3.3), gridspec_kw={"hspace": 0.12})
    for ax, (v, frames) in zip(axs, [("handstand", [30, 55, 80, 105, 125]), ("roll", [60, 110, 128, 140, 165])]):
        X, V, fps, B, E = load(v); Xg = X.copy(); Xg[:, :, 1] -= 0.0
        for k, t in enumerate(frames):
            for q, (P, col) in enumerate([(X, C["raw"]), (B, C["basic"]), (E, C["enh"])]):
                bad = np.where(P[t, :, 1] < -0.01)[0] if q else []
                draw_skel(ax, P[t], col, 0, lw=1.2, dx=k * 4.2 + q * 1.25, hi=[j for j in bad if j in ml.BODY])
            ax.text(k * 4.2 + 1.25, 2.05, f"f{t}", ha="center", fontsize=7)
        ax.axhline(0, color="k", lw=0.7); ax.set_aspect("equal"); ax.set_ylim(-1.2, 2.25); ax.set_xlim(-0.9, 20.2); ax.set_xticks([]); ax.set_ylabel(f"{LAB[v]}  y (m)", fontsize=7); ax.grid(False)
    axs[0].set_title("프레임별 왼쪽부터 Raw(회색) · Basic(주황) · Enhanced(파랑),  빨간 점 = 지면 아래 1cm 넘게 내려간 관절", fontsize=7)
    fig.savefig(os.path.join(FIG, "s4_strip.png")); plt.close(fig)


if __name__ == "__main__":
    fig_acquisition(); fig_problems(); fig_basic(); fig_sweeps(); fig_results(); fig_strip()
    print("figures ->", FIG, sorted(os.listdir(FIG)))
