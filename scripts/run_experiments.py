"""run_experiments.py - 모든 비교 실험을 한 번에 실행하고 results/*.csv 로 저장.

  1) E0~E6 방법 비교 (영상 5개)            -> results/methods_per_video.csv, methods_mean.csv
  2) 파라미터 sweep (MA / Gaussian / SavGol / One-Euro / spike 임계값 / 지면 추정) -> results/sweep_*.csv
  3) Ablation (최종 방법에서 요소를 하나씩 제거) -> results/ablation.csv
  4) 강건성: 입력 해상도를 낮춰 다시 추출한 데이터 -> results/robustness.csv
모든 수치는 이 스크립트 실행 결과이며 난수는 쓰지 않는다(같은 입력이면 항상 같은 결과).
"""
import os, sys, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from motion_lib import *
from p1_2_post_processing import enhanced_pipeline, FINAL

RAW = os.path.join(ROOT, "data", "raw_keypoints")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def load_all(d=RAW):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        X, V, fps, det = load_raw(f)
        out[os.path.splitext(os.path.basename(f))[0]] = (fill_missing(X), V, fps)
    return out


def methods(X, V, fps):
    """E0~E6. 반환: {이름: 시퀀스}"""
    b = height_basic(X)
    return {
        "E0 Raw": X,
        "E1 Basic(교수님)": b,
        "E2 Basic+MA(w=5)": smooth_ma(b, 5),
        "E3 Basic+Gauss(s=1.5)": smooth_gauss(b, 1.5),
        "E4 Basic+SavGol(9,3)": smooth_savgol(b, 9, 3),
        "E5 Outlier+Basic": height_basic(remove_outliers(X, FINAL["spike_k"], FINAL["spike_floor"])[0]),
        "E6 Enhanced(최종)": enhanced_pipeline(X, V, fps),
    }


def table(data, fn, ref="raw"):
    """fn(X,V,fps)->{name: seq}. 영상별 지표 DataFrame."""
    rows = []
    for vid, (X, V, fps) in data.items():
        st = still_mask(X, fps)
        outs = fn(X, V, fps)
        for name, Y in outs.items():
            R = X if ref == "raw" else outs[ref]
            rows.append({"video": vid, "method": name, **metrics(Y, fps, R, st)})
    return pd.DataFrame(rows)


def mean_table(df):
    return df.groupby("method", sort=False).mean(numeric_only=True).reset_index()


def main():
    data = load_all()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

    # ---------- 1) 방법 비교
    df = table(data, methods)
    df.to_csv(os.path.join(RES, "methods_per_video.csv"), index=False)
    mt = mean_table(df); mt.to_csv(os.path.join(RES, "methods_mean.csv"), index=False)
    print("\n[1] 방법 비교 (영상 5개 평균)\n", mt.round(2).to_string(index=False))

    # ---------- 2) 파라미터 sweep : 스무딩 필터는 E1(Basic) 출력에 적용, 기준(ref)도 E1
    def sweep(cands, fname, ref="E1"):
        def fn(X, V, fps):
            b = height_basic(X)
            return {"E1": b, **{k: f(b, V, fps) for k, f in cands.items()}}
        d = mean_table(table(data, fn, ref=ref)); d.to_csv(os.path.join(RES, fname), index=False)
        print(f"\n[2] {fname}\n", d[["method", "acc_m_s2", "jerk_m_s3", "still_speed_cm_s", "dev_rmse_cm",
                                     "rom_keep_pct", "peak_speed_keep_pct"]].round(2).to_string(index=False))
    sweep({f"MA w={w}": (lambda b, V, fps, w=w: smooth_ma(b, w)) for w in (3, 5, 7, 9, 11, 15)}, "sweep_ma.csv")
    sweep({f"Gauss s={s}": (lambda b, V, fps, s=s: smooth_gauss(b, s)) for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)}, "sweep_gauss.csv")
    sweep({f"SavGol w={w},p={p}": (lambda b, V, fps, w=w, p=p: smooth_savgol(b, w, p))
           for p in (2, 3, 4) for w in (5, 7, 9, 11, 15)}, "sweep_savgol.csv")
    sweep({f"OneEuro fc={c},b={be}": (lambda b, V, fps, c=c, be=be: smooth_one_euro(b, fps, c, be))
           for c in (0.5, 1.0, 2.0, 4.0) for be in (0.0, 0.3, 1.0)}, "sweep_one_euro.csv")

    # spike 임계값: 검출 개수와 보정 후 지표
    rows = []
    for k in (3, 4, 6, 8):
        for fl in (0.01, 0.02, 0.04):
            n = {vid: int(spike_mask(X, k, fl)[:, BODY].any(axis=2).sum()) for vid, (X, V, fps) in data.items()}
            d = mean_table(table(data, lambda X, V, fps: {"o": remove_outliers(X, k, fl)[0]}))
            rows.append({"k": k, "floor_cm": fl * 100, **{f"n_{v}": c for v, c in n.items()}, "n_total": sum(n.values()),
                         "acc_m_s2": d.acc_m_s2[0], "jerk_m_s3": d.jerk_m_s3[0], "dev_rmse_cm": d.dev_rmse_cm[0],
                         "peak_speed_keep_pct": d.peak_speed_keep_pct[0]})
    d = pd.DataFrame(rows); d.to_csv(os.path.join(RES, "sweep_spike.csv"), index=False)
    print("\n[2] sweep_spike.csv\n", d.round(2).to_string(index=False))

    # 지면 추정 방식
    g = {"feet/frame+clamp (=Basic)": dict(mode="feet", clamp=True),
         "feet/frame": dict(mode="feet"),
         "lowest1/frame": dict(mode="lowest", k=1),
         "lowest2/frame": dict(mode="lowest", k=2),
         **{f"lowest2 s={s}": dict(mode="lowest", k=2, sigma=s) for s in (1, 2, 3, 5)},
         **{f"lowest1 s={s}": dict(mode="lowest", k=1, sigma=s) for s in (1, 2, 3)},
         "lowest1 s=2 +envelope": dict(mode="lowest", k=1, sigma=2, envelope=True),
         "lowest2 s=2 +envelope": dict(mode="lowest", k=2, sigma=2, envelope=True),
         "lowest2 s=2 +clamp": dict(mode="lowest", k=2, sigma=2, clamp=True)}
    d = table(data, lambda X, V, fps: {k: height_adjust(X, **kw) for k, kw in g.items()})
    d.to_csv(os.path.join(RES, "sweep_ground_per_video.csv"), index=False)
    d = mean_table(d); d.to_csv(os.path.join(RES, "sweep_ground.csv"), index=False)
    print("\n[2] sweep_ground.csv\n", d[["method", "penetration_pct", "max_depth_cm", "floating_pct", "lowest_mean_cm",
                                       "root_y_jitter_m_s2", "bone_cv_pct", "dev_rmse_cm"]].round(2).to_string(index=False))

    # ---------- 3) Ablation
    ab = {"Full (O+S+B+G)": {}, "- Outlier(O)": dict(use_outlier=False), "- Smoothing(S)": dict(use_smooth=False),
          "- Bone(B)": dict(use_bone=False), "- Ground(G) -> Basic height": dict(use_ground=False),
          "S: plain SavGol only": dict(adaptive=False), "S: heavy Gauss only": dict(adaptive="heavy"),
          "order: G before S": dict(ground_first=True),
          **{f"S: adaptive heavy s={h}": dict(p={**FINAL, "heavy_sigma": h}) for h in (1.5, 3.0, 4.0)}}
    d = table(data, lambda X, V, fps: {k: enhanced_pipeline(X, V, fps, **kw) for k, kw in ab.items()})
    d.to_csv(os.path.join(RES, "ablation_per_video.csv"), index=False)
    d = mean_table(d); d.to_csv(os.path.join(RES, "ablation.csv"), index=False)
    print("\n[3] ablation.csv\n", d.round(2).to_string(index=False))

    # ---------- 4) 강건성 (저해상도 입력)
    rows = []
    for sc in ("1.0", "0.5", "0.25"):
        dd = data if sc == "1.0" else load_all(os.path.join(ROOT, "data", "robustness", f"scale_{sc}"))
        dd = {k: v for k, v in dd.items() if k in ("sidekick_high", "handstand")}
        t = table(dd, lambda X, V, fps: {"E0 Raw": X, "E1 Basic": height_basic(X), "E6 Enhanced": enhanced_pipeline(X, V, fps)})
        t.insert(0, "input_scale", sc); rows.append(t)
    d = pd.concat(rows); d.to_csv(os.path.join(RES, "robustness.csv"), index=False)
    print("\n[4] robustness.csv\n", d[["input_scale", "video", "method", "acc_m_s2", "jerk_m_s3", "bone_cv_pct",
                                       "penetration_pct", "root_y_jitter_m_s2", "peak_speed_keep_pct"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
