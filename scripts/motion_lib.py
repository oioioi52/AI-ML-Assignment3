"""motion_lib.py - 후처리 알고리즘과 평가 지표 모음 (모든 실험 스크립트가 공유)

좌표계: MediaPipe pose_world_landmarks. 단위 m, 원점 = 양 골반 중점, y는 위쪽 + (추출 시 부호 반전).
배열 모양: X = (T, 33, 3), V(visibility) = (T, 33)
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, maximum_filter1d
from scipy.signal import savgol_filter

JOINT_NAMES = ["nose", "L_eye_in", "L_eye", "L_eye_out", "R_eye_in", "R_eye", "R_eye_out", "L_ear", "R_ear",
               "mouth_L", "mouth_R", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist",
               "L_pinky", "R_pinky", "L_index", "R_index", "L_thumb", "R_thumb", "L_hip", "R_hip", "L_knee",
               "R_knee", "L_ankle", "R_ankle", "L_heel", "R_heel", "L_foot", "R_foot"]
POSE_CONNECTIONS = [(11, 12), (12, 14), (14, 16), (11, 13), (13, 15), (11, 23), (12, 24), (23, 24),
                    (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32)]
BODY = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]   # 지표 계산에 쓰는 17개 관절
BONES = [(11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28),
         (27, 31), (28, 32), (11, 12), (23, 24)]                                   # 길이가 일정해야 하는 뼈 12개
BONE_NAMES = ["L_upperarm", "L_forearm", "R_upperarm", "R_forearm", "L_thigh", "L_shin", "R_thigh", "R_shin",
              "L_foot", "R_foot", "shoulder_w", "hip_w"]
L_FOOT, R_FOOT = 31, 32


# ------------------------------------------------------------------ I/O
def load_raw(npz_path):
    """교수님 형식 npz -> X(T,33,3), V(T,33), fps, detected(T). 검출 실패 프레임은 NaN."""
    d = np.load(npz_path, allow_pickle=True)
    rows = d["data"]
    T = int(d["n_frames"]) if "n_frames" in d else int(rows[:, 0].max()) + 1
    X = np.full((T, 33, 3), np.nan)
    V = np.zeros((T, 33))
    fr, lm = rows[:, 0].astype(int), rows[:, 1].astype(int)
    X[fr, lm] = rows[:, 2:5]
    V[fr, lm] = rows[:, 5]
    fps = float(d["fps"]) if "fps" in d else 30.0
    return X, V, fps, ~np.isnan(X[:, 0, 0])


def fill_missing(X):
    """검출 실패(NaN) 프레임을 시간축 선형 보간으로 채움."""
    X = X.copy()
    t = np.arange(len(X))
    bad = np.isnan(X[:, 0, 0])
    if bad.any() and (~bad).sum() >= 2:
        for j in range(X.shape[1]):
            for c in range(3):
                X[bad, j, c] = np.interp(t[bad], t[~bad], X[~bad, j, c])
    return X


# ------------------------------------------------------------------ 높이 보정
def height_basic(X):
    """[교수님 제공 방법 그대로] 프레임마다 두 발끝(31,32) 중 낮은 y를 지면으로 보고 빼기 + 0 아래는 clamp."""
    seq = X.copy()
    foot_y = np.minimum(seq[:, L_FOOT, 1], seq[:, R_FOOT, 1])
    seq[:, :, 1] -= foot_y[:, None]
    seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)
    return seq


def ground_curve(X, mode="lowest", k=2):
    """프레임별 지면 높이 추정. feet=발끝 최소(교수님), lowest=전체 관절 중 가장 낮은 k개의 평균."""
    y = X[:, :, 1]
    if mode == "feet":
        return np.minimum(y[:, L_FOOT], y[:, R_FOOT])
    return np.sort(y, axis=1)[:, :k].mean(axis=1)


def height_adjust(X, mode="lowest", k=2, sigma=0.0, clamp=False, envelope=False):
    """일반화한 높이 보정. sigma>0이면 지면 곡선을 시간축으로 부드럽게 한 뒤 뺌(발 떨림이 전신으로 번지는 것 방지).
    envelope=True: 스무딩한 지면이 실제 최저점보다 높아진 구간만큼 지면을 다시 낮춤(관통 방지, 대신 약간 뜸)."""
    seq = X.copy()
    g = ground_curve(seq, mode, k)
    if sigma > 0:
        gs = gaussian_filter1d(g, sigma, mode="nearest")
        if envelope:
            over = maximum_filter1d(np.maximum(gs - g, 0), size=int(4 * sigma) + 1, mode="nearest")
            gs = gs - gaussian_filter1d(over, sigma, mode="nearest")
        g = gs
    seq[:, :, 1] -= g[:, None]
    if clamp:
        seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)
    return seq


# ------------------------------------------------------------------ 스무딩
def smooth_ma(X, w=5):
    return uniform_filter1d(X, size=w, axis=0, mode="nearest")


def smooth_gauss(X, sigma=1.5):
    return gaussian_filter1d(X, sigma, axis=0, mode="nearest")


def smooth_savgol(X, w=9, p=3):
    w = min(w, (len(X) // 2) * 2 - 1)
    return savgol_filter(X, w, p, axis=0, mode="nearest")


def smooth_one_euro(X, fps, min_cutoff=1.5, beta=0.5, d_cutoff=1.0):
    """One-Euro filter: 느릴 때는 강하게, 빠를 때는 약하게 거르는 적응형 저역통과 필터(인과적, 한 방향)."""
    def alpha(cut):
        tau = 1.0 / (2 * np.pi * cut)
        return 1.0 / (1.0 + tau * fps)
    out = np.empty_like(X)
    out[0] = X[0]
    dx = np.zeros_like(X[0])
    for t in range(1, len(X)):
        d = (X[t] - out[t - 1]) * fps
        dx = dx + alpha(d_cutoff) * (d - dx)
        a = alpha(min_cutoff + beta * np.abs(dx))
        out[t] = out[t - 1] + a * (X[t] - out[t - 1])
    return out


def smooth_vis_adaptive(X, V, light, heavy, v0=0.5, v1=0.9):
    """Visibility 기반 혼합: 잘 보이는 관절은 light, 가려진 관절은 heavy 스무딩 결과를 사용."""
    w = np.clip((gaussian_filter1d(V, 2, axis=0, mode="nearest") - v0) / (v1 - v0), 0, 1)[..., None]
    return w * light + (1 - w) * heavy


# ------------------------------------------------------------------ 이상치
def spike_mask(X, k=4.0, floor=0.02):
    """한 프레임짜리 튐(spike) 검출.
    e(t) = x(t) - (x(t-1)+x(t+1))/2 : 앞뒤 프레임의 중간 지점에서 벗어난 정도.
    조건 1) |e| > max(k * 1.4826 * MAD(e), floor)  (그 관절·축의 평소 떨림보다 훨씬 큼)
    조건 2) |x(t+1) - x(t-1)| < |e|                (앞뒤 프레임끼리는 가까움 = 갔다가 바로 돌아옴)
    조건 2 덕분에 발차기처럼 한 방향으로 빠르게 움직이는 실제 동작은 이상치로 잡지 않는다."""
    e = np.zeros_like(X)
    e[1:-1] = X[1:-1] - 0.5 * (X[:-2] + X[2:])
    span = np.zeros_like(X)
    span[1:-1] = np.abs(X[2:] - X[:-2])
    mad = np.median(np.abs(e - np.median(e, axis=0)), axis=0)
    return (np.abs(e) > np.maximum(k * 1.4826 * mad, floor)) & (span < np.abs(e))


def remove_outliers(X, k=4.0, floor=0.02):
    """spike로 표시된 좌표를 지우고 앞뒤 정상 값으로 선형 보간. (보정된 배열, mask) 반환."""
    m = spike_mask(X, k, floor)
    out = X.copy()
    t = np.arange(len(X))
    for j, c in zip(*np.where(m.any(axis=0))):
        bad = m[:, j, c]
        if (~bad).sum() >= 2:
            out[bad, j, c] = np.interp(t[bad], t[~bad], X[~bad, j, c])
    return out, m


# ------------------------------------------------------------------ 뼈 길이
_CHAINS = [(23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28), (28, 30), (28, 32),
           (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22)]
_SYM = {(23, 25): (24, 26), (25, 27): (26, 28), (27, 29): (28, 30), (27, 31): (28, 32),
        (11, 13): (12, 14), (13, 15): (14, 16)}


def stabilize_bones(X, symmetric=True):
    """뼈 길이 고정: 부모->자식 방향은 유지하고 길이만 시퀀스 중앙값으로 맞춤(골반/어깨에서 말단 순서로 전파)."""
    out = X.copy()
    med = {b: np.median(np.linalg.norm(X[:, b[1]] - X[:, b[0]], axis=1)) for b in _CHAINS}
    if symmetric:
        for l, r in _SYM.items():
            med[l] = med[r] = 0.5 * (med[l] + med[r])
    for a, b in ((23, 24), (11, 12)):                 # 골반 폭, 어깨 폭: 중점은 그대로 두고 폭만 중앙값으로
        mid = 0.5 * (X[:, a] + X[:, b])
        d = X[:, a] - X[:, b]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        half = 0.5 * np.median(n) * d / np.maximum(n, 1e-8)
        out[:, a], out[:, b] = mid + half, mid - half
    for p, c in _CHAINS:
        d = X[:, c] - X[:, p]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        out[:, c] = out[:, p] + d / np.maximum(n, 1e-8) * med[(p, c)]
    return out


# ------------------------------------------------------------------ 평가 지표
def still_mask(X_raw, fps, thr=0.15):
    """정지 구간: 강하게 스무딩한 raw의 평균 관절 속도가 thr(m/s) 미만인 프레임. 모든 방법에 같은 mask를 씀."""
    s = gaussian_filter1d(X_raw[:, BODY], 3, axis=0, mode="nearest")
    v = np.linalg.norm(np.gradient(s, axis=0), axis=2).mean(axis=1) * fps
    return v < thr


def metrics(X, fps, X_raw=None, still=None):
    """한 시퀀스의 품질 지표. 단위는 키 이름에 표기."""
    B = X[:, BODY]
    vel = np.diff(B, axis=0) * fps
    acc = np.diff(B, 2, axis=0) * fps ** 2
    jerk = np.diff(B, 3, axis=0) * fps ** 3
    m = {"acc_m_s2": np.linalg.norm(acc, axis=2).mean(),
         "jerk_m_s3": np.linalg.norm(jerk, axis=2).mean()}
    if still is not None and still[1:].sum() > 3:
        m["still_speed_cm_s"] = np.linalg.norm(vel, axis=2)[still[1:]].mean() * 100
    L = np.array([np.linalg.norm(X[:, a] - X[:, b], axis=1) for a, b in BONES])
    m["bone_cv_pct"] = (L.std(axis=1) / L.mean(axis=1)).mean() * 100
    y = X[:, :, 1]                                            # 지면 지표는 33개 관절 전체로 계산
    m["penetration_pct"] = (y < -0.01).mean() * 100          # 지면(y=0)보다 1cm 넘게 아래인 (프레임,관절) 비율
    m["max_depth_cm"] = max(0.0, -y.min()) * 100
    low = y.min(axis=1)
    m["floating_pct"] = (low > 0.02).mean() * 100             # 가장 낮은 관절이 2cm 넘게 떠 있는 프레임 비율
    m["lowest_mean_cm"] = low.mean() * 100
    m["n_spikes"] = int(spike_mask(X)[:, BODY].any(axis=2).sum())   # 남아 있는 한 프레임 튐 개수(관절 단위)
    m["root_y_jitter_m_s2"] = np.abs(np.diff(X[:, [23, 24], 1].mean(axis=1), 2)).mean() * fps ** 2
    if X_raw is not None:
        rr = lambda A: A[:, BODY] - A[:, [23, 24]].mean(axis=1, keepdims=True)   # 골반 기준 좌표(높이 보정 영향 제거)
        a, b = rr(X), rr(X_raw)
        m["dev_rmse_cm"] = np.sqrt(((a - b) ** 2).sum(axis=2).mean()) * 100
        rom_a, rom_b = np.ptp(a, axis=0), np.ptp(b, axis=0)
        big = rom_b > 0.10
        m["rom_keep_pct"] = (rom_a[big] / rom_b[big]).mean() * 100
        sp = lambda A: np.linalg.norm(A[4:] - A[:-4], axis=2) * fps / 4       # 4프레임 간격 속도(노이즈 영향 완화)
        j = sp(b).max(axis=0).argmax()                                         # raw에서 가장 빠른 관절
        m["peak_speed_keep_pct"] = sp(a)[:, j].max() / sp(b)[:, j].max() * 100
    return m
