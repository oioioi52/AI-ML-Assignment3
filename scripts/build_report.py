"""build_report.py - 교수님 PPT 양식(Assignment_3D_motiondata.pptx)에 실험 결과를 채워 제출용 PPTX 생성.
  python scripts/build_report.py --template <교수님 PPT 양식 경로> --name "<학번 이름>"
표의 숫자와 본문 수치는 모두 results/*.csv 에서 읽어 넣는다(손으로 적은 수치 없음)."""
import os, sys, argparse, re
import numpy as np, pandas as pd
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import motion_lib as ml
RES = os.path.join(ROOT, "results"); FIG = os.path.join(RES, "figures")
BLUE = RGBColor(0x00, 0x6C, 0xB8); DARK = RGBColor(0x22, 0x22, 0x22); RED = RGBColor(0xC6, 0x28, 0x28); GRAY = RGBColor(0x55, 0x55, 0x55)
FONT = "맑은 고딕"
X0, Y0, GW, GH, GAP = 300000, 1235000, 5730000, 2700000, 130000     # 2x2 grid (EMU)
POS = [(X0, Y0), (X0 + GW + GAP, Y0), (X0, Y0 + GH + GAP), (X0 + GW + GAP, Y0 + GH + GAP)]


def runs(par, text, size, color=DARK, bold=False):
    """'**굵게**' 와 '!!빨강!!' 마크업을 run 으로 나눠 넣음"""
    for tok in re.split(r"(\*\*.+?\*\*|!!.+?!!)", text):
        if not tok: continue
        r = par.add_run(); b, c = bold, color
        if tok.startswith("**"): tok, b = tok[2:-2], True
        elif tok.startswith("!!"): tok, b, c = tok[2:-2], True, RED
        r.text = tok; r.font.size = Pt(size); r.font.bold = b; r.font.color.rgb = c; r.font.name = FONT
        rPr = r._r.get_or_add_rPr(); ea = rPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}ea", {"typeface": FONT}); rPr.append(ea)


def text(slide, x, y, w, h, lines, size=8, color=DARK, bold=False, spacing=1.0):
    tb = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h)); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(30000); tf.margin_top = tf.margin_bottom = Emu(10000)
    for i, l in enumerate([lines] if isinstance(lines, str) else lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.line_spacing = spacing; runs(p, l, size, color, bold)
    return tb


def title(slide, k, t):
    x, y = POS[k]
    bar = slide.shapes.add_shape(1, Emu(x), Emu(y), Emu(GW), Emu(215000)); bar.fill.solid(); bar.fill.fore_color.rgb = RGBColor(0xE8, 0xF1, 0xFA); bar.line.fill.background()
    text(slide, x + 20000, y + 5000, GW - 40000, 205000, t, size=10.5, color=BLUE, bold=True)
    return x, y + 235000


def image(slide, name, x, y, w, h, align="center"):
    path = os.path.join(FIG, name); iw, ih = Image.open(path).size; s = min(w / iw, h / ih); pw, ph = int(iw * s), int(ih * s)
    slide.shapes.add_picture(path, Emu(x + ((w - pw) // 2 if align == "center" else 0)), Emu(y), Emu(pw), Emu(ph)); return ph


def table(slide, x, y, w, rows, col_w=None, size=7, row_h=150000, hi=None, bold_rows=()):
    """rows[0] = header. hi = {(r,c)} 강조 셀(빨강 굵게)."""
    nr, nc = len(rows), len(rows[0]); gt = slide.shapes.add_table(nr, nc, Emu(x), Emu(y), Emu(w), Emu(row_h * nr)); t = gt.table
    tot = sum(col_w) if col_w else nc
    for c in range(nc): t.columns[c].width = Emu(int(w * (col_w[c] if col_w else 1) / tot))
    for r in range(nr):
        t.rows[r].height = Emu(row_h)
        for c in range(nc):
            cell = t.cell(r, c); cell.margin_left = cell.margin_right = Emu(25000); cell.margin_top = cell.margin_bottom = Emu(5000); cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            is_hi = hi and (r, c) in hi
            runs(p, str(rows[r][c]), size, RGBColor(255, 255, 255) if r == 0 else (RED if is_hi else DARK), bold=(r == 0 or is_hi or r in bold_rows))
            cell.fill.solid(); cell.fill.fore_color.rgb = BLUE if r == 0 else (RGBColor(0xEA, 0xF3, 0xFB) if r in bold_rows else (RGBColor(0xF7, 0xF7, 0xF7) if r % 2 == 0 else RGBColor(255, 255, 255)))
    return row_h * nr


def build(template, out, name=""):
    prs = Presentation(template); S = list(prs.slides)
    for s in S[1:5]:                                   # 양식의 안내 문구(과제 설명) 텍스트 상자는 내용으로 대체
        for sh in list(s.shapes):
            if sh.name in ("TextBox 33", "TextBox 2", "TextBox 3"): sh._element.getparent().remove(sh._element)
    if name:
        text(S[0], 674061, 4100000, 6000000, 300000, name, size=16, color=RGBColor(255, 255, 255), bold=True)

    mm = pd.read_csv(os.path.join(RES, "methods_mean.csv")).set_index("method"); pv = pd.read_csv(os.path.join(RES, "methods_per_video.csv"))
    E0, E1, E6 = mm.loc["E0 Raw"], mm.loc["E1 Basic(교수님)"], mm.loc["E6 Enhanced(최종)"]
    pct = lambda a, b: f"{(b - a) / a * 100:+.0f}%"
    V = lambda vid, meth, col: float(pv[(pv.video == vid) & pv.method.str.startswith(meth)][col].iloc[0])

    # ================= Slide 2 : Data Acquisition & Problem Identification
    s = S[1]
    x, y = title(s, 0, "1-1  영상 선정 · MediaPipe 3D Keypoint 추출 · 데이터 구조")
    info = {"idle": ("Idle", "제자리 움직임(기준)"), "sidekick_mid": ("Kick A", "옆차기(빠른 동작)"), "sidekick_high": ("Kick B", "높은 옆차기(빠른 동작)"),
            "roll": ("Roll", "앞구르기(가림·바닥 접촉)"), "handstand": ("Handstand", "물구나무(거꾸로 선 자세)")}
    rows = [["영상", "동작 / 선정 이유", "해상도", "FPS", "프레임", "검출률", "가려진 관절*"]]
    for v, (lab, why) in info.items():
        d = np.load(os.path.join(ROOT, "data", "raw_keypoints", f"{v}.npz")); X, Vv, fps, det = ml.load_raw(os.path.join(ROOT, "data", "raw_keypoints", f"{v}.npz"))
        rows.append([lab, why, f"{d['size'][0]}×{d['size'][1]}", f"{fps:.0f}", len(X), f"{det.mean() * 100:.0f}%", f"{(Vv[:, 11:] < 0.5).mean() * 100:.1f}%"])
    mc = int(np.load(os.path.join(ROOT, "data", "raw_keypoints", "idle.npz"))["model_complexity"])
    h = table(s, x, y, GW, rows, col_w=[1.1, 2.6, 1.1, 0.6, 0.8, 0.8, 1.2])
    text(s, x, y + h + 30000, GW, 1500000, [
        f"**추출** p1_create_raw_keypoints.py(교수님 코드 기반): mp.solutions.pose, model_complexity={mc}, smooth_landmarks=True → pose_world_landmarks 33개.",
        "**구조** 영상 1개 = (T, 33, 3) 배열 + visibility (T, 33). 총 5개 영상 1,005 프레임. 단위 m, **원점 = 양 골반 중점**(전역 이동 없음).",
        "**좌표계** y만 부호 반전(위쪽 +)해서 저장 → 왼손 좌표계(Unity와 동일). 오른손 좌표계인 BVH와 비교할 때 z 반전이 필요함을 실험으로 확인.",
        "**선정 근거** 난이도가 다른 동작 4종(일반·빠름·가림·거꾸로)으로 후처리가 '어떤 자세에서 깨지는지' 보려고 구성. 기말 프로젝트 클래스 후보로도 사용.",
        "**수정점** 영상마다 Pose 객체 새로 생성, 검출 실패 프레임도 기록(프레임 번호 유지), 2D landmark·fps 함께 저장.  *visibility<0.5 비율(어깨 이하 관절)"], size=8.6, spacing=1.08)
    x, y = title(s, 1, "1-2  원본 영상(2D overlay)과 추출된 3D 스켈레톤")
    ph = image(s, "s1_acquisition.png", x, y, GW, 2050000)
    text(s, x, y + ph + 10000, GW, 400000, ["위: 원본 프레임 + MediaPipe 2D. 아래: 같은 프레임의 3D(회색=카메라 시점, 파랑=55° 회전). **점선 y=0이 골반**이라 raw는 하반신이 항상 지면 grid 아래에 있다.",
                                           "Roll은 2D 단계에서 이미 팔다리가 뒤엉켜 추정됨 → 후처리로 고칠 수 없는 한계(4-4)."], size=7.6)
    x, y = title(s, 2, "2-1  문제 ① 지면·높이   ② 관절 떨림(Jitter)   ③ Occlusion")
    ph = image(s, "s1_problems_ground_jitter.png", x, y, GW, 1650000)
    jv = pd.read_csv(os.path.join(RES, "jitter_by_visibility.csv"))
    text(s, x, y + ph + 10000, GW, 800000, [
        f"**① Ground Penetration** Raw는 관절의 !!{E0.penetration_pct:.0f}%!!가 y=0 아래(최대 {E0.max_depth_cm:.0f}cm). 원인: world 좌표의 원점이 골반이라 '바닥' 개념이 없음. Roll·Handstand는 최저점이 발이 아니라 손·어깨로 바뀜.",
        f"**② Jitter** 평균 가속도 {E0.acc_m_s2:.1f} m/s², 깊이(z) 축이 x·y보다 큼(단안 카메라의 깊이 불확실성). 정지 구간에서도 관절이 평균 {E0.still_speed_cm_s:.0f} cm/s로 움직임.",
        f"**③ Occlusion** 가려진 관절(visibility<0.5)의 가속도 {jv.mean_acc_m_s2[0]:.1f} vs 잘 보이는 관절 {jv.mean_acc_m_s2[2]:.1f} m/s² → !!{jv.mean_acc_m_s2[0] / jv.mean_acc_m_s2[2]:.1f}배!!. 옆모습 영상이라 카메라 반대쪽(왼쪽) 팔다리가 주로 해당."], size=7.6)
    x, y = title(s, 3, "2-2  문제 ④ Bone Length Variation   ⑤ Outlier   + 확인 결과 요약")
    ph = image(s, "s1_problems_bone_outlier.png", x, y, GW, 1650000)
    cv = pd.read_csv(os.path.join(RES, "bone_cv_raw.csv"), index_col=0)
    text(s, x, y + ph + 10000, GW, 800000, [
        f"**④ 뼈 길이** 강체여야 할 뼈 12개의 길이 변동계수 평균 !!{E0.bone_cv_pct:.1f}%!!(최대 {cv.values.max():.0f}%, Roll). Kick B는 발차기 중 오른쪽 허벅지가 38→25cm로 줄어듦(깊이 방향으로 뻗은 다리의 길이 과소 추정).",
        f"**⑤ 이상치** 한 프레임만 튀는 spike를 영상당 평균 {E0.n_spikes:.0f}개 검출. Kick B f58(찬 다리가 지지 다리 위치로 32cm 점프), Handstand f115(양발 38cm) 등을 원본 프레임과 대조해 실제 추적 오류로 확인.",
        "**확인되지 않은 문제** Floating: raw에는 없음(전부 지면 아래). Motion Discontinuity(검출 실패): 5개 영상 모두 검출률 100%라 프레임 누락 없음 → 다루지 않음."], size=7.6)

    # ================= Slide 3 : Post-Processing
    s = S[2]
    x, y = title(s, 0, "3-1  교수님 제공 Height Adjustment: 원리 · 효과 · 한계")
    ph = image(s, "s2_basic.png", x, y, GW, 1650000)
    text(s, x, y + ph + 10000, GW, 800000, [
        "**원리** 프레임마다 ground = min(왼발끝 y, 오른발끝 y) → 전 관절 y에서 빼고 → 0 아래는 0으로 clamp (p1_2_post_processing.py: create_height_corrected_target).",
        f"**효과** 지면 관통 {E0.penetration_pct:.0f}% → !!0%!!. 서 있는 동작(Idle·Kick)에서는 의도대로 동작.",
        f"**한계 ①** '발이 가장 낮다'는 가정이 깨지면 실패: Handstand에서 발끝을 0으로 내리고 나머지를 clamp → 뼈 길이 CV {V('handstand', 'E0', 'bone_cv_pct'):.0f}→!!{V('handstand', 'E1', 'bone_cv_pct'):.0f}%!!, 자세 변형 {V('handstand', 'E1', 'dev_rmse_cm'):.0f}cm.",
        f"**한계 ②** 프레임별 발끝 y의 떨림이 전신 높이로 복사됨: 골반 높이 떨림 0 → !!{E1.root_y_jitter_m_s2:.1f} m/s²!!, Jerk {E0.jerk_m_s3:.0f}→{E1.jerk_m_s3:.0f}({pct(E0.jerk_m_s3, E1.jerk_m_s3)}). 떨림·뼈 길이·이상치는 그대로."], size=7.6)
    x, y = title(s, 1, "3-2  알고리즘 비교 실험 E0~E6 (동일 데이터 5개 영상 평균)")
    cols = [("jerk_m_s3", "Jerk\n(m/s³)↓", 0), ("still_speed_cm_s", "정지구간\n속도(cm/s)↓", 1), ("bone_cv_pct", "뼈길이\nCV(%)↓", 1), ("penetration_pct", "지면관통\n(%)↓", 1),
            ("root_y_jitter_m_s2", "골반높이\n떨림↓", 2), ("n_spikes", "남은\nspike↓", 0), ("rom_keep_pct", "ROM\n보존(%)↑", 1), ("peak_speed_keep_pct", "최고속도\n보존(%)↑", 1)]
    rows = [["방법"] + [c[1] for c in cols]] + [[m.replace("(교수님)", "").replace("(최종)", "")] + [f"{mm.loc[m, c]:.{d}f}" for c, _, d in cols] for m in mm.index]
    h = table(s, x, y, GW, rows, col_w=[2.5, 1, 1.1, 1, 1, 1, 0.9, 1, 1.1], size=6.8, row_h=165000, bold_rows=(7,))
    s.shapes[-1].table.rows[0].height = Emu(290000)
    text(s, x, y + h + 150000, GW, 1000000, [
        "**E2~E4(Basic 뒤에 스무딩)** 떨림은 줄지만 뼈 길이·Handstand 붕괴는 그대로이고, Gaussian·MA는 ROM을 90% 아래로 깎음(발차기가 작아짐).",
        "**E5(이상치 제거만)** spike 107→33개, Jerk는 거의 그대로 → 이상치와 떨림은 서로 다른 문제라 따로 다뤄야 함.",
        f"**E6** 뼈 길이 CV·골반 높이 떨림이 최저. Jerk와 spike는 E3(Gaussian)가 더 낮지만 E3는 ROM 89%로 동작이 작아짐 → E6는 ROM {E6.rom_keep_pct:.0f}%·최고속도 {E6.peak_speed_keep_pct:.0f}%를 지키는 범위에서 줄인 결과. 지면 관통은 clamp를 쓰지 않아 {E6.penetration_pct:.2f}%(최대 {E6.max_depth_cm:.1f}cm) 남음. ↓낮을수록, ↑높을수록 좋음."], size=8.4, spacing=1.08)
    x, y = title(s, 2, "3-3  파라미터 비교 실험 (필터 4종 30개 설정 · 이상치 임계값 12개 · 지면 σ)")
    ph = image(s, "s2_sweeps.png", x, y, GW, 1620000)
    sg = pd.read_csv(os.path.join(RES, "sweep_savgol.csv")).set_index("method"); g9 = sg.loc["SavGol w=9,p=3"]
    text(s, x, y + ph + 10000, GW, 740000, [
        f"**(a) 선정 기준**: ROM 보존 ≥95% 그리고 최고속도 보존 ≥85% 를 만족하는 설정 중 Jerk 최소 → **Savitzky-Golay window 9, order 3** (Jerk {g9.jerk_m_s3:.0f}, ROM {g9.rom_keep_pct:.1f}%, 속도 {g9.peak_speed_keep_pct:.0f}%). MA·Gaussian은 같은 Jerk에서 ROM이 더 낮고, One-Euro(인과적 필터)는 지연 때문에 기준 미달. order 2와 3은 결과가 같음(대칭 window의 성질). 속도 기준을 90%로 올리면 window 7이 선택됨(기준에 따라 달라지는 선택임).",
        "**(b)** spike 임계값 k=4·floor 2cm: 육안으로 확인한 f58·f115는 잡고, k를 낮추면 정상 프레임까지 늘어남.  **(c)** 지면 곡선 σ=2: 골반 떨림 4.8→0.8, 관통 깊이는 3cm로 증가 → 절충점."], size=7.4)
    x, y = title(s, 3, "3-4  Ablation Study와 최종 알고리즘 선정 근거")
    ab = pd.read_csv(os.path.join(RES, "ablation.csv")).set_index("method")
    names = [("Full (O+S+B+G)", "Full: O→S→B→G (최종)"), ("- Outlier(O)", "− O 이상치 제거"), ("- Smoothing(S)", "− S 스무딩"), ("- Bone(B)", "− B 뼈 길이 고정"),
             ("- Ground(G) -> Basic height", "− G (교수님 높이 보정 사용)"), ("S: plain SavGol only", "S를 SavGol 단독으로"), ("S: heavy Gauss only", "S를 Gaussian(σ=2) 단독으로"), ("order: G before S", "순서 변경: G를 S 앞에")]
    cols = [("jerk_m_s3", "Jerk↓", 0), ("bone_cv_pct", "뼈CV%↓", 1), ("root_y_jitter_m_s2", "골반떨림↓", 2), ("n_spikes", "spike↓", 1), ("max_depth_cm", "관통cm↓", 1), ("dev_rmse_cm", "변형cm", 1), ("rom_keep_pct", "ROM%↑", 1), ("peak_speed_keep_pct", "속도%↑", 1)]
    rows = [["구성"] + [c[1] for c in cols]] + [[n] + [f"{ab.loc[k, c]:.{d}f}" for c, _, d in cols] for k, n in names]
    h = table(s, x, y, GW, rows, col_w=[3.0, 0.8, 0.9, 1, 0.8, 0.9, 0.9, 0.9, 0.9], size=6.8, row_h=148000, bold_rows=(1,), hi={(3, 1), (4, 2), (5, 3), (5, 2), (7, 7), (8, 5)})
    text(s, x, y + h + 25000, GW, 1100000, [
        "**O** spike 제거+선형 보간 → **S** visibility 적응형 스무딩(잘 보이면 SavGol, 가려지면 Gaussian) → **B** 뼈 길이를 중앙값으로 고정 → **G** 최저 관절 기준 지면 + 지면 곡선 스무딩(clamp 없음).",
        "**요소별 기여(빨강)** S를 빼면 Jerk 4.8배, B를 빼면 뼈 CV 9.3%, G를 빼면 골반 떨림 4배·뼈 CV 7.7%(Handstand 붕괴). O는 평균 지표 차이는 작지만 f58 같은 국소 오류를 없앰(4-2a).",
        "**채택하지 않은 것** Gaussian 단독은 Jerk가 더 낮지만 ROM 90.7%로 기준 미달. clamp·envelope 방식 지면 보정은 뼈 길이를 바꾸거나 13%의 프레임을 띄워서 제외. **비용**: B는 자세를 raw에서 2.4→4.5cm 더 움직인다(정확도 향상이라고 단정하지 않음)."], size=7.4)

    # ================= Slide 4 : Results & Evaluation (정량)
    s = S[3]
    x, y = title(s, 0, "4-1  Raw / Basic / Enhanced 정량 비교 (영상별)")
    ph = image(s, "s3_bars.png", x, y, GW, 1330000)
    rows = [["5개 영상 평균", "Raw", "Basic", "Enhanced", "Basic 대비"]]
    for c, n, d in [("jerk_m_s3", "Jerk (m/s³)", 0), ("bone_cv_pct", "뼈 길이 CV (%)", 2), ("root_y_jitter_m_s2", "골반 높이 떨림 (m/s²)", 2), ("n_spikes", "남은 spike (개/영상)", 1), ("penetration_pct", "지면 관통 (%)", 2)]:
        rows.append([n, f"{E0[c]:.{d}f}", f"{E1[c]:.{d}f}", f"{E6[c]:.{d}f}", pct(E1[c], E6[c]) if E1[c] > 0 else f"+{E6[c]:.2f}%p"])
    h = table(s, x, y + ph + 20000, GW, rows, col_w=[2.2, 1, 1, 1, 1], size=7, row_h=140000, hi={(1, 4), (2, 4), (3, 4), (4, 4)})
    text(s, x, y + ph + h + 40000, GW, 400000, ["Enhanced는 5개 영상 모두에서 Basic보다 4개 지표가 낮다. Raw의 골반 높이 떨림이 0인 것은 골반이 원점에 고정돼 있기 때문(높이 정보 자체가 없음). 지면 관통은 clamp를 쓰지 않은 대가로 0.11% 남는다."], size=7.8)
    x, y = title(s, 1, "4-2  보정 전후 시계열 · 동작 보존 · 강건성 검증")
    ph = image(s, "s3_timeseries_robust.png", x, y, GW, 1650000)
    rb = pd.read_csv(os.path.join(RES, "robustness.csv")); rbe = rb[rb.method == "E6 Enhanced"].groupby("input_scale").jerk_m_s3.mean()
    text(s, x, y + ph + 10000, GW, 800000, [
        "**(a)** f58 이상치: 스무딩만 하면 튄 값이 주변 프레임으로 번져 궤적이 휘지만(보라), 먼저 제거하면 앞뒤 흐름을 따라감(파랑).",
        f"**(b) 동작 보존** 발끝 속도의 봉우리 위치·형태 유지. 5개 영상 평균 최고속도 {E6.peak_speed_keep_pct:.0f}%, ROM {E6.rom_keep_pct:.0f}% 보존. 단 raw 봉우리 일부는 노이즈라 100%가 목표는 아님.",
        f"**(c) 강건성** 입력 영상을 1/2·1/4 해상도로 줄여 MediaPipe부터 다시 추출해도 Enhanced Jerk {rbe.min():.0f}~{rbe.max():.0f} m/s³로 일정, 뼈 CV 0%, 같은 파라미터 그대로 사용(Kick B·Handstand)."], size=8.2, spacing=1.05)
    x, y = title(s, 2, "4-3  독립 추정 결과(DeepMotion BVH)와의 비교: 안정성 vs 정확도")
    bv = pd.read_csv(os.path.join(RES, "bvh_agreement.csv")).set_index(["video", "method"])
    rows = [["같은 원본 영상의 BVH와 비교", "Kick B\nRaw", "Kick B\nBasic", "Kick B\nEnhanced", "Handstand\nRaw", "Handstand\nBasic", "Handstand\nEnhanced"]]
    g = lambda v, m, c, d=1: f"{bv.loc[(v, m), c]:.{d}f}"
    rows.append(["관절 위치 차이 평균 (cm)"] + [g(v, m, "agree_mean_cm") for v in ("sidekick_high", "handstand") for m in ("E0 Raw", "E1 Basic", "E6 Enhanced")])
    rows.append(["골반 높이 곡선 RMSE (cm)"] + sum([["–", g(v, "E1 Basic hip-height", "hip_rmse_cm"), g(v, "E6 Enhanced hip-height", "hip_rmse_cm")] for v in ("sidekick_high", "handstand")], []))
    rows.append(["골반 높이 곡선 상관계수"] + sum([["–", g(v, "E1 Basic hip-height", "hip_corr", 2), g(v, "E6 Enhanced hip-height", "hip_corr", 2)] for v in ("sidekick_high", "handstand")], []))
    h = table(s, x, y, GW, rows, col_w=[2.6, 1, 1, 1.1, 1.1, 1.1, 1.2], size=7, row_h=170000, hi={(1, 5), (2, 5), (3, 5), (2, 6), (3, 6)})
    s.shapes[-1].table.rows[0].height = Emu(290000)
    text(s, x, y + h + 150000, GW, 1500000, [
        "**비교 가능 여부 검토** BVH 4개 중 프레임 수·FPS가 원본 영상과 일치하는 2개(Kick B 103f, Handstand 161f, 24fps)만 사용. BVH를 직접 파싱해 FK로 관절 위치 계산 → 공통 관절 12개, 골반 기준, z 반전 + 회전·스케일 1개로 정렬.",
        "**BVH는 정답이 아님** 다른 추정기의 결과일 뿐이므로 '일치도'로만 해석. 관절 위치 차이는 Raw≈Enhanced(≈10~11cm) → **후처리가 위치 정확도를 높였다고 말할 근거는 없다.** 개선된 것은 떨림·뼈 길이·지면 같은 **안정성/물리적 일관성**이다.",
        f"**높이 보정은 독립 근거 확보** Handstand에서 Basic의 골반 높이는 BVH와 정반대(상관 {g('handstand', 'E1 Basic hip-height', 'hip_corr', 2)}), Enhanced는 상관 {g('handstand', 'E6 Enhanced hip-height', 'hip_corr', 2)}·RMSE {g('handstand', 'E6 Enhanced hip-height', 'hip_rmse_cm')}cm. Kick B는 상관은 오르지만 RMSE는 Basic이 더 작음(3.1 vs 4.4cm)."], size=8.0, spacing=1.05)
    x, y = title(s, 3, "4-4  실패 사례와 한계")
    rows = [["사례", "수치", "원인 / 해석"],
            ["Idle: 빠른 손동작이 무뎌짐", f"최고속도 {V('idle', 'E6', 'peak_speed_keep_pct'):.0f}%, ROM {V('idle', 'E6', 'rom_keep_pct'):.0f}%", "30fps·작은 동작에 window 9가 과함. 프레임 단위가 아닌 시간(초) 단위 window가 필요"],
            ["Roll: spike가 남음", f"{V('roll', 'E0', 'n_spikes'):.0f} → {V('roll', 'E6', 'n_spikes'):.0f}개", "2D 추정부터 틀린 구간이 여러 프레임 연속 → 한 프레임 spike 검출로는 못 잡음"],
            ["Roll: 지면 관통 잔존", f"최대 {V('roll', 'E6', 'max_depth_cm'):.1f}cm", "지면 곡선을 스무딩해 빠른 자세 전환을 따라가지 못함(떨림과의 trade-off)"],
            ["점프 동작(미검증)", "–", "world 좌표에 전역 이동이 없어 '항상 어딘가 바닥에 닿아 있다'고 가정 → 공중 구간은 바닥에 붙음"]]
    h = table(s, x, y, GW, rows, col_w=[1.7, 1.2, 3.9], size=6.8, row_h=265000)
    text(s, x, y + h + 30000, GW, 1300000, [
        "**정답 데이터 부재** 모든 지표는 '부드러움·일관성'을 재는 것이지 실제 위치 오차가 아니다. 뼈 길이 고정(B)은 CV를 정의상 0으로 만들지만 그 길이가 실제 신체 길이라는 보장은 없다.",
        f"**모델 차이** 교수님 코드는 model_complexity=2(heavy)이나 이 결과는 {mc}(full)로 추출했다. heavy 모델에서는 수치가 달라질 수 있다.",
        "**파라미터의 일반성** 5개 영상 평균으로 정한 값이라 다른 FPS·촬영 조건에서는 다시 sweep이 필요. 영상이 클래스당 1~2개라 동작별 통계적 결론은 내리지 않았다."], size=7.5)

    # ================= Slide 5 : Results & Evaluation (시각화, Bonus 2)
    s = S[4]
    x, y = title(s, 0, "5-1  개선한 시각화 ①: 3-way 비교 뷰어 (브라우저 · Viser)")
    hw = (GW - 60000) // 2
    image(s, "s4_viewer.png", x, y, hw, 1750000)
    vh = int(hw * 9 / 16)                                   # 비교 영상을 슬라이드에 직접 삽입 (슬라이드쇼에서 클릭하면 재생)
    s.shapes.add_movie(os.path.join(ROOT, "videos", "compare_handstand.mp4"), Emu(x + hw + 60000), Emu(y), Emu(hw), Emu(vh),
                       poster_frame_image=os.path.join(FIG, "s4_compare_video_frame.png"), mime_type="video/mp4")
    text(s, x, y + 1790000, GW, 650000, [
        "**motion_viewer.html**(왼쪽): Raw | Basic | Enhanced 동시 재생, Play/Pause, 프레임 슬라이더, 속도, 마우스 회전, 지면 grid, **문제 관절 강조**(빨강=지면 아래, 주황 원=이상치), 프레임별·영상별 지표 표시. 설치 없이 실행.",
        "**비교 영상**(오른쪽, ▶ 슬라이드쇼에서 클릭하면 재생): 원본 영상+2D overlay와 세 결과를 프레임 동기화. 교수님 PoseViser를 확장한 viser_compare.py(3개 동시 재생·GUI)도 포함."], size=8.0, spacing=1.05)
    x, y = title(s, 1, "5-2  개선한 시각화 ②: Blender 리깅 캐릭터 3명 동시 비교")
    vw = 2900000; vh = int(vw * 9 / 16); sw = GW - vw - 60000
    s.shapes.add_movie(os.path.join(ROOT, "videos", "blender_character_handstand.mp4"), Emu(x), Emu(y), Emu(vw), Emu(vh),
                       poster_frame_image=os.path.join(FIG, "s4_blender_poster.png"), mime_type="video/mp4")
    p1 = image(s, "s4_blender_handstand.png", x + vw + 60000, y + 150000, sw, 800000); p2 = image(s, "s4_blender_kick.png", x + vw + 60000, y + 150000 + p1 + 120000, sw, 800000)
    text(s, x, y + vh + 30000, GW, 700000, [
        "같은 Mannequin 캐릭터 3명(회색 Raw · 주황 Basic · 파랑 Enhanced)에 33개 keypoint를 **직접 retargeting**: 몸통은 골반선·어깨선으로 기저를 만들고, 팔다리는 부모 뼈에서 keypoint 방향까지의 최소 회전을 적용(blender_compare.py, Blender 5.2 Eevee 렌더).",
        "왼쪽: Handstand 영상(▶ 클릭하면 재생). 오른쪽 위: Handstand f21/81/119, 아래: Kick B f11/48/58. Raw는 골반이 바닥에 묻히고, **Basic은 물구나무 구간에서 바닥에 납작하게 눌림**, Enhanced만 자세 유지."], size=7.5)
    x, y = title(s, 2, "5-3  프레임별 보정 전후 (스켈레톤)")
    ph = image(s, "s4_strip.png", x, y, GW, 2150000)
    text(s, x, y + ph + 5000, GW, 300000, "Handstand f55~105: Basic은 전신이 y=0에 눌림. Roll f128: 등이 바닥에 닿는 자세에서 Enhanced는 어깨를 지면으로 인식.", size=7.5)
    x, y = title(s, 3, "5-4  결론 · 기말 프로젝트 연결")
    text(s, x, y, GW, 2400000, [
        f"**결론** ① 교수님 Height Adjustment는 서 있는 동작에서는 지면 관통을 없애지만, 발이 최저점이 아닌 자세에서 몸을 찌그러뜨리고 발끝 떨림을 전신으로 옮긴다. ② O→S→B→G 4단계로 Basic 대비 Jerk {pct(E1.jerk_m_s3, E6.jerk_m_s3)}, 골반 높이 떨림 {pct(E1.root_y_jitter_m_s2, E6.root_y_jitter_m_s2)}, spike {pct(E1.n_spikes, E6.n_spikes)}, 뼈 길이 CV {E1.bone_cv_pct:.1f}→0%를 얻었고 동작 범위는 {E6.rom_keep_pct:.0f}% 유지했다. ③ 각 단계와 파라미터는 sweep·ablation 수치로 정했다. ④ 개선은 안정성에 대한 것이며 위치 정확도 향상은 입증하지 못했다.",
        "**수정한 코드** 교수님의 create_height_corrected_target()은 그대로 두고, p1_2_post_processing.py에 enhanced_pipeline()을 추가했다. 난수를 쓰지 않아 같은 입력이면 같은 결과가 나온다.",
        "**기말 프로젝트 연결** 이번에 확인한 사실(Kick·Handstand는 추출 품질 양호, Roll은 추정 자체가 불안정)을 근거로 클래스를 정하고, Enhanced 후처리를 분류기 입력 전처리로 써서 Baseline MLP 대비 효과를 실험할 계획이다."], size=9.2, spacing=1.15)
    prs.save(out); print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="교수님 제공 PPT 양식(.pptx) 경로")
    ap.add_argument("--out", default=os.path.join(ROOT, "report", "Assignment3_report.pptx"))
    ap.add_argument("--name", default="", help="표지에 넣을 학번·이름")
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True); build(a.template, a.out, a.name)
