"""blender_compare.py - Raw / Basic / Enhanced 를 같은 리깅 캐릭터 3명에게 동시에 적용해 비교하는 Blender 스크립트 (Bonus 2)

실행 (Blender 4.2+ / 5.x):
  blender --background --python visualization/blender_compare.py -- --video handstand --render
  (--render 를 빼면 .blend 만 저장. 저장된 visualization/<video>_compare.blend 를 열어 Space 로 재생 가능)

원리 (33개 keypoint -> 리깅 캐릭터 retargeting)
  1) 좌표 변환: 데이터 (x, y위, z깊이) -> Blender (x, z깊이, y위). 축 2개를 맞바꿔 왼손 좌표계를 오른손(Blender)으로.
  2) 몸통: 골반 좌우선 + 척추 방향으로 기저(basis)를 만들어 hips 회전, 어깨 좌우선으로 가슴 회전, 귀/코로 머리 회전.
  3) 팔다리: 부모 뼈의 현재 방향에서 keypoint 방향(예: 무릎->발목)까지 가장 작은 회전(swing)만 추가.
  4) hips 위치 = 골반 중점 keypoint (캐릭터 다리 길이에 맞춰 스케일).
  캐릭터: DeepMotion 이 제공한 Mannequin FBX (visualization/assets/character_tpose.fbx)
"""
import bpy, sys, os, math
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
VIDEO = argv[argv.index("--video") + 1] if "--video" in argv else "handstand"
RENDER = "--render" in argv
ENGINE = argv[argv.index("--engine") + 1] if "--engine" in argv else "EEVEE"
RES = [int(v) for v in argv[argv.index("--res") + 1].split("x")] if "--res" in argv else [1280, 720]
FRAMES = [int(v) for v in argv[argv.index("--frames") + 1].split(",")] if "--frames" in argv else None
FPS = {"handstand": 24, "sidekick_high": 24, "sidekick_mid": 25, "roll": 25, "idle": 30}.get(VIDEO, 24)
VARIANTS = [("raw", "Raw", (0.55, 0.55, 0.55, 1)), ("target", "Basic", (0.9, 0.45, 0.05, 1)), ("enhanced", "Enhanced", (0.1, 0.4, 0.85, 1))]
SPACING = 1.7

# (뼈, 시작 keypoint, 끝 keypoint, rest 방향을 정하는 자식 뼈)
LIMBS = [("l_upleg_JNT", 23, 25, "l_leg_JNT"), ("l_leg_JNT", 25, 27, "l_foot_JNT"), ("l_foot_JNT", 27, 31, "l_toebase_JNT"),
         ("r_upleg_JNT", 24, 26, "r_leg_JNT"), ("r_leg_JNT", 26, 28, "r_foot_JNT"), ("r_foot_JNT", 28, 32, "r_toebase_JNT"),
         ("l_arm_JNT", 11, 13, "l_forearm_JNT"), ("l_forearm_JNT", 13, 15, "l_hand_JNT"),
         ("r_arm_JNT", 12, 14, "r_forearm_JNT"), ("r_forearm_JNT", 14, 16, "r_hand_JNT")]
TORSO_HIPS = ["hips_JNT", "spine_JNT", "spine1_JNT"]
TORSO_CHEST = ["spine2_JNT", "neck_JNT", "l_shoulder_JNT", "r_shoulder_JNT"]


def to_blender(P):
    return np.stack([P[..., 0], P[..., 2], P[..., 1]], axis=-1)


def unit(v, fallback):
    n = np.linalg.norm(v)
    return v / n if n > 1e-5 else fallback


def basis(left, up):
    """캐릭터 rest 기저(왼쪽=+X, 뒤=+Y, 위=+Z)를 주어진 left/up 방향으로 돌리는 회전 행렬."""
    l = unit(left, np.array([1.0, 0, 0])); u = unit(up - np.dot(up, l) * l, np.array([0, 0, 1.0])); b = np.cross(u, l)
    return np.stack([l, b, u], axis=1)


def swing(a, b):
    """단위벡터 a 를 b 로 보내는 최소 회전."""
    v = np.cross(a, b); c = float(np.dot(a, b)); s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(unit(np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0]), a), unit(np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0]), a))
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def retarget(arm, P):
    """P: (T,33,3) Blender 좌표. 캐릭터 뼈에 회전/위치 키프레임 삽입."""
    bones = arm.data.bones; W = np.array(arm.matrix_world.to_3x3()); Wn = W / np.linalg.norm(W[:, 0])
    rest = {b.name: Wn @ np.array(b.matrix_local.to_3x3()) for b in bones}             # rest 방향(월드)
    head = {b.name: np.array(arm.matrix_world @ b.head_local) for b in bones}
    order = [b.name for b in bones]                                                      # 부모가 항상 먼저
    leg_rest = np.linalg.norm(head["l_leg_JNT"] - head["l_upleg_JNT"]) + np.linalg.norm(head["l_foot_JNT"] - head["l_leg_JNT"])
    leg_kp = np.median(np.linalg.norm(P[:, 25] - P[:, 23], axis=1) + np.linalg.norm(P[:, 27] - P[:, 25], axis=1))
    scale = leg_rest / max(leg_kp, 1e-3)
    limb = {n: (a, b, unit(head[c] - head[n], np.array([0, 0, -1.0]))) for n, a, b, c in LIMBS}
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    hips = arm.pose.bones["hips_JNT"]; Winv = np.linalg.inv(np.array(arm.matrix_world))
    hips_rest_inv = np.linalg.inv(np.array(bones["hips_JNT"].matrix_local))
    for t in range(len(P)):
        p = P[t]; D = {}
        hip_mid = (p[23] + p[24]) / 2; sh_mid = (p[11] + p[12]) / 2; ear_mid = (p[7] + p[8]) / 2
        Dh = basis(p[23] - p[24], sh_mid - hip_mid); Dc = basis(p[11] - p[12], sh_mid - hip_mid)
        f = p[0] - ear_mid; l = p[7] - p[8]
        Dhead = basis(l, np.cross(unit(f, np.array([0, -1.0, 0])), unit(l, np.array([1.0, 0, 0])))) if np.linalg.norm(l) > 1e-4 else Dc
        for n in order:
            par = bones[n].parent.name if bones[n].parent else None
            if n in TORSO_HIPS: D[n] = Dh
            elif n in TORSO_CHEST: D[n] = Dc
            elif n == "head_JNT": D[n] = Dhead
            elif n in limb:
                a, b, d0 = limb[n]; tgt = p[b] - p[a]
                D[n] = swing(D[par] @ d0, unit(tgt, D[par] @ d0)) @ D[par]
            else: D[n] = D[par]                                                          # 손가락, 발끝 등은 부모를 따라감
            A = D[n] @ rest[n]                                                           # 이번 프레임의 뼈 방향(월드)
            Ap = D[par] @ rest[par] if par else Wn
            R0 = (rest[par] if par else Wn).T @ rest[n]
            q = Matrix((R0.T @ Ap.T @ A).tolist()).to_quaternion()
            pb = arm.pose.bones[n]; pb.rotation_quaternion = q; pb.keyframe_insert("rotation_quaternion", frame=t + 1)
        tgt_world = np.append(hip_mid * scale + np.array(arm.location), 1.0)             # hips 위치
        loc = hips_rest_inv @ (Winv @ tgt_world)
        hips.location = Vector(loc[:3]); hips.keyframe_insert("location", frame=t + 1)
    return scale


def material(name, rgba, emit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF"); bsdf.inputs["Base Color"].default_value = rgba; bsdf.inputs["Roughness"].default_value = 0.6
    m.diffuse_color = rgba
    return m


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene; sc.render.fps = FPS
    T = None
    for k, (tag, label, col) in enumerate(VARIANTS):
        P = to_blender(np.load(os.path.join(ROOT, "data", "processed", f"{VIDEO}_{tag}.npy"))); T = len(P)
        before = set(bpy.data.objects)
        bpy.ops.import_scene.fbx(filepath=os.path.join(HERE, "assets", "character_tpose.fbx"))
        new = [o for o in bpy.data.objects if o not in before]
        arm = [o for o in new if o.type == "ARMATURE"][0]; mesh = [o for o in new if o.type == "MESH"][0]
        if arm.animation_data: arm.animation_data_clear()
        arm.name = f"RIG_{label}"; mesh.name = f"MESH_{label}"; arm.location = ((k - 1) * SPACING, 0, 0)
        bpy.context.view_layer.update()
        mesh.data.materials.clear(); mesh.data.materials.append(material(f"MAT_{label}", col))
        s = retarget(arm, P)
        # 원본 keypoint 도 작은 구로 같이 표시 (캐릭터와 데이터가 일치하는지 눈으로 확인용)
        txt = bpy.data.curves.new(f"TXT_{label}", "FONT"); txt.body = label; txt.align_x = "CENTER"; txt.size = 0.22
        to = bpy.data.objects.new(f"LABEL_{label}", txt); sc.collection.objects.link(to)
        to.location = ((k - 1) * SPACING, 0.3, 2.25); to.rotation_euler = (math.radians(84), 0, 0)
        to.data.materials.append(material(f"MATTXT_{label}", tuple(c * 0.6 for c in col[:3]) + (1,)))
        print(f"{label}: frames={T} scale={s:.3f}")
    sc.frame_start, sc.frame_end = 1, T
    # 지면 + grid
    bpy.ops.mesh.primitive_plane_add(size=1); g = bpy.context.object; g.name = "Ground"; g.scale = (7, 5, 1); g.location = (0, 0, -0.002)
    g.data.materials.append(material("MAT_Ground", (0.82, 0.82, 0.8, 1)))
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=28, y_subdivisions=20, size=1); w = bpy.context.object; w.name = "GroundGrid"; w.scale = (7, 5, 1)
    mod = w.modifiers.new("wire", "WIREFRAME"); mod.thickness = 0.004; w.data.materials.append(material("MAT_Grid", (0.35, 0.35, 0.35, 1)))
    # 지면 아래로 내려간 부분이 보이도록 지면을 반투명하게 하지 않고, 카메라를 살짝 낮게 + 옆 단면 벽 대신 raw 가 묻히는 모습 그대로 표시
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera")); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = (0.0, -6.6, 1.75); cam.rotation_euler = (math.radians(84), 0, 0); cam.data.lens = 36
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN")); sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(25)); sun.data.energy = 3.0
    world = bpy.data.worlds.new("World"); sc.world = world; world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.93, 0.94, 0.96, 1); world.node_tree.nodes["Background"].inputs[1].default_value = 0.55
    sc.render.resolution_x, sc.render.resolution_y = RES
    os.makedirs(os.path.join(HERE, "blend"), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "blend", f"{VIDEO}_compare.blend"))
    return sc


def render(sc):
    eng = {"EEVEE": ["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"], "WORKBENCH": ["BLENDER_WORKBENCH"], "CYCLES": ["CYCLES"]}[ENGINE]
    for e in eng:
        try:
            sc.render.engine = e; break
        except TypeError:
            continue
    if sc.render.engine == "CYCLES":
        sc.cycles.samples = 12; sc.cycles.use_denoising = False
    if "--mp4" in argv:                                   # 영상 파일로 바로 출력
        try: sc.render.image_settings.media_type = "VIDEO"   # Blender 5.x
        except Exception: pass
        sc.render.image_settings.file_format = "FFMPEG"; sc.render.ffmpeg.format = "MPEG4"; sc.render.ffmpeg.codec = "H264"
        sc.render.ffmpeg.constant_rate_factor = "HIGH"
        if FRAMES: sc.frame_start, sc.frame_end = FRAMES[0], FRAMES[-1]
        sc.render.filepath = os.path.join(ROOT, "videos", f"blender_character_{VIDEO}.mp4")
        bpy.ops.render.render(animation=True); print("rendered ->", sc.render.filepath, "engine", sc.render.engine); return
    out = os.path.join(ROOT, "videos", "blender_frames", VIDEO); os.makedirs(out, exist_ok=True)
    sc.render.image_settings.file_format = "PNG"
    for f in (FRAMES or range(sc.frame_start, sc.frame_end + 1)):
        sc.frame_set(f); sc.render.filepath = os.path.join(out, f"{f:04d}.png"); bpy.ops.render.render(write_still=True)
    print("rendered ->", out, "engine", sc.render.engine)


if __name__ == "__main__":
    scene = build()
    if RENDER:
        render(scene)
