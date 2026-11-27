"""viser_compare.py - 교수님 PoseViser(utils/viser_test.py)를 확장한 3-way Viser 뷰어.
  python scripts/viser_compare.py handstand     -> http://localhost:8080 (viser 기본 포트)
Raw | Basic | Enhanced 를 나란히 재생. GUI: Play/Pause, 프레임 슬라이더, 속도. 지면 아래 관절은 빨간색."""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from motion_lib import POSE_CONNECTIONS


def run(name="handstand", offset=1.5):
    import viser
    seqs = [np.load(os.path.join(ROOT, "data", "processed", f"{name}_{t}.npy")) for t in ("raw", "target", "enhanced")]
    cols = [(0.55, 0.55, 0.55), (0.9, 0.45, 0.05), (0.1, 0.4, 0.85)]
    T = len(seqs[0])
    server = viser.ViserServer()
    server.scene.set_up_direction("+y")
    g = np.arange(-3, 3.01, 0.25)
    grid = np.concatenate([[[[x, 0, -1.5], [x, 0, 1.5]] for x in g], [[[-3, 0, z], [3, 0, z]] for z in np.arange(-1.5, 1.51, 0.25)]])
    server.scene.add_line_segments("/grid", points=grid, colors=np.full((len(grid), 2, 3), 0.8), line_width=1.0)
    play = server.gui.add_checkbox("Play", True)
    frame = server.gui.add_slider("Frame", 0, T - 1, 1, 0)
    speed = server.gui.add_slider("Speed", 0.1, 2.0, 0.1, 1.0)
    fps = server.gui.add_number("FPS", 24)
    while True:
        t = int(frame.value)
        for k, (s, c) in enumerate(zip(seqs, cols)):
            p = s[t].copy(); p[:, 0] += (k - 1) * offset
            pc = np.tile(c, (33, 1)); pc[s[t][:, 1] < -0.01] = (0.85, 0.1, 0.1)
            server.scene.add_point_cloud(f"/m{k}/pts", points=p, colors=pc, point_size=0.03)
            seg = np.array([[p[a], p[b]] for a, b in POSE_CONNECTIONS])
            server.scene.add_line_segments(f"/m{k}/lines", points=seg, colors=np.tile(c, (len(seg), 2, 1)), line_width=3.0)
        if play.value:
            frame.value = (t + 1) % T
        time.sleep(1.0 / (fps.value * speed.value))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "handstand")
