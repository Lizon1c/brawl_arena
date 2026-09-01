"""Evaluate the omni student in real games: it drives unit 0 from raw frames
only (90x126, 16-frame window at stride 2), everyone else is scripted.

Usage:
    .venv/Scripts/python eval_student_omni.py runs/student_vision/omni_ep20.pt 20 [mode]
"""
import sys
from collections import deque

import numpy as np
import torch

from brawl_arena import BrawlArenaEnv, Config
from student_omni import OmniStudent

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/student_vision/omni_ep20.pt"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MODE = sys.argv[3] if len(sys.argv) > 3 else "gem_grab"
N_FRAMES, STRIDE = 16, 2
SPAN = (N_FRAMES - 1) * STRIDE          # 30: need the last 31 frames

device = "cuda" if torch.cuda.is_available() else "cpu"
net = OmniStudent(n_frames=N_FRAMES)
net.load_state_dict(torch.load(CKPT, map_location="cpu"))
net.eval().to(device)


def letterbox(f: np.ndarray) -> np.ndarray:
    if f.shape[0] == 90 and f.shape[1] == 126:
        return f
    t = torch.from_numpy(f).permute(2, 0, 1)[None].float()
    t = torch.nn.functional.interpolate(t, size=(90, 126), mode="bilinear",
                                        align_corners=False)
    return t[0].permute(1, 2, 0).byte().numpy()


@torch.no_grad()
def act(buf: deque) -> dict:
    frames = list(buf)
    while len(frames) < SPAN + 1:               # front-pad episode start
        frames.insert(0, frames[0])
    win = np.stack(frames[-SPAN - 1::STRIDE])   # (16, 90, 126, 3)
    x = torch.from_numpy(win).permute(0, 3, 1, 2)[None].float().to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16,
                        enabled=(device == "cuda")):
        out = net(x)
    move = out["move"][0].float().cpu().numpy()
    aim = out["aim"][0].float().cpu().numpy()
    aim = aim / (np.linalg.norm(aim) + 1e-6)
    return {"move": move.astype(np.float32), "aim": aim.astype(np.float32),
            "shoot": int(out["shoot"][0].argmax()),
            "super": int(out["super"][0].argmax())}


wins = draws = losses = 0
gem_diffs = []
for seed in range(N):
    env = BrawlArenaEnv(config=Config(mode=MODE), seed=6000 + seed,
                        include_frame=True)
    obs, _ = env.reset(seed=6000 + seed)
    buf = deque(maxlen=SPAN + 1)
    buf.append(letterbox(obs["frame"]))
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(act(buf))
        buf.append(letterbox(obs["frame"]))
        done = terminated or truncated
    w = info["winner"]
    wins += w == 0
    losses += w == 1
    draws += w not in (0, 1)
    gem_diffs.append(info["gems"][0] - info["gems"][1])
print(f"{CKPT} [{MODE}]: over {N} games W{wins} D{draws} L{losses}, "
      f"avg gem diff {np.mean(gem_diffs):+.1f}")
