"""A/B: same checkpoint, two inference paths, identical seeds."""
from collections import deque

import numpy as np
import torch

from brawl_arena import BrawlArenaEnv, Config
from student_omni import OmniStudent
from student_rl import RLStudent, _lb, SPAN, N_FRAMES, STRIDE

CKPT_RL = "runs/student_rl/rl_iter25.pt"
CKPT_OMNI = "runs/student_rl/rl_iter25_as_omni.pt"
N = 10
device = "cuda"


def play_omni(net, env, buf):
    frames = list(buf)
    while len(frames) < SPAN + 1:
        frames.insert(0, frames[0])
    win = np.stack(frames[-SPAN - 1::STRIDE])
    x = torch.from_numpy(win).permute(0, 3, 1, 2)[None].float().to(device)
    with torch.no_grad():
        out = net(x)
    aim = out["aim"][0].cpu().numpy()
    aim = aim / (np.linalg.norm(aim) + 1e-6)
    return {"move": out["move"][0].cpu().numpy().astype(np.float32),
            "aim": aim.astype(np.float32),
            "shoot": int(out["shoot"][0].argmax()),
            "super": int(out["super"][0].argmax())}


def play_rl(net, buf, bf16):
    while len(buf) < SPAN + 1:
        buf.appendleft(buf[0])
    fw = torch.stack(list(buf)[::STRIDE])[-N_FRAMES:][None]
    with torch.no_grad():
        out = net.forward_feats(fw)
    aim = out["aim_mu"][0].cpu().numpy()
    aim = aim / (np.linalg.norm(aim) + 1e-6)
    return {"move": out["move_mu"][0].cpu().numpy().astype(np.float32),
            "aim": aim.astype(np.float32),
            "shoot": int(out["shoot"][0].argmax()),
            "super": int(out["super"][0].argmax())}


omni = OmniStudent().to(device).eval()
omni.load_state_dict(torch.load(CKPT_OMNI, map_location="cpu"))
rl = RLStudent(encoder="osc").to(device).eval()
rl.load_state_dict(torch.load(CKPT_RL, map_location="cpu"))

res = {"omni_fp32": [], "rl_bf16": [], "rl_fp32": []}
for seed in range(N):
    for variant in res:
        env = BrawlArenaEnv(config=Config(mode="gem_grab"),
                            seed=6000 + seed, include_frame=True)
        obs, _ = env.reset(seed=6000 + seed)
        if variant == "omni_fp32":
            buf = deque(maxlen=SPAN + 1)
            buf.append(_lb(obs["frame"]))
        else:
            buf = deque(maxlen=SPAN + 1)
            bf16 = variant == "rl_bf16"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                f = rl.encode(torch.from_numpy(_lb(obs["frame"]))
                              .permute(2, 0, 1)[None].to(device))[0]
            buf.append(f)
        done = False
        while not done:
            if variant == "omni_fp32":
                a = play_omni(omni, env, buf)
            else:
                a = play_rl(rl, buf, bf16)
            obs, _, term, _, info = env.step(a)
            if variant == "omni_fp32":
                buf.append(_lb(obs["frame"]))
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=bf16):
                    f = rl.encode(torch.from_numpy(_lb(obs["frame"]))
                                  .permute(2, 0, 1)[None].to(device))[0]
                buf.append(f)
            done = term
        res[variant].append(info["winner"])
for k, v in res.items():
    w = sum(x == 0 for x in v)
    l = sum(x == 1 for x in v)
    d = sum(x not in (0, 1) for x in v)
    print(f"{k}: W{w} D{d} L{l}  {v}")
