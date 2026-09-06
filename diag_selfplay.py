"""Diagnose self-play WR=1.00: is it side bias or a FrozenStudentPolicy bug?

Test A: scripted bot drives unit 0, all others scripted bots -> team-0 base rate.
Test B: exact training selfplay setup (deterministic RLStudent on unit 0,
FrozenStudentPolicy drives units 1-5) with IDENTICAL bc_warm weights on both
sides -> should be ~50% if the frozen path is sound.
"""
import sys
from collections import deque

import numpy as np
import torch

from brawl_arena import BrawlArenaEnv, Config
from brawl_arena.bots import ScriptedBot
from student_rl import (N_FRAMES, SPAN, STRIDE, FrozenStudentPolicy,
                        RLStudent, _lb)

CKPT = "runs/student_tx/bc_warm.pt"


def test_bot_mirror(n=20):
    wins0 = 0
    for s in range(n):
        env = BrawlArenaEnv(config=Config(mode="gem_grab"), seed=30000 + s,
                            include_frame=False)
        bot0 = ScriptedBot(seed=1000 + s)
        env.reset()
        done = False
        while not done:
            a = bot0.act(env.game, 0)
            _, _, done, _, info = env.step({
                "move": a.move, "aim": a.aim, "shoot": int(a.shoot),
                "super": int(a.use_super)})
        wins0 += info["winner"] == 0
    print(f"[A] bot mirror: team0 wins {wins0}/{n}", flush=True)


def det_student_act(net, buf, frame, device):
    buf.append(net.encode(torch.from_numpy(_lb(frame))
                          .permute(2, 0, 1)[None].to(device))[0])
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


def test_selfplay_setup(n=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = RLStudent()
    net.load_bc(CKPT)
    net.eval().to(device)
    rival = FrozenStudentPolicy(CKPT, device)
    wins = 0
    for s in range(n):
        env = BrawlArenaEnv(config=Config(mode="gem_grab"), seed=40000 + s,
                            include_frame=True)
        env.opponent_policy = rival
        env.teammate_policy = rival
        obs, _ = env.reset()
        buf = deque(maxlen=SPAN + 1)
        done = False
        while not done:
            obs, _, done, _, info = env.step(
                det_student_act(net, buf, obs["frame"], device))
        wins += info["winner"] == 0
        print(f"  game {s}: winner={info['winner']} t={info['t']} "
              f"gems={info.get('gems')}", flush=True)
    print(f"[B] selfplay setup (identical weights): team0 wins {wins}/{n}",
          flush=True)


if __name__ == "__main__":
    test_bot_mirror(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
    test_selfplay_setup(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
