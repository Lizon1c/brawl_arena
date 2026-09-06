"""Duel combat-metrics eval: the sensitive early-stage instrument.

Gem-grab winrate conflates combat skill with strategic skill (countdown
swings, gem-carrier protection, team coordination) and stays in a wide
noise band while fundamentals are still forming. Duel vs the scripted bot
isolates raw combat: rounds won, kills, hits landed/taken per game.

Prints BOTH policies: argmax (deployment form; measurably passive --
09-05 mirror diagnostic) and sampled (training/rollout form; the one that
actually fights). The argmax-vs-sampled gap is itself a signal.

Usage:
    .venv/Scripts/python scripts/eval_duel.py [ckpt] [n_games]
Defaults: runs/student_rl/rl_latest.pt, 10 games per policy, seeds 6000+.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from brawl_arena import BrawlArenaEnv, Config
from student_rl import RLStudent, FeatRing, _window, _self_ring, _lb

ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/student_rl/rl_latest.pt"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

device = "cuda" if torch.cuda.is_available() else "cpu"
net = RLStudent(encoder="tx", backbone="cnn")
net.load_state_dict(torch.load(ckpt, map_location="cpu"))
net.eval().to(device)


def run(sampled: bool):
    tot = dict(hits_by=0, hits_on=0, kills=0, deaths=0, rw0=0, rw1=0, w=0)
    for k in range(n):
        env = BrawlArenaEnv(config=Config(mode="duel"), seed=6000 + k,
                            include_frame=True)
        obs, _ = env.reset(seed=6000 + k)
        captured = []
        orig = env._reward

        def wrapped(orig=orig, captured=captured):
            captured.extend(list(env.game.events))
            return orig()
        env._reward = wrapped

        ring = FeatRing()
        done = False
        while not done:
            f = _lb(obs["frame"])
            _self_ring(f, env.game, 0)
            ring.append(net.encode(torch.from_numpy(
                f).permute(2, 0, 1)[None].to(device))[0])
            fw = _window(ring, net)[None].to(device)
            with torch.no_grad():
                out = net.forward_feats(fw)
            if sampled:
                sm = torch.exp(net.log_std_move).expand_as(out["move_mu"])
                sa = torch.exp(net.log_std_aim).expand_as(out["aim_mu"])
                mv = torch.tanh(torch.distributions.Normal(
                    out["move_mu"], sm).sample())
                av = torch.distributions.Normal(out["aim_mu"], sa).sample()
                shoot = int(torch.distributions.Categorical(
                    logits=out["shoot"]).sample())
                sup = int(torch.distributions.Categorical(
                    logits=out["super"]).sample())
                move = mv[0].cpu().numpy().astype(np.float32)
            else:
                av = out["aim_mu"]
                shoot = int(out["shoot"][0].argmax())
                sup = int(out["super"][0].argmax())
                move = out["move_mu"][0].cpu().numpy().astype(np.float32)
            aim = av[0].cpu().numpy()
            aim = aim / (np.linalg.norm(aim) + 1e-6)
            obs, r, done, _, info = env.step({
                "move": move, "aim": aim.astype(np.float32),
                "shoot": shoot, "super": sup})
        tot["w"] += info["winner"] == 0
        for ev in captured:
            if ev[0] == "hit":
                _, a, v = ev
                tot["hits_by"] += (a == 0)
                tot["hits_on"] += (v == 0)
            elif ev[0] == "kill":
                _, a, v = ev
                tot["kills"] += (a == 0)
                tot["deaths"] += (v == 0)
            elif ev[0] == "round_end":
                tot["rw0"] += (ev[1] == 0)
                tot["rw1"] += (ev[1] == 1)
    return tot


for sampled in (False, True):
    t = run(sampled)
    tag = "sampled" if sampled else "argmax"
    print(f"{ckpt} [duel-metrics {tag}] over {n} games: "
          f"games {t['w']}W/{n - t['w']}L, "
          f"rounds {t['rw0']}W/{t['rw1']}L, "
          f"kills {t['kills']}, deaths {t['deaths']}, "
          f"hits {t['hits_by'] / n:.1f}/game landed, "
          f"{t['hits_on'] / n:.1f}/game taken")
