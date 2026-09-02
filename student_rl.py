"""Online RL for the vision student: three gradient streams, one network.

Streams (8 envs in one process, modes cycle, selfplay on gem_grab only):
  A. student vs scripted bots  (x4) -> PPO + DAgger aux (oracle labels free)
  B. self-play vs frozen past self (x2, gem_grab) -> PPO
     (frozen opponent drives team 1 via opponent_policy; per-unit magenta
      ring marker tells it which unit it is driving this call)
  C. spectate bot vs bot (x2) -> behaviour cloning on never-repeating states

Data is produced online, never reused across iterations (PPO is on-policy;
the BC streams are fresh episodes every time).

Engineering: the DINO backbone is FROZEN during RL. Features (384-d) are
computed once per env step into a rolling per-env cache; PPO/BC updates run
the oscillator bank + readout + heads on cached feature windows. Full-
backbone PPO is infeasible at RL throughput on a single GPU.

Usage:
    .venv/Scripts/python student_rl.py --iters 2000
    .venv/Scripts/python student_rl.py --eval runs/student_rl/rl_latest.pt 20
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brawl_arena import BrawlArenaEnv, Config
from brawl_arena.core import MODES, Action
from student_vision import OscillatorBank
from student_omni import OmniStudent, IMG, DINO_DIM

OUT = os.path.join("runs", "student_rl")
N_FRAMES, STRIDE = 16, 2
SPAN = (N_FRAMES - 1) * STRIDE
INIT = "runs/student_vision/omni_ep37.pt"
MARKER = (255, 0, 255)  # magenta ring = "you are this unit" (self-play only)


# ------------------------------------------------------------------ model
class RLStudent(nn.Module):
    """Frozen DINO trunk (features precomputed) + CPG + actor-critic heads.

    move/aim heads output Gaussian means; shoot/super are Bernoulli logits;
    value head predicts P(team 0 wins) * 2 - 1."""

    def __init__(self, n_modes: int = 128):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2",
                                       "dinov2_vits14", pretrained=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.osc = OscillatorBank(DINO_DIM, n_modes)
        self.readout = nn.Sequential(nn.Linear(2 * n_modes, 256), nn.ReLU())
        self.move_head = nn.Linear(256, 2)
        self.aim_head = nn.Linear(256, 2)
        self.shoot_head = nn.Linear(256, 2)
        self.super_head = nn.Linear(256, 2)
        # critic reads DETACHED features: value loss may never reshape the
        # policy trunk (BC features sit in a sharp basin; v2 measured a
        # 40%->20% winrate drop from 2% weight drift caused by value grads)
        self.value_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                        nn.Linear(128, 1))
        self.log_std_move = nn.Parameter(torch.full((2,), -1.4))
        self.log_std_aim = nn.Parameter(torch.full((2,), -1.4))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) uint8-range -> (B, 384) frozen backbone features."""
        v = frames.float() / 255.0
        v = (v - self.mean) / self.std
        v = F.interpolate(v, size=(IMG, IMG), mode="bilinear",
                          align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=frames.is_cuda):
            f = self.backbone.forward_features(v)["x_norm_clstoken"]
        return f.float()

    def forward_feats(self, fw: torch.Tensor):
        """(B, T, 384) feature window -> action dists params + value."""
        z = self.readout(self.osc(fw))
        return {"move_mu": torch.tanh(self.move_head(z)),
                "aim_mu": self.aim_head(z),
                "shoot": self.shoot_head(z),
                "super": self.super_head(z),
                "value": self.value_head(z.detach()).squeeze(-1)}

    def sample(self, fw: torch.Tensor):
        out = self.forward_feats(fw)
        sm = torch.exp(self.log_std_move).expand_as(out["move_mu"])
        sa = torch.exp(self.log_std_aim).expand_as(out["aim_mu"])
        dm = torch.distributions.Normal(out["move_mu"], sm)
        da = torch.distributions.Normal(out["aim_mu"], sa)
        am, aa = dm.sample(), da.sample()
        logp = dm.log_prob(am).sum(-1) + da.log_prob(aa).sum(-1)
        ent = dm.entropy().sum(-1) + da.entropy().sum(-1)
        btns = []
        for k in ("shoot", "super"):
            d = torch.distributions.Categorical(logits=out[k])
            b = d.sample()
            logp = logp + d.log_prob(b)
            ent = ent + d.entropy()
            btns.append(b)
        return out, am, aa, btns, logp, ent

    def load_bc(self, path: str):
        """Warm start from an OmniStudent BC checkpoint (shape-filtered)."""
        sd = torch.load(path, map_location="cpu")
        own = self.state_dict()
        ok = {k: v for k, v in sd.items()
              if k in own and own[k].shape == v.shape}
        own.update(ok)
        self.load_state_dict(own)
        print(f"warm start: {len(ok)}/{len(sd)} tensors from {path}")


# ------------------------------------------------------------- frozen rival
def _marker_ring(img: np.ndarray, x: float, y: float, s: int):
    """Paint a magenta ring around tile position (x, y) on the frame."""
    px, py = int(x * s), int(y * s)
    r_out, r_in = int(0.75 * s), int(0.5 * s)
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    d2 = (xx - px) ** 2 + (yy - py) ** 2
    img[(d2 <= r_out ** 2) & (d2 >= r_in ** 2)] = MARKER


class FrozenStudentPolicy:
    """Wraps a frozen RLStudent as an opponent_policy (act_batch API).
    Drives each enemy unit from a viewer_team=1 render with a magenta ring
    marking which unit is being driven. Own rolling feature cache per unit."""

    def __init__(self, ckpt: str, device: str):
        self.net = RLStudent()
        self.net.load_bc(ckpt)
        self.net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self._feats = {}       # unit idx -> deque of features
        self._game_id = None

    def act_batch(self, game, idxs):
        if id(game) != self._game_id or game.t < 2:
            self._feats = {i: deque(maxlen=SPAN + 1) for i in idxs}
            self._game_id = id(game)
        s = game.cfg.render_scale
        frames = []
        for i in idxs:
            from brawl_arena.render import render
            img = render(game, viewer_team=game.units[i].team)
            u = game.units[i]
            if img.shape[0] != 90 or img.shape[1] != 126:
                t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
                t = F.interpolate(t, size=(90, 126), mode="bilinear",
                                  align_corners=False)
                img = t[0].permute(1, 2, 0).byte().numpy()
            _marker_ring(img, u.pos[0], u.pos[1], s)
            frames.append(img)
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        feats = self.net.encode(x.to(self.device))
        acts = []
        for k, i in enumerate(idxs):
            buf = self._feats[i]
            buf.append(feats[k])
            while len(buf) < SPAN + 1:
                buf.appendleft(buf[0])
            fw = torch.stack(list(buf)[::STRIDE])[-N_FRAMES:][None]
            out = self.net.forward_feats(fw)
            aim = out["aim_mu"][0].cpu().numpy()
            aim = aim / (np.linalg.norm(aim) + 1e-6)
            acts.append(Action(
                move=out["move_mu"][0].cpu().numpy().astype(np.float64),
                aim=aim.astype(np.float64),
                shoot=bool(out["shoot"][0].argmax()),
                use_super=bool(out["super"][0].argmax())))
        return acts


# ------------------------------------------------------------- letterboxing
def _lb(f: np.ndarray) -> np.ndarray:
    if f.shape[0] == 90 and f.shape[1] == 126:
        return f
    t = torch.from_numpy(f).permute(2, 0, 1)[None].float()
    t = F.interpolate(t, size=(90, 126), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).byte().numpy()


# ----------------------------------------------------------------- training
def train(n_iters: int, device: str, init: str = INIT, horizon: int = 256,
          lr: float = 3e-5, bc_dagger: float = 1.0, warmup_iters: int = 30,
          target_kl: float = 0.02, ppo_epochs: int = 2):
    os.makedirs(OUT, exist_ok=True)
    roles = ["vs_bots"] * 4 + ["selfplay"] * 2 + ["spectate"] * 2
    envs = []
    for i, role in enumerate(roles):
        if role == "selfplay":
            mode = "gem_grab"
        elif role == "vs_bots":
            mode = MODES[i % len(MODES)]          # one per mode
        else:
            mode = MODES[(i + 2) % len(MODES)]    # knockout, showdown
        envs.append(BrawlArenaEnv(
            config=Config(mode=mode, reward_shaping=True),
            seed=20000 + i, include_frame=True))

    net = RLStudent().to(device)
    net.load_bc(init)
    rival = FrozenStudentPolicy(init, device)
    for env, role in zip(envs, roles):
        if role == "selfplay":
            env.opponent_policy = rival

    trainable = [p for n_, p in net.named_parameters()
                 if not n_.startswith("backbone")]
    opt = torch.optim.Adam(trainable, lr=lr)
    GAMMA, LAM, CLIP, ENT, VF = 0.995, 0.95, 0.2, 0.003, 0.5
    BC_SPECT, BC_DAGGER = 0.5, bc_dagger

    # per-env rolling feature cache and episode state
    feats = [None] * len(envs)
    obs_frames = [None] * len(envs)
    ep_ret = np.zeros(len(envs))
    results = {}                 # mode -> deque of 1/0.5/0, vs_bots only
    for m in MODES:
        results[m] = deque(maxlen=100)
    sp_results = deque(maxlen=100)   # selfplay: student vs frozen self

    def reset_env(i):
        o, _ = envs[i].reset()
        obs_frames[i] = _lb(o["frame"])
        feats[i] = deque(maxlen=SPAN + 1)
        ep_ret[i] = 0.0

    for i in range(len(envs)):
        reset_env(i)

    student_envs = [i for i, r in enumerate(roles) if r != "spectate"]

    t_start = time.perf_counter()
    total_steps = 0
    for it in range(1, n_iters + 1):
        # ---------------- rollout (per-env buffers keep GAE intact) --------
        bufs = {i: {k: [] for k in ("fw", "am", "aa", "btns", "logp", "val",
                                    "rew", "done", "dmv", "dam", "dbt")}
                for i in student_envs}
        spec = {k: [] for k in ("fw", "mv", "am", "bt")}
        for t in range(horizon):
            # one batched backbone forward for all envs' fresh frames
            x = torch.from_numpy(np.stack(obs_frames)).permute(0, 3, 1, 2)
            f_new = net.encode(x.to(device))
            for i in range(len(envs)):
                feats[i].append(f_new[i])

            for i, role in enumerate(roles):
                buf_ = feats[i]
                while len(buf_) < SPAN + 1:
                    buf_.appendleft(buf_[0])
                fw = torch.stack(list(buf_)[::STRIDE])[-N_FRAMES:]
                game = envs[i].game
                if role == "spectate":
                    bot = envs[i].teammate_policy.act(game, 0)
                    spec["fw"].append(fw)
                    spec["mv"].append(torch.from_numpy(
                        np.asarray(bot.move, dtype=np.float32)))
                    a = np.asarray(bot.aim, dtype=np.float32)
                    spec["am"].append(torch.from_numpy(
                        a / (np.linalg.norm(a) + 1e-6)))
                    spec["bt"].append((int(bot.shoot), int(bot.use_super)))
                    act_dict = {"move": bot.move.astype(np.float32),
                                "aim": bot.aim.astype(np.float32),
                                "shoot": int(bot.shoot),
                                "super": int(bot.use_super)}
                else:
                    b = bufs[i]
                    out, am, aa, btns, logp, ent = net.sample(
                        fw[None].to(device))
                    # DAgger oracle label on the student's own state
                    bot = envs[i].teammate_policy.act(game, 0)
                    b["fw"].append(fw)
                    b["am"].append(am[0].cpu())
                    b["aa"].append(aa[0].cpu())
                    b["btns"].append((int(btns[0]), int(btns[1])))
                    b["logp"].append(float(logp.detach()))
                    b["val"].append(float(out["value"]))
                    b["dmv"].append(torch.from_numpy(
                        np.asarray(bot.move, dtype=np.float32)))
                    ba = np.asarray(bot.aim, dtype=np.float32)
                    b["dam"].append(torch.from_numpy(
                        ba / (np.linalg.norm(ba) + 1e-6)))
                    b["dbt"].append((int(bot.shoot), int(bot.use_super)))
                    aim = aa[0].cpu().numpy()
                    aim = aim / (np.linalg.norm(aim) + 1e-6)
                    act_dict = {"move": torch.tanh(am[0]).cpu().numpy()
                                .astype(np.float32),
                                "aim": aim.astype(np.float32),
                                "shoot": int(btns[0]), "super": int(btns[1])}
                o, r, done, _, info = envs[i].step(act_dict)
                obs_frames[i] = _lb(o["frame"])
                if role != "spectate":
                    bufs[i]["rew"].append(r)
                    bufs[i]["done"].append(float(done))
                    ep_ret[i] += r
                if done:
                    if info["winner"] is not None:
                        sc = (1.0 if info["winner"] == 0 else
                              0.5 if info["winner"] not in (0, 1) else 0.0)
                        if role == "vs_bots":
                            results[info["mode"]].append(sc)
                        elif role == "selfplay":
                            sp_results.append(sc)
                    reset_env(i)
            total_steps += len(envs)

        # ---------------- PPO update on cached features ----------------
        # bootstrap value for the final state of each env's horizon
        last_vals = {}
        for i in student_envs:
            buf_ = feats[i]
            if len(buf_) == 0:   # episode ended on the final step
                x = torch.from_numpy(obs_frames[i]).permute(2, 0, 1)[None]
                buf_.append(net.encode(x.to(device))[0])
            while len(buf_) < SPAN + 1:
                buf_.appendleft(buf_[0])
            fw_last = torch.stack(list(buf_)[::STRIDE])[-N_FRAMES:]
            with torch.no_grad():
                last_vals[i] = float(net.forward_feats(
                    fw_last[None].to(device))["value"])

        fw_l, am_l, aa_l, btns_l, logp_l, adv_l, ret_l = [], [], [], [], [], [], []
        dmv_l, dam_l, dbt_l = [], [], []
        for i in student_envs:
            b = bufs[i]
            T_i = len(b["rew"])
            rew = np.array(b["rew"], dtype=np.float32)
            don = np.array(b["done"], dtype=np.float32)
            val = np.array(b["val"] + [last_vals[i]], dtype=np.float32)
            adv = np.zeros(T_i, dtype=np.float32)
            lastgae = 0.0
            for t in reversed(range(T_i)):
                delta = rew[t] + GAMMA * val[t + 1] * (1 - don[t]) - val[t]
                lastgae = delta + GAMMA * LAM * (1 - don[t]) * lastgae
                adv[t] = lastgae
            ret = adv + val[:T_i]
            fw_l.extend(b["fw"])
            am_l.extend(b["am"])
            aa_l.extend(b["aa"])
            btns_l.extend(b["btns"])
            logp_l.extend(b["logp"])
            adv_l.extend(adv)
            ret_l.extend(ret)
            dmv_l.extend(b["dmv"])
            dam_l.extend(b["dam"])
            dbt_l.extend(b["dbt"])

        n_std = len(adv_l)
        fw = torch.stack(fw_l).to(device)            # (N,16,384)
        am = torch.stack(am_l).to(device)
        aa = torch.stack(aa_l).to(device)
        btns = torch.tensor(btns_l, device=device)   # (N,2)
        old_logp = torch.tensor(logp_l, device=device)
        adv = torch.tensor(adv_l, device=device)
        ret = torch.tensor(ret_l, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # spectate BC tensors
        s_fw = torch.stack(spec["fw"]).to(device)
        s_mv = torch.stack(spec["mv"]).to(device)
        s_am = torch.stack(spec["am"]).to(device)
        s_bt = torch.tensor(spec["bt"], device=device)
        d_mv = torch.stack(dmv_l).to(device)
        d_am = torch.stack(dam_l).to(device)
        d_bt = torch.tensor(dbt_l, device=device)

        idx_all = np.arange(n_std)
        pl = vl = bl = 0.0
        nup = 0
        kl_stop = False
        for ep in range(ppo_epochs):
            if kl_stop:
                break
            np.random.shuffle(idx_all)
            kl_acc = 0.0
            for b0 in range(0, n_std, 512):
                mb = idx_all[b0:b0 + 512]
                out = net.forward_feats(fw[mb])
                sm = torch.exp(net.log_std_move).expand_as(out["move_mu"])
                sa = torch.exp(net.log_std_aim).expand_as(out["aim_mu"])
                lp = (torch.distributions.Normal(out["move_mu"], sm)
                      .log_prob(am[mb]).sum(-1)
                      + torch.distributions.Normal(out["aim_mu"], sa)
                      .log_prob(aa[mb]).sum(-1))
                ent = (torch.distributions.Normal(out["move_mu"], sm)
                       .entropy().sum(-1)
                       + torch.distributions.Normal(out["aim_mu"], sa)
                       .entropy().sum(-1))
                for k, key in enumerate(("shoot", "super")):
                    d = torch.distributions.Categorical(logits=out[key])
                    lp = lp + d.log_prob(btns[mb, k])
                    ent = ent + d.entropy()
                kl_acc += float((old_logp[mb] - lp).mean())
                ratio = torch.exp(lp - old_logp[mb])
                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[mb]
                loss_pi = -torch.min(s1, s2).mean()
                loss_v = F.mse_loss(out["value"], ret[mb])
                # value-head warmup: freeze the policy loss until the critic
                # has seen enough data to produce sane advantages
                if it <= warmup_iters:
                    loss = VF * loss_v
                else:
                    loss = loss_pi + VF * loss_v - ENT * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 0.5)
                opt.step()
                pl += float(loss_pi)
                vl += float(loss_v)
                nup += 1
            if kl_acc / max(1, len(range(0, n_std, 512))) > target_kl:
                kl_stop = True
        # BC passes (spectate + DAgger), one epoch
        for src_fw, src_mv, src_am, src_bt, w in (
                (s_fw, s_mv, s_am, s_bt, BC_SPECT),
                (fw, d_mv, d_am, d_bt, BC_DAGGER)):
            nsrc = len(src_fw)
            order = np.random.permutation(nsrc)
            for b0 in range(0, nsrc, 512):
                mb = order[b0:b0 + 512]
                out = net.forward_feats(src_fw[mb])
                loss = w * (F.mse_loss(out["move_mu"], src_mv[mb])
                            + F.mse_loss(out["aim_mu"], src_am[mb])
                            + F.cross_entropy(out["shoot"], src_bt[mb][:, 0])
                            + F.cross_entropy(out["super"], src_bt[mb][:, 1]))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 0.5)
                opt.step()
                bl += float(loss)

        if it % 5 == 0:
            fps = total_steps / (time.perf_counter() - t_start)
            wr = {m: (np.mean(d) if d else float("nan"))
                  for m, d in results.items()}
            sp = np.mean(sp_results) if sp_results else float("nan")
            print(f"[it {it}] steps {total_steps} ({fps:.0f}/s) "
                  f"pi {pl / nup:.4f} v {vl / nup:.4f} bc {bl:.3f} "
                  f"wr gem {wr['gem_grab']:.2f} ball {wr['brawl_ball']:.2f} "
                  f"ko {wr['knockout']:.2f} sd {wr['showdown']:.2f} "
                  f"selfplay {sp:.2f}", flush=True)
        if it % 25 == 0:
            torch.save(net.state_dict(), os.path.join(OUT, "rl_latest.pt"))
            torch.save(net.state_dict(),
                       os.path.join(OUT, f"rl_iter{it}.pt"))
    torch.save(net.state_dict(), os.path.join(OUT, "rl_latest.pt"))
    print("done")


# ------------------------------------------------------------------- eval
def evaluate(ckpt: str, n: int, mode: str = "gem_grab", seed0: int = 7000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = RLStudent()
    try:
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    except RuntimeError:
        net.load_bc(ckpt)   # OmniStudent-era checkpoint
    net.eval().to(device)
    wins = draws = losses = 0
    gem_diffs = []
    for seed in range(n):
        env = BrawlArenaEnv(config=Config(mode=mode), seed=seed0 + seed,
                            include_frame=True)
        obs, _ = env.reset(seed=seed0 + seed)
        buf = deque(maxlen=SPAN + 1)
        done = False
        while not done:
            buf.append(net.encode(torch.from_numpy(
                _lb(obs["frame"])).permute(2, 0, 1)[None].to(device))[0])
            while len(buf) < SPAN + 1:
                buf.appendleft(buf[0])
            fw = torch.stack(list(buf)[::STRIDE])[-N_FRAMES:][None]
            with torch.no_grad():
                out = net.forward_feats(fw)
            aim = out["aim_mu"][0].cpu().numpy()
            aim = aim / (np.linalg.norm(aim) + 1e-6)
            obs, _, term, _, info = env.step({
                "move": out["move_mu"][0].cpu().numpy().astype(np.float32),
                "aim": aim.astype(np.float32),
                "shoot": int(out["shoot"][0].argmax()),
                "super": int(out["super"][0].argmax())})
            done = term
        w = info["winner"]
        wins += w == 0
        losses += w == 1
        draws += w not in (0, 1)
        if "gems" in info:
            gem_diffs.append(info["gems"][0] - info["gems"][1])
    gd = f", avg gem diff {np.mean(gem_diffs):+.1f}" if gem_diffs else ""
    print(f"{ckpt} [{mode}]: over {n} games W{wins} D{draws} L{losses}{gd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--eval", type=str, default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", type=str, default="gem_grab")
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--init", type=str, default=INIT)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bc-dagger", type=float, default=0.5)
    ap.add_argument("--warmup-iters", type=int, default=30)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    args = ap.parse_args()
    if args.eval:
        evaluate(args.eval, args.n, args.mode, args.seed0)
    if args.iters:
        train(args.iters, args.device, args.init, lr=args.lr,
              bc_dagger=args.bc_dagger, warmup_iters=args.warmup_iters,
              ppo_epochs=args.ppo_epochs)


if __name__ == "__main__":
    main()
