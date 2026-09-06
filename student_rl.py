"""Online RL for the vision student: three gradient streams, one network.

Streams (9 envs in one process; v5.3 mix: vs_bots x4 + selfplay x4
  + spectate x1; duo envs run TWO learning students each):
  A. student(s) vs scripted bots (gem x2, duo 2v1 x2) -> PPO + DAgger aux
     (duo = 2 learning students vs 1 bot on knockout machinery: reachable
      positive reward + two gradient streams per game. 1v1-vs-bot duel was
      removed from vs_bots in v5.3: no reachable positive reward there --
      0 kills, 1.9 hits/game, wr pinned 0.00 across two versions)
  B. self-play vs frozen past self (x4: gem/ball/ko/duel) -> PPO
     (symmetric: the frozen policy drives ALL non-student units via
      opponent_policy + teammate_policy and SAMPLES actions exactly like
      the training-time student -- an argmax rival plays passively, which
      inflated ko/duel self-play wr to ~0.9 before the v5.2b fix; a gated
      ratchet promotes the current net to frozen rival once rolling
      self-play WR >= 0.60)

Every policy-facing frame carries a magenta self-marker ring on the driven
unit (see _self_ring): student rollout, eval, spectate BC and frozen-rival
views all use it. Without self-localization a multi-unit policy cannot
navigate; the ring is also what the on-device deployment will need.
  C. spectate bot vs bot (x2) -> behaviour cloning on never-repeating states

Data is produced online, never reused across iterations (PPO is on-policy;
the BC streams are fresh episodes every time).

Engineering: the DINO backbone is FROZEN during RL. Features (384-d) are
computed once per env step into a rolling per-env cache; PPO/BC updates run
the temporal encoder (RoPE transformer by default, oscillator bank with
--encoder osc) + readout + heads on cached feature windows. Full-backbone
PPO is infeasible at RL throughput on a single GPU.

NOTE: --encoder selects the temporal-encoder architecture. It must match
the checkpoint: new RL checkpoints are "tx" (default); pre-tx checkpoints
(oscillator bank) need --encoder osc. A mismatch loads nothing trainable
via the shape-filtered fallback and plays randomly.

Usage:
    .venv/Scripts/python student_rl.py --iters 2000
    .venv/Scripts/python student_rl.py --eval runs/student_rl/rl_latest.pt 20
    .venv/Scripts/python student_rl.py --eval runs/student_rl/rl_iter100.pt --encoder osc
"""
import argparse
import copy
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brawl_arena import BrawlArenaEnv, Config
from brawl_arena.core import ARCHETYPES, MODES, TILE_GOAL, Action
from student_vision import (OscillatorBank, RoPEFrameEncoder, AttentionPool,
                            CNNEncoder, build_two_scale_window, N_FRAMES,
                            STRIDE, SPAN, HIST_LEN, N_COARSE, TOK_TOTAL)
from student_omni import OmniStudent, IMG, DINO_DIM

OUT = os.path.join("runs", "student_rl")
INIT = "runs/student_vision/omni_ep37.pt"


# ------------------------------------------------------------------ model
class RLStudent(nn.Module):
    """Frozen DINO trunk (features precomputed) + CPG + actor-critic heads.

    move/aim heads output Gaussian means; shoot/super are Bernoulli logits;
    value head predicts P(team 0 wins) * 2 - 1."""

    def __init__(self, encoder: str = "tx", n_modes: int = 128,
                 backbone: str = "dino"):
        super().__init__()
        self.backbone_kind = backbone
        if backbone == "cnn":
            # trainable perception (stage-1 BC: detection + action losses),
            # frozen during RL. Same 384-d interface as the DINO trunk.
            self.cnn = CNNEncoder(DINO_DIM)
        else:
            self.backbone = torch.hub.load("facebookresearch/dinov2",
                                           "dinov2_vits14", pretrained=True)
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
        self.encoder = encoder
        if encoder == "tx":
            self.tx = RoPEFrameEncoder(DINO_DIM, 256)
            # two-scale context: attention-pools the 4096-step coarse
            # history into 128 tokens (see student_vision.AttentionPool);
            # absent from pre-two-scale checkpoints -> stays random-init
            # there until BC pretraining fills it in
            self.pool = AttentionPool(DINO_DIM, 256)
            self.readout = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        else:  # "osc": legacy oscillator bank (old checkpoint eval)
            self.osc = OscillatorBank(DINO_DIM, n_modes)
            self.readout = nn.Sequential(nn.Linear(2 * n_modes, 256), nn.ReLU())
        self.move_head = nn.Linear(256, 2)
        self.aim_head = nn.Linear(256, 2)
        self.shoot_head = nn.Linear(256, 2)
        self.super_head = nn.Linear(256, 2)
        self.pred_head = nn.Linear(256, DINO_DIM)   # next-feature prediction
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
        if self.backbone_kind == "cnn":
            f, _ = self.cnn(frames)
            return f
        v = frames.float() / 255.0
        v = (v - self.mean) / self.std
        v = F.interpolate(v, size=(IMG, IMG), mode="bilinear",
                          align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=frames.is_cuda):
            f = self.backbone.forward_features(v)["x_norm_clstoken"]
        return f.float()

    def _heads(self, z: torch.Tensor):
        return {"move_mu": torch.tanh(self.move_head(z)),
                "aim_mu": self.aim_head(z),
                "shoot": self.shoot_head(z),
                "super": self.super_head(z),
                "value": self.value_head(z.detach()).squeeze(-1)}

    def forward_feats(self, fw: torch.Tensor):
        """(B, T, 384) feature window -> action dists params + value."""
        enc = self.tx if self.encoder == "tx" else self.osc
        return self._heads(self.readout(enc(fw)))

    def forward_full(self, fw: torch.Tensor):
        """(B, T, 384) -> (out_dict, pred_seq).

        out_dict is exactly forward_feats(fw) (z from the last position).
        tx (two-scale, T=158): pred_seq = pred_head over the first 29 FINE
        tokens (target fw[:, -29:]) -- a coarse token's "next frame" is
        pooled history, so predicting it is meaningless. osc (legacy):
        pred at every position (target fw[:, 1:])."""
        enc = self.tx if self.encoder == "tx" else self.osc
        h = enc.forward_full(fw)
        if self.encoder == "tx":
            pred = self.pred_head(h[:, N_COARSE:-1])
        else:
            pred = self.pred_head(h[:, :-1])
        return self._heads(self.readout(h[:, -1])), pred

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
MARKER = (255, 0, 255)  # magenta ring = "this unit is you" (ALL policy views)


def _self_ring(img: np.ndarray, game, unit_idx: int = 0):
    """Paint a magenta ring on the driven unit + a fixed-position self panel
    (bottom-left: HP bar + ammo pips) so the policy can localize itself AND
    read its own state. This is a universal interface: every policy-facing
    frame (student rollout, eval, spectate BC, frozen rival) carries exactly
    one ring+panel on the unit that frame's actions will be applied to.
    Without the ring a policy driving several units issues one shared
    direction and never reaches gems (diagnosed 09-03: all-frozen teams
    scored 0 gems in 8/8 games). The panel exists because probe audits
    (09-04) showed hp/ammo are undecodable from over-head bars: detection of
    a 3px bar at an unknown location is hard even for a CNN, while a
    fixed-position readout is trivial."""
    u = game.units[unit_idx]
    if not u.alive:
        return
    sx = img.shape[1] / game.cfg.map_w
    sy = img.shape[0] / game.cfg.map_h
    px, py = int(u.pos[0] * sx), int(u.pos[1] * sy)
    r_out, r_in = max(2, int(0.75 * sx)), max(1, int(0.5 * sx))
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    d2 = (xx - px) ** 2 + (yy - py) ** 2
    img[(d2 <= r_out ** 2) & (d2 >= r_in ** 2)] = MARKER
    # fixed self panel, bottom-left: hp bar over 3 ammo pips
    h, w = img.shape[:2]
    frac = max(0.0, min(1.0, u.hp / ARCHETYPES[u.archetype]["hp"]))
    bw = min(40, w - 4)
    img[h - 8:h - 4, 2:2 + bw] = (60, 60, 60)
    img[h - 8:h - 4, 2:2 + int(bw * frac)] = (90, 220, 90)
    for k in range(3):
        c = (255, 200, 80) if u.ammo >= k + 0.5 else (70, 60, 45)
        img[h - 4:h, 2 + k * 5:5 + k * 5] = c


# -------------------------------------------------------- feature ring buffer
class FeatRing:
    """Preallocated CPU ring buffer of per-step features, replacing the
    per-stream deque of GPU tensors.

    (HIST_LEN, 384) storage + write pointer + valid length. window() builds
    the oldest-first history with at most two slice copies (wraparound)
    instead of stacking thousands of per-step tensors every call; history
    lives on CPU so it no longer pins VRAM (WDDM-paging risk on 8GB cards).

    Padding convention (unchanged from the deque path): when the valid
    history is short, missing steps are filled by repeating the OLDEST valid
    feature."""

    __slots__ = ("buf", "ptr", "len")

    def __init__(self, dim: int = 384):
        self.buf = torch.zeros(HIST_LEN, dim)
        self.ptr = 0                       # next write slot
        self.len = 0                       # valid feature count

    def append(self, f: torch.Tensor):
        self.buf[self.ptr] = f.cpu()
        self.ptr = (self.ptr + 1) % HIST_LEN
        if self.len < HIST_LEN:
            self.len += 1

    def _oldest_first(self) -> torch.Tensor:
        n = self.len
        start = (self.ptr - n) % HIST_LEN
        if start + n <= HIST_LEN:
            return self.buf[start:start + n]
        return torch.cat([self.buf[start:], self.buf[:start + n - HIST_LEN]])

    def history(self) -> torch.Tensor:
        """(len, 384) oldest-first valid features, unpadded."""
        return self._oldest_first()

    def window(self) -> torch.Tensor:
        """(HIST_LEN, 384) oldest-first CPU tensor, front-padded with the
        oldest valid feature (expand + one cat, no per-step fill loop)."""
        seq = self._oldest_first()
        pad = HIST_LEN - self.len
        if pad:
            seq = torch.cat([seq[:1].expand(pad, -1), seq])
        return seq


def _window(ring: FeatRing, net) -> torch.Tensor:
    """Build the encoder input from a per-stream FeatRing. tx: shared
    two-scale constructor (128 pooled coarse + 30 fine = 158 tokens; short
    histories front-padded with the oldest feature). osc (legacy
    checkpoints): the old fine-only 30-token window. Returns a CPU tensor;
    callers move it to the compute device."""
    if net.encoder == "tx":
        return build_two_scale_window(ring.window(), net.pool)[0]
    # legacy osc: stride the WHOLE history then keep the last 30 (the old
    # deque behaviour -- for histories > 118 steps this is NOT the last 59)
    seq = ring.history()
    if seq.shape[0] < SPAN + 1:
        seq = torch.cat([seq[:1].expand(SPAN + 1 - seq.shape[0], -1), seq])
    return seq[::STRIDE][-N_FRAMES:]


class FrozenStudentPolicy:
    """Wraps a frozen RLStudent as an opponent_policy (act_batch API).
    Drives each assigned unit from a viewer_team=<that unit's team> render
    with the self-marker ring on the driven unit. Own rolling feature cache
    per unit.

    Samples actions exactly like the training-time student (Normal move/aim
    with the ckpt's own log_std, Categorical buttons, tanh on the move
    sample). A deterministic argmax rival plays PASSIVELY -- 09-05 mirror
    diagnostic: same-ckpt duel 6/6 draws, zero kills in 300s -- which
    inflates self-play winrates (~0.9 in ko/duel) and idles the ratchet."""

    def __init__(self, ckpt: str, device: str, encoder: str = "tx",
                 backbone: str = "dino"):
        self.net = RLStudent(encoder=encoder, backbone=backbone)
        self.net.load_bc(ckpt)
        self.net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self._rings = {}       # unit idx -> FeatRing (CPU feature history)
        self._game_id = None

    def act_batch(self, game, idxs):
        if id(game) != self._game_id or game.t < 2:
            self._rings = {i: FeatRing() for i in idxs}
            self._game_id = id(game)
        frames = []
        for i in idxs:
            from brawl_arena.render import render
            img = render(game, viewer_team=game.units[i].team)
            if img.shape[0] != 90 or img.shape[1] != 126:
                t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
                t = F.interpolate(t, size=(90, 126), mode="bilinear",
                                  align_corners=False)
                img = t[0].permute(1, 2, 0).byte().numpy()
            _self_ring(img, game, i)
            frames.append(img)
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        feats = self.net.encode(x.to(self.device))
        acts = []
        for k, i in enumerate(idxs):
            ring = self._rings[i]
            ring.append(feats[k])
            fw = _window(ring, self.net)[None].to(self.device)
            out = self.net.forward_feats(fw)
            # sample like the student does in rollouts (see class docstring)
            sm = torch.exp(self.net.log_std_move).expand_as(out["move_mu"])
            sa = torch.exp(self.net.log_std_aim).expand_as(out["aim_mu"])
            am = torch.distributions.Normal(out["move_mu"], sm).sample()
            aa = torch.distributions.Normal(out["aim_mu"], sa).sample()
            shoot = torch.distributions.Categorical(logits=out["shoot"][0]).sample()
            sup = torch.distributions.Categorical(logits=out["super"][0]).sample()
            aim = aa[0].cpu().numpy()
            aim = aim / (np.linalg.norm(aim) + 1e-6)
            acts.append(Action(
                move=torch.tanh(am[0]).cpu().numpy().astype(np.float64),
                aim=aim.astype(np.float64),
                shoot=bool(shoot), use_super=bool(sup)))
        return acts


# ------------------------------------------------------------- letterboxing
def _lb(f: np.ndarray) -> np.ndarray:
    if f.shape[0] == 90 and f.shape[1] == 126:
        return f
    t = torch.from_numpy(f).permute(2, 0, 1)[None].float()
    t = F.interpolate(t, size=(90, 126), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).byte().numpy()


# ------------------------------------------------------- mid-game snapshots
SNAPSHOT_CAP = 200          # per-mode FIFO bucket size


def _snapshot_trigger(g) -> str | None:
    """Classify a high-value mid-game moment worth snapshotting, or None.

    One snapshot per class per episode (caller tracks); only taken while
    unit 0 is alive. Classes:
      gem_grab:  'countdown' -- a team first reaches the gem-win threshold
                 'lowhp'    -- unit 0 below 30% hp with an enemy nearby
      brawl_ball:'goalzone' -- ball within 5 tiles of a goal
      knockout:  'endgame'  -- 4 or fewer units alive
      duel:      'endgame'  -- round 2+ (deciding rounds)
      showdown:  'endgame'  -- 4 or fewer alive, or poison past half-way
    """
    u0 = g.units[0]
    if not u0.alive:
        return None
    if g.mode == "gem_grab":
        if g.countdown_team is not None:
            return "countdown"
        if u0.hp < 0.3 * ARCHETYPES[u0.archetype]["hp"]:
            near = any(u.alive and u.team != u0.team
                       and np.linalg.norm(u.pos - u0.pos) < 6.0
                       for u in g.units)
            if near:
                return "lowhp"
    elif g.mode == "brawl_ball":
        gy, gx = np.nonzero(g.tiles == TILE_GOAL)
        if len(gx):
            d = np.sqrt((gx + 0.5 - g.ball_pos[0]) ** 2
                        + (gy + 0.5 - g.ball_pos[1]) ** 2).min()
            if d < 5.0:
                return "goalzone"
    elif g.mode == "knockout":
        if sum(1 for u in g.units if u.alive) <= 4:
            return "endgame"
    elif g.mode == "duel":
        # round 2+: deciding rounds are the high-value duel states
        if g.round_num >= 2:
            return "endgame"
    elif g.mode == "showdown":
        alive = sum(1 for u in g.units if u.alive)
        if alive <= 4 or g.poison_radius <= 0.5 * g.poison_max:
            return "endgame"
    return None


# ----------------------------------------------------------------- training
def train(n_iters: int, device: str, init: str = INIT, horizon: int = 256,
          lr: float = 3e-5, bc_dagger: float = 1.0, warmup_iters: int = 30,
          target_kl: float = 0.02, ppo_epochs: int = 2,
          encoder: str = "tx", aux_fp: float = 0.1, mb_size: int = 256,
          snapshot_prob: float = 0.15, backbone: str = "dino"):
    os.makedirs(OUT, exist_ok=True)
    # v5.3: duo (2 LEARNING students vs 1 bot, knockout machinery) takes
    # over the vs_bots combat slots. 1v1-vs-bot duel gave no reachable
    # positive reward (0 kills, 1.9 hits/game, wr pinned 0.00 across two
    # versions); duo doubles the gradient streams per game AND gives each
    # student real kill/hit opportunities. ball/ko stay self-play only
    # (bot-carry channels); duel stays in the self-play mix.
    VS_SPECS = [("gem_grab", None), ("gem_grab", None),
                ("knockout", (2, 1)), ("knockout", (2, 1))]
    SP_MODES = ["gem_grab", "brawl_ball", "knockout", "duel"]
    roles = ["vs_bots"] * len(VS_SPECS) + ["selfplay"] * len(SP_MODES) \
        + ["spectate"]
    envs = []
    env_meta = []                # (role, mode, team_sizes) per env
    vi = si = 0
    for i, role in enumerate(roles):
        ts = None
        if role == "selfplay":
            mode = SP_MODES[si]
            si += 1
        elif role == "vs_bots":
            mode, ts = VS_SPECS[vi]
            vi += 1
        else:
            mode = "gem_grab"      # BC on the strategic bottleneck mode
        envs.append(BrawlArenaEnv(
            config=Config(mode=mode, team_sizes=ts, reward_shaping=True),
            seed=20000 + i, include_frame=True,
            n_students=ts[0] if ts else 1))
        env_meta.append((role, mode, ts))
    nslots = [e.n_students for e in envs]

    net = RLStudent(encoder=encoder, backbone=backbone).to(device)
    net.load_bc(init)
    rival = FrozenStudentPolicy(init, device, encoder=encoder,
                                backbone=backbone)
    oracle = {}                      # env idx -> scripted bot (DAgger labels)
    for i, (env, role) in enumerate(zip(envs, roles)):
        if role == "selfplay":
            # stash the scripted bot BEFORE overriding: it stays the DAgger
            # oracle even though the frozen policy now drives its teammates
            oracle[i] = env.teammate_policy
            # symmetric self-play: the frozen policy drives ALL 5 non-student
            # units (env.step groups by policy identity, so both hooks get
            # one shared act_batch call per step)
            env.opponent_policy = rival
            env.teammate_policy = rival

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
    results["duo"] = deque(maxlen=100)   # 2v1 handicap (knockout machinery)
    sp_results = deque(maxlen=200)   # selfplay: student vs frozen self
    sp_mode = {m: deque(maxlen=100) for m in SP_MODES}  # per-mode breakdown
    last_promote = 0

    # mid-game snapshot buffers (AgentENV-style: restart episodes from
    # high-value states). Per-mode FIFO buckets of deepcopied Game objects;
    # snap_taken[i] dedupes trigger classes within an episode; snap_stat is
    # [restored, total] reset counts for the current iter's log line.
    snapshots = {m: deque(maxlen=SNAPSHOT_CAP) for m in MODES}
    snap_taken = [set() for _ in envs]
    snap_stat = [0, 0]

    def reset_env(i):
        env = envs[i]
        snap_stat[1] += 1
        bucket = snapshots[env.game.mode]
        # duo envs skip snapshot restore: their 2v1 states would break the
        # unit-count assumptions of the single-student buckets
        if roles[i] != "spectate" and nslots[i] == 1 and bucket \
                and random.random() < snapshot_prob:
            # swap in a FRESH deepcopy: id(game) changes, which resets the
            # frozen rival's per-unit feature cache automatically
            env.set_game(copy.deepcopy(random.choice(list(bucket))))
            snap_stat[0] += 1
            o = env._obs()
        else:
            o, _ = env.reset()
        snap_taken[i] = set()
        base = _lb(o["frame"])
        frames = []
        for s in range(nslots[i]):
            f = base.copy()
            _self_ring(f, env.game, s)   # ring on THIS student's unit
            frames.append(f)
        obs_frames[i] = frames
        feats[i] = [FeatRing() for _ in range(nslots[i])]
        ep_ret[i] = 0.0

    for i in range(len(envs)):
        reset_env(i)

    student_envs = [i for i, r in enumerate(roles) if r != "spectate"]

    t_start = time.perf_counter()
    total_steps = 0
    for it in range(1, n_iters + 1):
        # ---------------- rollout (per-env buffers keep GAE intact) --------
        bufs = {(i, s): {k: [] for k in ("fw", "am", "aa", "btns", "logp",
                                         "val", "rew", "done",
                                         "dmv", "dam", "dbt")}
                for i in student_envs for s in range(nslots[i])}
        spec = {k: [] for k in ("fw", "mv", "am", "bt")}
        snap_stat[0] = snap_stat[1] = 0
        for t in range(horizon):
            # one batched backbone forward for all fresh frames (duo envs
            # contribute one frame per student slot)
            flat = [(i, s) for i in range(len(envs))
                    for s in range(nslots[i])]
            x = torch.from_numpy(np.stack(
                [obs_frames[i][s] for i, s in flat])).permute(0, 3, 1, 2)
            f_new = net.encode(x.to(device))
            for k, (i, s) in enumerate(flat):
                feats[i][s].append(f_new[k])

            for i, role in enumerate(roles):
                game = envs[i].game
                if role == "spectate":
                    fw = _window(feats[i][0], net)
                    bot = envs[i].teammate_policy.act(game, 0)
                    spec["fw"].append(fw)
                    spec["mv"].append(torch.from_numpy(
                        np.asarray(bot.move, dtype=np.float32)))
                    a = np.asarray(bot.aim, dtype=np.float32)
                    spec["am"].append(torch.from_numpy(
                        a / (np.linalg.norm(a) + 1e-6)))
                    spec["bt"].append((int(bot.shoot), int(bot.use_super)))
                    act_in = {"move": bot.move.astype(np.float32),
                              "aim": bot.aim.astype(np.float32),
                              "shoot": int(bot.shoot),
                              "super": int(bot.use_super)}
                else:
                    # one sampled action per student slot (duo: two streams)
                    act_list = []
                    for s in range(nslots[i]):
                        b = bufs[(i, s)]
                        fw = _window(feats[i][s], net)
                        out, am, aa, btns, logp, ent = net.sample(
                            fw[None].to(device))
                        # DAgger oracle label on each student's own state; in
                        # selfplay envs teammate_policy was overridden by the
                        # frozen rival, so use the stashed scripted bot
                        bot = oracle.get(i, envs[i].teammate_policy).act(
                            game, s)
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
                        act_list.append({
                            "move": torch.tanh(am[0]).cpu().numpy()
                                    .astype(np.float32),
                            "aim": aim.astype(np.float32),
                            "shoot": int(btns[0]),
                            "super": int(btns[1])})
                    act_in = act_list if nslots[i] > 1 else act_list[0]
                o, r, done, _, info = envs[i].step(act_in)
                base = _lb(o["frame"])
                for s in range(nslots[i]):
                    f = base.copy()
                    _self_ring(f, envs[i].game, s)
                    obs_frames[i][s] = f
                if role != "spectate":
                    rs = r if isinstance(r, list) else [r]
                    for s in range(nslots[i]):
                        bufs[(i, s)]["rew"].append(rs[s])
                        bufs[(i, s)]["done"].append(float(done))
                    ep_ret[i] += sum(rs)
                    # snapshot high-value mid-game states (events were
                    # already consumed by _reward inside step, so the copy
                    # restores cleanly). Single-student envs only: duo
                    # states would break the buckets' unit-count assumptions.
                    if not done and nslots[i] == 1:
                        trig = _snapshot_trigger(game)
                        if trig is not None and trig not in snap_taken[i]:
                            snapshots[game.mode].append(copy.deepcopy(game))
                            snap_taken[i].add(trig)
                if done:
                    if info["winner"] is not None:
                        sc = (1.0 if info["winner"] == 0 else
                              0.5 if info["winner"] not in (0, 1) else 0.0)
                        if role == "vs_bots":
                            _, m_, ts_ = env_meta[i]
                            results["duo" if ts_ else m_].append(sc)
                        elif role == "selfplay":
                            sp_results.append(sc)
                            sp_mode[info["mode"]].append(sc)
                    elif role == "selfplay":
                        # a draw (0-0 stalemate) is a real outcome: counting
                        # it as 0.5 keeps draw-heavy stretches from inflating
                        # the ratchet's winrate and fills the gate 2-3x faster
                        sp_results.append(0.5)
                        sp_mode[info["mode"]].append(0.5)
                    reset_env(i)
            total_steps += len(envs)

        # ---------------- PPO update on cached features ----------------
        # bootstrap value for the final state of each stream's horizon
        last_vals = {}
        for i in student_envs:
            for s in range(nslots[i]):
                ring_ = feats[i][s]
                if ring_.len == 0:   # episode ended on the final step
                    x = torch.from_numpy(
                        obs_frames[i][s]).permute(2, 0, 1)[None]
                    ring_.append(net.encode(x.to(device))[0])
                fw_last = _window(ring_, net)
                with torch.no_grad():
                    last_vals[(i, s)] = float(net.forward_feats(
                        fw_last[None].to(device))["value"])

        fw_l, am_l, aa_l, btns_l, logp_l, adv_l, ret_l = [], [], [], [], [], [], []
        dmv_l, dam_l, dbt_l = [], [], []
        for (i, s), b in bufs.items():
            T_i = len(b["rew"])
            rew = np.array(b["rew"], dtype=np.float32)
            don = np.array(b["done"], dtype=np.float32)
            val = np.array(b["val"] + [last_vals[(i, s)]], dtype=np.float32)
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
        fw = torch.stack(fw_l).to(device)            # (N,158,384) two-scale
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
        pl = vl = bl = fpl = 0.0
        nfp = 0
        nup = 0
        kl_stop = False
        for ep in range(ppo_epochs):
            if kl_stop:
                break
            np.random.shuffle(idx_all)
            kl_acc = 0.0
            for b0 in range(0, n_std, mb_size):
                mb = idx_all[b0:b0 + mb_size]
                out, pred_seq = net.forward_full(fw[mb])
                # pred_seq covers the 29 fine-token pairs (tx) or all
                # positions (legacy osc); slice the target to match
                loss_fp = F.mse_loss(pred_seq,
                                     fw[mb][:, -pred_seq.shape[1]:].detach())
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
                    loss = VF * loss_v + aux_fp * loss_fp
                else:
                    loss = (loss_pi + VF * loss_v - ENT * ent.mean()
                            + aux_fp * loss_fp)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 0.5)
                opt.step()
                pl += float(loss_pi)
                vl += float(loss_v)
                fpl += float(loss_fp)
                nfp += 1
                nup += 1
            if kl_acc / max(1, len(range(0, n_std, mb_size))) > target_kl:
                kl_stop = True
        # BC passes (spectate + DAgger), one epoch
        for src_fw, src_mv, src_am, src_bt, w in (
                (s_fw, s_mv, s_am, s_bt, BC_SPECT),
                (fw, d_mv, d_am, d_bt, BC_DAGGER)):
            nsrc = len(src_fw)
            order = np.random.permutation(nsrc)
            for b0 in range(0, nsrc, mb_size):
                mb = order[b0:b0 + mb_size]
                out, pred_seq = net.forward_full(src_fw[mb])
                loss_fp = F.mse_loss(
                    pred_seq, src_fw[mb][:, -pred_seq.shape[1]:].detach())
                loss = (w * (F.mse_loss(out["move_mu"], src_mv[mb])
                             + F.mse_loss(out["aim_mu"], src_am[mb])
                             + F.cross_entropy(out["shoot"], src_bt[mb][:, 0])
                             + F.cross_entropy(out["super"], src_bt[mb][:, 1]))
                        + aux_fp * loss_fp)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 0.5)
                opt.step()
                bl += float(loss)
                fpl += float(loss_fp)
                nfp += 1

        if it % 5 == 0:
            fps = total_steps / (time.perf_counter() - t_start)
            wr = {m: (np.mean(d) if d else float("nan"))
                  for m, d in results.items()}
            sp = np.mean(sp_results) if sp_results else float("nan")
            spm = " ".join(
                f"{m.split('_')[0][:4]} "
                f"{(np.mean(d) if d else float('nan')):.2f}"
                for m, d in sp_mode.items())
            print(f"[it {it}] steps {total_steps} ({fps:.0f}/s) "
                  f"pi {pl / nup:.4f} v {vl / nup:.4f} bc {bl:.3f} "
                  f"fp {fpl / max(nfp, 1):.4f} "
                  f"wr gem {wr['gem_grab']:.2f} duo {wr['duo']:.2f} "
                  f"selfplay {sp:.2f} [{spm}] "
                  f"snap {snap_stat[0]}/{snap_stat[1]}", flush=True)
        if it % 25 == 0:
            torch.save(net.state_dict(), os.path.join(OUT, "rl_latest.pt"))
            torch.save(net.state_dict(),
                       os.path.join(OUT, f"rl_iter{it}.pt"))
        # gated ratchet: promote the current net to the frozen rival once
        # self-play winrate over the recent window clears the bar
        if (it > warmup_iters and len(sp_results) >= 60
                and it - last_promote >= 50):
            sp_mean = float(np.mean(sp_results))
            if sp_mean >= 0.60:
                src = os.path.join(OUT, "frozen_src.pt")
                torch.save(net.state_dict(),
                           os.path.join(OUT, f"promote_it{it}.pt"))
                torch.save(net.state_dict(), src)
                rival = FrozenStudentPolicy(src, device, encoder=encoder,
                                            backbone=backbone)
                for i, role in enumerate(roles):
                    if role == "selfplay":
                        envs[i].opponent_policy = rival
                        envs[i].teammate_policy = rival
                sp_results.clear()
                last_promote = it
                print(f"[promote] it {it} wr {sp_mean:.2f}", flush=True)
    torch.save(net.state_dict(), os.path.join(OUT, "rl_latest.pt"))
    print("done")


# ------------------------------------------------------------------- eval
def evaluate(ckpt: str, n: int, mode: str = "gem_grab", seed0: int = 7000,
             encoder: str = "tx", backbone: str = "dino",
             sampled: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = RLStudent(encoder=encoder, backbone=backbone)
    try:
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    except RuntimeError:
        net.load_bc(ckpt)   # OmniStudent-era checkpoint (shape-filtered)
    net.eval().to(device)
    wins = draws = losses = 0
    gem_diffs = []
    for seed in range(n):
        env = BrawlArenaEnv(config=Config(mode=mode), seed=seed0 + seed,
                            include_frame=True)
        obs, _ = env.reset(seed=seed0 + seed)
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
                # rollout-time policy: sample like training does. argmax
                # plays measurably more passive (09-05 mirror-duel
                # diagnostic: same-ckpt duel 6/6 draws, zero kills)
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
            obs, _, term, _, info = env.step({
                "move": move, "aim": aim.astype(np.float32),
                "shoot": shoot, "super": sup})
            done = term
        w = info["winner"]
        wins += w == 0
        losses += w == 1
        draws += w not in (0, 1)
        if "gems" in info:
            gem_diffs.append(info["gems"][0] - info["gems"][1])
    gd = f", avg gem diff {np.mean(gem_diffs):+.1f}" if gem_diffs else ""
    tag = ", sampled" if sampled else ""
    print(f"{ckpt} [{mode}{tag}]: over {n} games W{wins} D{draws} L{losses}{gd}")


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
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--aux-fp", type=float, default=0.1,
                    help="weight of the next-feature prediction aux loss")
    ap.add_argument("--encoder", choices=("tx", "osc"), default="tx",
                    help="temporal encoder; must match the checkpoint "
                         "(pre-tx checkpoints need --encoder osc)")
    ap.add_argument("--backbone", choices=("dino", "cnn"), default="dino",
                    help="per-frame perception; must match the checkpoint "
                         "(cnn = trainable CNNEncoder from stage-1 BC)")
    ap.add_argument("--eval-sampled", action="store_true",
                    help="eval with the rollout-time sampled policy instead "
                         "of argmax (off-chain diagnostic; the eval chain "
                         "stays argmax for comparability)")
    ap.add_argument("--snapshot-prob", type=float, default=0.15,
                    help="probability that a student-env episode restarts "
                         "from a mid-game snapshot instead of a fresh reset")
    ap.add_argument("--mb-size", type=int, default=256,
                    help="PPO/BC minibatch size; 256 keeps the 8GB laptop "
                         "card clear of WDDM paging at T=158")
    args = ap.parse_args()
    if args.eval:
        evaluate(args.eval, args.n, args.mode, args.seed0, args.encoder,
                 backbone=args.backbone, sampled=args.eval_sampled)
    if args.iters:
        train(args.iters, args.device, args.init, horizon=args.horizon,
              lr=args.lr, bc_dagger=args.bc_dagger,
              warmup_iters=args.warmup_iters, ppo_epochs=args.ppo_epochs,
              encoder=args.encoder, aux_fp=args.aux_fp,
              snapshot_prob=args.snapshot_prob, mb_size=args.mb_size,
              backbone=args.backbone)


if __name__ == "__main__":
    main()
