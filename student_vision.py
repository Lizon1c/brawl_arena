"""Student vision prototype: DINOv2-S/14 backbone + CPG-style oscillator bank.

Pipeline per user spec:
  frames -> DINOv2 ViT-S/14 (fully fine-tuned) -> per-frame 384-d feature
         -> oscillator bank (LRU-like: K damped rotation modes over the last
            T frames; population readout = the CPG-style motion encoding)
         -> 4 action heads (move 9 | aim 9 | shoot 2 | super 2)

Stages:
  --collect N   scripted bot drives unit 0 across all modes; frames resized
                to 126x126 uint8 and stored with oracle actions + episode
                boundaries (windows are gathered at train time, no dup)
  --train E     behaviour cloning on 16-frame windows (stride 2 = 32 game
                steps of context), bf16 autocast, checkpoint per epoch

Usage:
    .venv/Scripts/python student_vision.py --collect 40000
    .venv/Scripts/python student_vision.py --train 8
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brawl_arena import BrawlArenaEnv, Config
from brawl_arena.core import MODES
from pretrain_teacher import action_to_multi
from distill_student import FLIP_LUT, split_logits

ACTION_DIMS = (9, 9, 2, 2)
IMG = 126          # DINOv2-S/14: 126/14 = 9 patches per side
DINO_DIM = 384
DATA = os.path.join("runs", "student_vision", "bc_data.npz")
OUT = os.path.join("runs", "student_vision")


# ------------------------------------------------------------------ model
class OscillatorBank(nn.Module):
    """K damped 2D rotation modes driven by the input (LRU-flavoured CPG).

    Each mode k rotates its state at a fixed learnable frequency theta_k and
    decays with learnable radius r_k in (0, 1); the motion readout is the
    population vector of all modes, mirroring CPG population coding.
    Frequencies are initialised log-spaced so the bank covers timescales
    from ~1 frame to the full window (wavelet-like scale coverage)."""

    def __init__(self, d_in: int = DINO_DIM, n_modes: int = 128):
        super().__init__()
        self.n_modes = n_modes
        self.log_r = nn.Parameter(torch.zeros(n_modes))
        t = torch.linspace(0, 1, n_modes)
        self.theta = nn.Parameter(0.05 * (2.8 / 0.05) ** t)  # 0.05..2.8 rad
        self.in_proj = nn.Linear(d_in, 2 * n_modes)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, 2K) population state at every position."""
        B, T, _ = x.shape
        K = self.n_modes
        r = torch.sigmoid(self.log_r) * 0.999
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        h = x.new_zeros(B, 2 * K)
        hs = []
        for t in range(T):
            u = self.in_proj(x[:, t])
            re, im = h[:, :K], h[:, K:]
            h = torch.cat([r * (re * c - im * s) + u[:, :K],
                           r * (re * s + im * c) + u[:, K:]], dim=1)
            hs.append(h)
        return torch.stack(hs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, 2K) population state after the last frame."""
        return self.forward_full(x)[:, -1]


class _TxBlock(nn.Module):
    """Pre-LN transformer block; RoPE is applied by the caller (the encoder
    owns the position table) via the `rope(q, k)` callback."""

    def __init__(self, d_model: int, nhead: int, d_ff: int):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x, rope):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.nhead, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        q, k = rope(q, k)                       # (B, H, T, hd)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


# ---------------------------------------------------- two-scale window spec
# NSA-style fine+coarse context. Fine window (unchanged): the last SPAN+1
# raw features strided by STRIDE -> N_FRAMES tokens covering 58 steps (~2 s).
# Coarse window (new): the COARSE_STEPS features before that, in N_COARSE
# blocks of COARSE_BLOCK steps, attention-pooled to one token per block ->
# 4096 steps (~136 s, most of a match). Total sequence = 128 + 30 = 158.
N_FRAMES, STRIDE = 30, 2
SPAN = (N_FRAMES - 1) * STRIDE                  # 58
N_COARSE, COARSE_BLOCK = 128, 32
COARSE_STEPS = N_COARSE * COARSE_BLOCK          # 4096
HIST_LEN = COARSE_STEPS + SPAN + 1              # 4155: features kept per stream
TOK_TOTAL = N_COARSE + N_FRAMES                 # 158


def two_scale_positions() -> torch.Tensor:
    """RoPE position (real step offset within the retained history, index 0
    = oldest retained step) of every token in the 158-token sequence:
      coarse token k -> midpoint of its 32-step block: k*32 + 16
      fine token j   -> its exact step: COARSE_STEPS + j*STRIDE (4096..4154)
    Strictly increasing; the layout is fixed, short histories are padded
    content-wise (oldest feature repeated), not position-wise."""
    coarse = torch.arange(N_COARSE) * COARSE_BLOCK + COARSE_BLOCK // 2
    fine = COARSE_STEPS + torch.arange(N_FRAMES) * STRIDE
    return torch.cat([coarse, fine]).float()


class RoPEFrameEncoder(nn.Module):
    """Causal transformer over per-frame DINO features with 1D RoPE.

    (B, T, 384) -> Linear to d_model -> n_layers pre-LN blocks (multi-head
    attention with rotary position embeddings on q/k under a causal mask,
    GELU FFN) -> final LayerNorm. RoPE rotates the FULL head dim using the
    half-split convention: (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos)
    with theta_i = 10000^(-2i/head_dim).

    Two-scale input: T == TOK_TOTAL (158) is the transport format produced
    by two_scale_tokens/build_two_scale_window -- rows 0..127 hold the
    attention-pooled coarse tokens (256-d, in columns 0..255), rows 128..157
    hold the raw 384-d fine features (projected here by in_proj). Any other
    T is a plain raw-feature sequence (legacy fine-only windows).

    forward(x, pos)      -> last token hidden (B, d_model)
    forward_full(x, pos) -> hidden at every position (B, T, d_model), for the
                            next-feature prediction auxiliary loss.
    pos: explicit RoPE positions (real step offsets); None uses the
    two-scale layout for T == 158, else arange(T) (the old behaviour)."""

    def __init__(self, d_in: int = DINO_DIM, d_model: int = 256,
                 nhead: int = 4, n_layers: int = 4, d_ff: int = 1024):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.in_proj = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList(
            [_TxBlock(d_model, nhead, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float()
                                 / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.register_buffer("pos_two_scale", two_scale_positions(),
                             persistent=False)

    def _rope(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor):
        """Rotate q/k (B, H, T, hd) by the given per-token positions (T,)."""
        pos = pos.to(device=q.device, dtype=q.dtype)
        freqs = torch.outer(pos, self.inv_freq.to(q.dtype))   # (T, hd/2)
        cos, sin = freqs.cos(), freqs.sin()

        def rot(x):
            x1, x2 = x[..., :self.head_dim // 2], x[..., self.head_dim // 2:]
            return torch.cat([x1 * cos - x2 * sin,
                              x1 * sin + x2 * cos], dim=-1)
        return rot(q), rot(k)

    def forward_full(self, x: torch.Tensor,
                     pos: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, d_model) hidden at every position."""
        B, T, _ = x.shape
        if T == TOK_TOTAL:
            # two-scale transport: coarse rows are already pooled to d_model
            h = torch.cat([x[:, :N_COARSE, :self.in_proj.out_features],
                           self.in_proj(x[:, N_COARSE:])], dim=1)
            if pos is None:
                pos = self.pos_two_scale
        else:
            h = self.in_proj(x)
            if pos is None:
                pos = torch.arange(T, device=x.device, dtype=torch.float)
        for blk in self.layers:
            h = blk(h, lambda q, k: self._rope(q, k, pos))
        return self.ln_f(h)

    def forward(self, x: torch.Tensor,
                pos: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D) -> (B, d_model) hidden of the last frame."""
        return self.forward_full(x, pos)[:, -1]


class AttentionPool(nn.Module):
    """Perceiver-style pooling of the coarse history: n_blocks learned
    queries, ONE PER BLOCK (each 32-step block has distinct temporal
    semantics, so it gets its own query). q_k attends over its block's raw
    features; the weighted sum (LayerNorm'ed) becomes coarse token k.

    Output width = d_model, so coarse tokens join the transformer sequence
    exactly where the fine tokens land after the input projection.

    blocks: (B, n_blocks, block_len, d_in) -> (B, n_blocks, d_out).
    Fully vectorized: one batched attention over all 128 blocks, no
    per-block python loop."""

    def __init__(self, d_in: int = DINO_DIM, d_out: int = 256,
                 n_blocks: int = N_COARSE):
        super().__init__()
        self.n_blocks = n_blocks
        self.queries = nn.Parameter(torch.randn(n_blocks, d_out) * 0.02)
        self.k_proj = nn.Linear(d_in, d_out)
        self.v_proj = nn.Linear(d_in, d_out)
        self.ln = nn.LayerNorm(d_out)

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        k = self.k_proj(blocks)                        # (B, L, C, d)
        v = self.v_proj(blocks)
        att = torch.einsum("ld,blcd->blc", self.queries, k)
        w = (att / k.shape[-1] ** 0.5).softmax(dim=-1)
        return self.ln(torch.einsum("blc,blcd->bld", w, v))


def two_scale_tokens(feats: torch.Tensor, pool: AttentionPool) -> torch.Tensor:
    """(B, L, 384) real feature histories (oldest first, L <= HIST_LEN) ->
    (B, 158, 384) transport tensor for RoPEFrameEncoder: rows 0..127 hold
    the pooled coarse tokens in columns 0..255 (rest zero), rows 128..157
    hold the 30 raw fine features. Short histories are front-padded by
    repeating the oldest available feature (the existing convention -- no
    mask, no new mechanism)."""
    feats = feats[:, -HIST_LEN:]
    pad = HIST_LEN - feats.shape[1]
    if pad:
        feats = torch.cat([feats[:, :1].expand(-1, pad, -1), feats], dim=1)
    fine = feats[:, COARSE_STEPS::STRIDE]              # (B, 30, 384)
    blocks = feats[:, :COARSE_STEPS].reshape(
        feats.shape[0], N_COARSE, COARSE_BLOCK, feats.shape[-1])
    coarse = pool(blocks)                              # (B, 128, 256)
    out = feats.new_zeros(feats.shape[0], TOK_TOTAL, feats.shape[-1])
    out[:, :N_COARSE, :coarse.shape[-1]] = coarse
    out[:, N_COARSE:] = fine
    return out


def build_two_scale_window(feat_list, pool: AttentionPool):
    """Shared window constructor for every rollout/eval call site.
    feat_list: (L, 384) tensor (e.g. FeatRing.window() output) or legacy
    deque/list of (384,) features, oldest first (<= HIST_LEN entries).
    Returns (tokens (158, 384), positions (158,)) on the input's device.

    Pooling runs on the pool's device: callers keep feature history on CPU
    (GPU-resident deques of per-step tensors cost thousands of kernel
    launches per step and pin gigabytes of VRAM) and pay one H2D copy here.

    no_grad: rollout-time pooling must not build a graph -- the returned
    tensor is stored in the PPO/BC buffers and replayed later. The pool is
    trained by the BC pretrain path (two_scale_tokens keeps grad there);
    during RL it acts as a fixed compressor, like the frozen DINO trunk."""
    if torch.is_tensor(feat_list):
        feats = feat_list
    else:
        feats = torch.stack(list(feat_list))
    dev = next(pool.parameters()).device
    with torch.no_grad():
        tokens = two_scale_tokens(feats[None].to(dev), pool)[0]
    return tokens.to(feats.device), two_scale_positions().to(feats.device)


class CNNEncoder(nn.Module):
    """Trainable per-frame encoder replacing the frozen DINO trunk.

    (B, 3, 90, 126) uint8-range -> conv stack -> (B, 96, 12, 16) spatial map:
      - pooled frame feature: GAP -> Linear -> DINO_DIM (keeps the 384-d
        per-frame interface, so FeatRing / two-scale window / RoPE trunk /
        heads / PPO are all untouched)
      - detection head: 96->64 3x3 -> 64->6 1x1 grid predictions over the
        12x16 cell grid (YOLO-style, no anchors/NMS: <=6 units, labels are
        free from the sim). Channels: [presence, team, dx, dy, hp_frac,
        item(gem/ball)].

    Probe audit 09-04: this inductive bias reads self pos R2 .94/.93, enemy
    rel pos +0.4-0.5 and (with the fixed self panel) hp .99 / ammo .85 --
    all undecodable from frozen DINOv2 CLS features (R2 < 0)."""

    def __init__(self, feat_dim: int = DINO_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2), nn.ReLU(),      # 45x63
            nn.Conv2d(24, 48, 3, 2, 1), nn.ReLU(),     # 23x32
            nn.Conv2d(48, 96, 3, 2, 1), nn.ReLU(),     # 12x16
            nn.Conv2d(96, 96, 3, 1, 1), nn.ReLU())
        self.feat_proj = nn.Linear(96, feat_dim)
        self.det1 = nn.Conv2d(96, 64, 3, 1, 1)
        self.det2 = nn.Conv2d(64, 6, 1)

    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, W) uint8-range -> (feat (B, 384), det (B, 6, 12, 16)).
        NOT under no_grad: stage-1 BC trains this; RLStudent.encode wraps it
        in no_grad for rollout (frozen compressor, same role as DINO had)."""
        v = x.float() / 255.0 - 0.5
        h = self.conv(v)
        feat = self.feat_proj(F.adaptive_avg_pool2d(h, 1).flatten(1))
        det = self.det2(F.relu(self.det1(h)))
        return feat, det


class DinoCPGStudent(nn.Module):
    """DINOv2-S/14 (fine-tuned) per frame + oscillator bank + heads."""

    def __init__(self, n_modes: int = 128, n_frames: int = 16):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2",
                                       "dinov2_vits14", pretrained=True)
        self.n_frames = n_frames
        self.osc = OscillatorBank(DINO_DIM, n_modes)
        self.readout = nn.Sequential(nn.Linear(2 * n_modes, 256), nn.ReLU())
        self.heads = nn.ModuleList(
            [nn.Linear(256, d) for d in ACTION_DIMS])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) uint8-range floats [0,255] -> (B, 22) logits."""
        B, T = x.shape[:2]
        v = x.reshape(B * T, *x.shape[2:]) / 255.0
        v = (v - self.mean) / self.std
        v = F.interpolate(v, size=(IMG, IMG), mode="bilinear",
                          align_corners=False)
        feats = self.backbone.forward_features(v)["x_norm_clstoken"]
        feats = feats.reshape(B, T, DINO_DIM)
        h = self.osc(feats)
        z = self.readout(h)
        return torch.cat([hd(z) for hd in self.heads], dim=1)


# --------------------------------------------------------------- collection
def collect(n: int, seed: int = 42):
    os.makedirs(OUT, exist_ok=True)
    frames = np.zeros((n, 90, 126, 3), dtype=np.uint8)
    acts = np.zeros((n, 4), dtype=np.int64)
    ep_start = np.zeros(n, dtype=bool)
    i = 0
    t0 = time.perf_counter()
    for round_ in range(9999):
        for mode in MODES:
            if i >= n:
                break
            env = BrawlArenaEnv(config=Config(mode=mode),
                                seed=seed + round_, include_frame=True)
            obs, _ = env.reset(seed=seed + round_)
            game = env.game
            ep_start[i] = True
            done = False
            while not done and i < n:
                bot = env.teammate_policy.act(game, 0)
                f = obs["frame"]
                if f.shape[0] != 90 or f.shape[1] != 126:
                    # showdown maps are 114x150: letterbox into 90x126
                    ff = torch.from_numpy(f).permute(2, 0, 1)[None].float()
                    ff = F.interpolate(ff, size=(90, 126), mode="bilinear",
                                       align_corners=False)
                    f = ff[0].permute(1, 2, 0).byte().numpy()
                frames[i] = f
                acts[i] = action_to_multi(bot.move, bot.aim, bot.shoot,
                                          bot.use_super)
                obs, _, terminated, _, _ = env.step(
                    {"move": bot.move.astype(np.float32),
                     "aim": bot.aim.astype(np.float32),
                     "shoot": int(bot.shoot), "super": int(bot.use_super)})
                done = terminated
                i += 1
                if i % 5000 == 0:
                    print(f"  collected {i}/{n} "
                          f"({i / (time.perf_counter() - t0):.0f}/s)")
        if i >= n:
            break
    np.savez_compressed(DATA, frames=frames, acts=acts, ep_start=ep_start)
    print(f"saved {DATA}: {len(frames)} frames")


# ----------------------------------------------------------------- training
def make_windows(idx: np.ndarray, ep_bounds: np.ndarray, n_frames: int,
                 stride: int):
    """For each sample i return frame indices [i-(T-1)*stride .. i], clamped
    below at its episode start (front-padded by repeating the first valid
    frame). ep_bounds: sorted array of episode start positions."""
    starts = ep_bounds[np.searchsorted(ep_bounds, idx, side="right") - 1]
    lags = np.arange(n_frames - 1, -1, -1) * stride       # (T,)
    j = idx[:, None] - lags[None, :]                      # (B, T)
    return np.maximum(j, starts[:, None])


def train(n_epochs: int, device: str, batch_size: int = 8,
          n_frames: int = 16, stride: int = 2):
    d = np.load(DATA)
    frames, acts, ep_start = d["frames"], d["acts"], d["ep_start"]
    x_all = torch.from_numpy(frames)            # uint8 (N, 90, 126, 3), cpu
    y_all = torch.from_numpy(acts)
    n = len(frames)
    ep_bounds = np.flatnonzero(ep_start)
    print(f"dataset: {n} frames, window={n_frames}x stride={stride}")

    net = DinoCPGStudent(n_frames=n_frames).to(device)
    # differential lr: backbone gentle, new modules aggressive
    opt = torch.optim.AdamW([
        {"params": net.backbone.parameters(), "lr": 1e-5},
        {"params": (list(net.osc.parameters())
                    + list(net.readout.parameters())
                    + list(net.heads.parameters())), "lr": 1e-3},
    ])
    n_val = max(1, n // 10)
    perm = np.random.RandomState(0).permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    for ep in range(n_epochs):
        net.train()
        order = np.random.permutation(len(tr_idx))
        tot, nb = 0.0, 0
        t0 = time.perf_counter()
        for b in range(0, len(order), batch_size):
            idx = tr_idx[order[b:b + batch_size]]
            widx = make_windows(idx, ep_bounds, n_frames, stride)
            x = x_all[torch.from_numpy(widx)]          # (B,T,90,126,3)
            x = x.permute(0, 1, 4, 2, 3).float().to(device)
            y = y_all[idx].clone()
            flip = torch.rand(len(idx)) < 0.5
            if flip.any():
                x[flip] = torch.flip(x[flip], dims=[4])
                y[flip, 0] = FLIP_LUT[y[flip, 0]]
                y[flip, 1] = FLIP_LUT[y[flip, 1]]
            y = y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(x)
                loss = sum(F.cross_entropy(lg, y[:, k])
                           for k, lg in enumerate(split_logits(logits)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
            if nb % 200 == 0:
                print(f"  ep {ep + 1} batch {nb} loss {tot / nb:.4f} "
                      f"({nb * batch_size / (time.perf_counter() - t0):.0f} "
                      f"samples/s)")
        # validation per head
        net.eval()
        correct = np.zeros(4)
        with torch.no_grad():
            for b in range(0, len(val_idx), 16):
                idx = val_idx[b:b + 16]
                widx = make_windows(idx, ep_bounds, n_frames, stride)
                x = x_all[torch.from_numpy(widx)]
                x = x.permute(0, 1, 4, 2, 3).float().to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = net(x)
                for k, lg in enumerate(split_logits(pred)):
                    correct[k] += (lg.argmax(1).cpu()
                                   == y_all[idx, k]).sum()
        acc = correct / len(val_idx)
        print(f"[epoch {ep + 1}/{n_epochs}] loss {tot / max(nb, 1):.4f} "
              f"val acc move {acc[0]:.3f} aim {acc[1]:.3f} "
              f"shoot {acc[2]:.3f} super {acc[3]:.3f} "
              f"({(time.perf_counter() - t0) / 60:.1f} min)")
        torch.save(net.state_dict(),
                   os.path.join(OUT, f"dino_cpg_ep{ep + 1}.pt"))
    print("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=int, default=0)
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    if args.collect:
        collect(args.collect)
    if args.train:
        train(args.train, args.device, args.batch_size)


if __name__ == "__main__":
    main()
