"""Smoke tests for the two-scale (fine+coarse) student window (CPU).

1. window construction: token count, monotonic positions, padding matches
   the old front-pad convention exactly
2. encoder: shapes, causality, explicit RoPE positions take effect
3. old checkpoint compat: load_bc(ckpt_v32_it45.pt) loads every checkpoint
   tensor; the only uncovered model params are pool.*
"""
from collections import deque

import torch

from student_vision import (AttentionPool, RoPEFrameEncoder,
                            build_two_scale_window, HIST_LEN, N_FRAMES,
                            N_COARSE, COARSE_STEPS, STRIDE, SPAN, TOK_TOTAL)

torch.manual_seed(0)
pool = AttentionPool()

# --- 1. window construction ------------------------------------------------
feats = [torch.randn(384) for _ in range(4200)]
tok, pos = build_two_scale_window(feats, pool)
assert tok.shape == (TOK_TOTAL, 384), tok.shape
assert pos.shape == (TOK_TOTAL,)
assert torch.all(pos[1:] > pos[:-1]), "positions must be strictly increasing"
assert not torch.isnan(tok).any()
# fine rows equal the old construction exactly (full history)
F = torch.stack(feats[-HIST_LEN:])
assert torch.equal(tok[N_COARSE:], F[COARSE_STEPS::STRIDE])
old = torch.stack(feats)[-SPAN - 1:][::STRIDE][-N_FRAMES:]
assert torch.equal(tok[N_COARSE:], old), "fine tokens differ from old layout"
# short history (10 steps): fine rows must match the old appendleft padding
buf = deque(feats[:10], maxlen=HIST_LEN)
tok10, _ = build_two_scale_window(buf, pool)
buf_old = deque(feats[:10], maxlen=SPAN + 1)
while len(buf_old) < SPAN + 1:
    buf_old.appendleft(buf_old[0])
old10 = torch.stack(list(buf_old)[::STRIDE])[-N_FRAMES:]
assert torch.equal(tok10[N_COARSE:], old10), "padding differs from old conv"
assert not torch.isnan(tok10).any()
print("1. window construction OK "
      f"(tokens {tuple(tok.shape)}, pos {pos[0]:.0f}..{pos[-1]:.0f} monotone, "
      "padding == old convention)")

# --- 2. encoder: shapes, causality, explicit RoPE positions -----------------
enc = RoPEFrameEncoder()
x = torch.randn(2, TOK_TOTAL, 384)
h = enc.forward_full(x)
assert h.shape == (2, TOK_TOTAL, 256), h.shape
assert enc(x).shape == (2, 256)
# causality: perturb the last token -> hidden at position 0 unchanged
x2 = x.clone()
x2[:, -1] += 1.0
h2 = enc.forward_full(x2)
assert torch.allclose(h2[:, 0], h[:, 0], atol=1e-5), "causality broken"
# explicit shifted RoPE positions must change the output
h3 = enc.forward_full(x, pos=pos + 100.0)
assert not torch.allclose(h3, h), "RoPE position parameter has no effect"
# legacy fine-only path (T=30, old checkpoints' diagnostic scripts) works
assert enc(torch.randn(1, N_FRAMES, 384)).shape == (1, 256)
print("2. encoder OK (shapes, causal, explicit positions take effect, "
      "legacy T=30 path intact)")

# --- 3. old checkpoint compatibility ----------------------------------------
from student_rl import RLStudent

CKPT = "runs/student_rl/ckpt_v32_it45.pt"
net = RLStudent(encoder="tx")
sd = torch.load(CKPT, map_location="cpu")
uncovered = [k for k in net.state_dict() if k not in sd]
net.load_bc(CKPT)   # prints "warm start: N/M tensors"
assert uncovered and all(k.startswith("pool.") for k in uncovered), uncovered
print(f"3. old ckpt OK (only pool.* is new: {len(uncovered)} tensors, "
      "random-init as expected)")
print("ALL SMOKE OK")
