"""Perf microbenchmark for the two-scale change (run on the training GPU).

A. per-step rollout cost: build_two_scale_window (stack 4155 + attention
   pool) vs the old fine-only construction (stack 30) -- this is the added
   per-env-per-step cost in train()'s rollout loop.
B. per-update cost: forward_full fwd+bwd at batch 512, T=158 vs the old
   fine-only T=30 window.
"""
import time
from collections import deque

import torch

from student_vision import (AttentionPool, RoPEFrameEncoder,
                            build_two_scale_window, HIST_LEN, N_FRAMES,
                            STRIDE, SPAN)

device = "cuda"
pool = AttentionPool().to(device)
enc = RoPEFrameEncoder().to(device)


def bench(fn, n=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000  # ms


buf = deque(maxlen=HIST_LEN)
f = torch.randn(HIST_LEN, 384, device=device)
buf.extend(f)


def old_window():
    return torch.stack(list(buf)[-SPAN - 1:][::STRIDE])[-N_FRAMES:]


def new_window():
    return build_two_scale_window(buf, pool)[0]


print(f"A. window build/env-step: old {bench(old_window):.3f} ms  "
      f"new {bench(new_window):.3f} ms")

x158 = torch.randn(512, 158, 384, device=device)
x30 = torch.randn(512, 30, 384, device=device)


def fwd_bwd(x):
    enc.zero_grad()
    h = enc.forward_full(x)
    h.square().mean().backward()


print(f"B. forward_full fwd+bwd (512,T,384): T=30 {bench(lambda: fwd_bwd(x30), 20, 3):.1f} ms  "
      f"T=158 {bench(lambda: fwd_bwd(x158), 20, 3):.1f} ms")
print(f"GPU mem allocated: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
