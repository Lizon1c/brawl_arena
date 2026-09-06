"""Smoke test for student_rl.train() with the two-scale window.

Redirects OUT to temp dirs so runs/student_rl/ (live training) is never
touched. Runs 1 iter then 3 iters; the difference gives a steady-state
steps/s uncontaminated by env/DINO setup.
"""
import tempfile
import time

import student_rl

H, ENVS = 32, 8

student_rl.OUT = tempfile.mkdtemp(prefix="rl_smoke_2s_a_")
print("OUT ->", student_rl.OUT)
t1 = time.perf_counter()
student_rl.train(n_iters=1, device="cuda", horizon=H, warmup_iters=0,
                 ppo_epochs=1)
t1 = time.perf_counter() - t1

student_rl.OUT = tempfile.mkdtemp(prefix="rl_smoke_2s_b_")
print("OUT ->", student_rl.OUT)
t3 = time.perf_counter()
student_rl.train(n_iters=3, device="cuda", horizon=H, warmup_iters=0,
                 ppo_epochs=1)
t3 = time.perf_counter() - t3

steady = (2 * H * ENVS) / (t3 - t1)
print(f"1 iter: {t1:.1f}s (incl. setup), 3 iters: {t3:.1f}s, "
      f"steady-state {steady:.1f} env-steps/s")
print("SMOKE OK")
