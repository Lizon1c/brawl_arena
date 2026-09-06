"""CPU smoke test for student_rl.train() with snapshots enabled.

Redirects OUT to a temp dir so runs/student_rl/rl_latest.pt (being written
by the live training run) is never touched. Bucket starts empty so
snapshot_prob=1.0 falls back to normal resets -- this only proves the new
code path doesn't crash train(); restore correctness is in test_snapshot.py.
"""
import tempfile

import student_rl

student_rl.OUT = tempfile.mkdtemp(prefix="rl_smoke_")
print("OUT ->", student_rl.OUT)
student_rl.train(n_iters=1, device="cpu", horizon=8, warmup_iters=0,
                 ppo_epochs=1, snapshot_prob=1.0)
print("SMOKE OK")
