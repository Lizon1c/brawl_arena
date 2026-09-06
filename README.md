# brawl-arena

A small, fully offline, original top-down arena battler environment +
training pipeline for vision-based game agents. Inspired by Brawl-Stars-style
gameplay (gem grab / brawl ball / knockout / showdown) but written from
scratch — no assets, code, or data from any commercial game.

Pipeline:

```
scripted bot (oracle)  --BC-->  vision student  --(planned) RL-->  stronger agent
        ^                        DINOv2 + CPG oscillator bank,
        |                        continuous joysticks, frames only
        +-- PPO teacher (privileged state vector, self-play)
```

## Layout

- `brawl_arena/` — the game package
  - `core.py` — physics, combat, modes, reward events
  - `maps.py` — map definitions + validation (loads user maps from
    `maps_custom/` at import)
  - `bots.py` — scripted baseline bot (vision-fair: respects bush stealth)
  - `gym_env.py` — Gymnasium wrapper
  - `render.py` — rasterizer (game frames + map previews)
- `editor.py` — local web map editor (`python editor.py` ->
  http://127.0.0.1:8787), symmetry-assisted, saves to `maps_custom/`
- `maps_custom/` — user-designed maps (ASCII + preview PNG)

Teacher track (privileged state, PPO):
- `train_teacher.py` — PPO self-play / vs-scripted, reward shaping, KL anchor
- `pretrain_teacher.py` — behaviour-clone the scripted bot as warm start
- `dagger_teacher.py` — DAgger rounds
- `eval_teacher.py`, `record_teacher.py`

Student track (raw pixels, 90x126):
- `student_vision.py` — DINOv2-S/14 + CPG oscillator bank, discrete heads (legacy)
- `student_omni.py` — same trunk, continuous move/aim joysticks (legacy BC track)
- `student_rl.py` — **current (v5.3)**: trainable CNN fine stream -> 384-d feat ->
  FeatRing(4155) -> dual-scale windows (fine 30 + coarse 128 attention-pool) ->
  4-layer causal RoPE transformer -> readout + 4 action heads + value head +
  next-frame pred head. Online PPO + DAgger + spectate BC over a 9-env mix
  (gem/duo vs bots, gem/ball/ko/duel self-play, spectate), 3.92M params.
- `student_cnn_pretrain.py`, `student_tx_pretrain.py` — pretraining stages
- `scripts/eval_duel.py` — duel combat-metrics eval (argmax + sampled)
- `run_chunked.sh` — continue training in 10-epoch chunks with periodic
  in-game evaluation until a target winrate

Misc: `record_match.py` (bot-vs-bot videos), `balance_check.py`,
`balance_builtin.py`, `diag_teacher.py`, `diagnose_idle.py`, `smoke_test.py`

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
# for CUDA torch, install from the cu126 index instead (see requirements.txt)
```

The DINOv2 backbone is downloaded via `torch.hub` on first use (internet
required once; cached afterwards).

## Quick start

```
.venv/Scripts/python smoke_test.py                          # sanity check
.venv/Scripts/python editor.py                              # map editor
.venv/Scripts/python record_match.py gem_grab out/demo.mp4 60
.venv/Scripts/python student_omni.py --train 10             # student BC
.venv/Scripts/python train_teacher.py --steps 1000000 --mode all
```

Datasets and checkpoints live under `runs/` (git-ignored).
