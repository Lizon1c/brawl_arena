"""Stage 2: distill the privileged teacher into a vision-only student.

Pipeline (each stage is a flag; with no stage flag the full pipeline runs):

  --bc       behaviour cloning: collect (frame, teacher action) pairs with
             the frozen teacher driving unit 0 and frozen teacher snapshots
             driving everyone else; train a small CNN by cross-entropy
             (with horizontal-flip augmentation) -> runs/student/bc.pt
  --rl       PPO fine-tune of the BC network against the frozen teacher
             snapshot (frames only, state ignored) -> runs/student/rl_final.zip
  --export   export the student CNN to ONNX (opset 17)
             -> runs/student/student.onnx
  --eval     play N games vs scripted bots with the BC/RL student

The student net is a ~0.4M-param CNN (runs on-device). Its forward takes a
(B, 3, 90, 126) float32 frame in [0, 255] (normalised inside) and returns
(B, 22) logits = [move(9) | aim(9) | shoot(2) | super(2)].

Examples:
    .venv/Scripts/python distill_student.py --bc --bc-samples 50000 --bc-epochs 5
    .venv/Scripts/python distill_student.py --rl --steps 500000
    .venv/Scripts/python distill_student.py --export --eval

Note: --device defaults to cpu so this can run alongside GPU training.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from brawl_arena import BrawlArenaEnv, Config
from brawl_arena.core import MODES
from brawl_arena.maps import _MODE_SIZE
from train_teacher import DiscreteActionWrapper, SelfPlayPolicy, multi_to_action

ACTION_DIMS = (9, 9, 2, 2)
# x-mirror remap for the 8-way direction indices: angle k*45deg maps to
# 180deg - k*45deg, i.e. index (4 - k) % 8; 8 (stay) is unchanged
FLIP_LUT = torch.tensor([4, 3, 2, 1, 0, 7, 6, 5, 8])


class StudentNet(nn.Module):
    """Small CNN policy: frame -> 4 categorical action heads."""

    def __init__(self, features_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 48, 5, 2, 1), nn.ReLU(),
            nn.Conv2d(48, 96, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(96, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(128, features_dim), nn.ReLU())
        self.heads = nn.ModuleList(
            [nn.Linear(features_dim, d) for d in ACTION_DIMS])

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float32 in [0, 255]; returns (B, 22) logits."""
        z = self.features(x / 255.0)
        return torch.cat([h(z) for h in self.heads], dim=1)


def split_logits(logits: torch.Tensor) -> list[torch.Tensor]:
    return list(torch.split(logits, list(ACTION_DIMS), dim=1))


class CNNExtractor(BaseFeaturesExtractor):
    """SB3 features extractor: CNN over the frame, ignores the state vector."""

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.student = StudentNet(features_dim)

    def forward(self, obs):
        x = obs["frame"].float()
        # SB3's VecTransposeImage usually delivers (B, C, H, W) already;
        # fall back to permuting if the frame still has channels-last
        if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        return self.student.features(x / 255.0)


def load_bc_into_ppo(policy, bc_path: str):
    """Copy BC weights into a PPO policy (extractor cnn+fc, action heads
    concatenated into the 22-wide action_net)."""
    student = StudentNet()
    student.load_state_dict(torch.load(bc_path, map_location="cpu",
                                       weights_only=True))
    ext = policy.features_extractor.student
    ext.cnn.load_state_dict(student.cnn.state_dict())
    ext.fc.load_state_dict(student.fc.state_dict())
    with torch.no_grad():
        policy.action_net.weight.copy_(
            torch.cat([h.weight for h in student.heads], dim=0))
        policy.action_net.bias.copy_(
            torch.cat([h.bias for h in student.heads], dim=0))


def load_student_from_ppo(policy) -> StudentNet:
    """Inverse of load_bc_into_ppo: rebuild a standalone StudentNet."""
    student = StudentNet()
    ext = policy.features_extractor.student
    student.cnn.load_state_dict(ext.cnn.state_dict())
    student.fc.load_state_dict(ext.fc.state_dict())
    w = policy.action_net.weight.detach().cpu()
    b = policy.action_net.bias.detach().cpu()
    for h, ws, bs in zip(student.heads,
                         torch.split(w, list(ACTION_DIMS), dim=0),
                         torch.split(b, list(ACTION_DIMS), dim=0)):
        with torch.no_grad():
            h.weight.copy_(ws)
            h.bias.copy_(bs)
    return student


# ------------------------------------------------------------- BC stage
def collect_samples(teacher_path: str, mode: str, n: int, seed: int):
    """Teacher drives unit 0 (deterministic); everyone else runs the frozen
    teacher snapshot. Records (frame, MultiDiscrete action) pairs."""
    teacher = PPO.load(teacher_path, device="cpu")
    sp_opp = SelfPlayPolicy(teacher_path, seed=0)
    sp_tm = SelfPlayPolicy(teacher_path, seed=1)
    env = BrawlArenaEnv(config=Config(mode=mode), seed=seed,
                        opponent_policy=sp_opp, teammate_policy=sp_tm,
                        include_frame=True)
    fshape = env.observation_space["frame"].shape
    frames = np.zeros((n, *fshape), dtype=np.uint8)
    acts = np.zeros((n, 4), dtype=np.int64)
    obs, _ = env.reset(seed=seed)
    t0 = time.perf_counter()
    for i in range(n):
        t_obs = {"frame": np.zeros((1, 1, 3), dtype=np.uint8),
                 "state": env.game.state_vector(0)}
        a, _ = teacher.predict(t_obs, deterministic=True)
        a = np.asarray(a).ravel()
        frames[i] = obs["frame"]
        acts[i] = a
        act = multi_to_action(a)
        obs, _, done, _, _ = env.step({"move": act.move, "aim": act.aim,
                                       "shoot": int(act.shoot),
                                       "super": int(act.use_super)})
        if done:
            obs, _ = env.reset()
        if (i + 1) % 10000 == 0:
            fps = (i + 1) / (time.perf_counter() - t0)
            print(f"  collected {i + 1}/{n} ({fps:.0f} samples/s)")
    return frames, acts


def train_bc(frames: np.ndarray, acts: np.ndarray, epochs: int, device: str,
             batch_size: int = 64, lr: float = 1e-3, augment: bool = True):
    net = StudentNet().to(device)
    n = len(frames)
    n_val = max(1, n // 10)
    perm = np.random.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    x_all = torch.from_numpy(frames).permute(0, 3, 1, 2)   # uint8, cpu
    y_all = torch.from_numpy(acts)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lut = FLIP_LUT
    for ep in range(epochs):
        net.train()
        order = np.random.permutation(len(tr_idx))
        tot_loss = nb = 0
        for b in range(0, len(order), batch_size):
            idx = tr_idx[order[b:b + batch_size]]
            x = x_all[idx].float().to(device)
            y = y_all[idx].clone()
            if augment:
                m = torch.rand(len(idx)) < 0.5
                x[m] = torch.flip(x[m], dims=[3])
                y[m, 0] = lut[y[m, 0]]
                y[m, 1] = lut[y[m, 1]]
            y = y.to(device)
            logits = net(x)
            loss = sum(nn.functional.cross_entropy(lg, y[:, k])
                       for k, lg in enumerate(split_logits(logits)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            nb += 1
        # validation accuracy per head
        net.eval()
        with torch.no_grad():
            correct = np.zeros(4)
            for b in range(0, len(val_idx), 512):
                idx = val_idx[b:b + 512]
                x = x_all[idx].float().to(device)
                pred = net(x)
                for k, lg in enumerate(split_logits(pred)):
                    correct[k] += (lg.argmax(1).cpu() == y_all[idx, k]).sum()
            acc = correct / len(val_idx)
        print(f"  epoch {ep + 1}/{epochs} loss {tot_loss / nb:.4f} "
              f"val acc move {acc[0]:.3f} aim {acc[1]:.3f} "
              f"shoot {acc[2]:.3f} super {acc[3]:.3f} mean {acc.mean():.3f}")
    return net


# ------------------------------------------------------------- RL stage
def run_rl(args):
    bc_path = os.path.join(args.out, "bc.pt")

    def make_env(rank: int):
        def _init():
            sp = SelfPlayPolicy(args.teacher, seed=rank)   # frozen teacher
            env = BrawlArenaEnv(config=Config(mode=args.mode), seed=2000 + rank,
                                opponent_policy=sp, teammate_policy=sp,
                                include_frame=True)
            return DiscreteActionWrapper(env)
        return _init

    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(args.n_envs)]))
    model = PPO(
        "MultiInputPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=CNNExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[], vf=[]),   # action_net == BC heads (22 logits)
        ),
        learning_rate=1e-4, n_steps=1024, batch_size=512,
        gamma=0.995, verbose=1, device=args.device,
        tensorboard_log=args.out,
    )
    if os.path.exists(bc_path):
        load_bc_into_ppo(model.policy, bc_path)
        print(f"loaded BC weights from {bc_path}")
    else:
        print("warning: no bc.pt found, RL starts from scratch")
    model.learn(total_timesteps=args.steps)
    out = os.path.join(args.out, "rl_final")
    model.save(out)
    print("saved to", out + ".zip")


# --------------------------------------------------------- export stage
def export_onnx(args):
    rl_path = os.path.join(args.out, "rl_final.zip")
    bc_path = os.path.join(args.out, "bc.pt")
    src = args.export_from
    if src == "auto":
        src = "rl" if os.path.exists(rl_path) else "bc"
    if src == "rl":
        model = PPO.load(rl_path, device="cpu")
        net = load_student_from_ppo(model.policy)
        print(f"exporting from {rl_path}")
    else:
        net = StudentNet()
        net.load_state_dict(torch.load(bc_path, map_location="cpu",
                                       weights_only=True))
        print(f"exporting from {bc_path}")
    net.eval()

    w, h = _MODE_SIZE[args.mode]
    cfg = Config(mode=args.mode)
    hh, ww = h * cfg.render_scale, w * cfg.render_scale
    dummy = torch.zeros(1, 3, hh, ww)
    out = os.path.join(args.out, "student.onnx")
    try:
        # legacy TorchScript exporter: no extra deps (dynamo path needs
        # the onnxscript package)
        torch.onnx.export(net, dummy, out, opset_version=17, dynamo=False,
                          input_names=["frame"], output_names=["logits"])
    except (TypeError, ModuleNotFoundError):
        torch.onnx.export(net, dummy, out, opset_version=17,
                          input_names=["frame"], output_names=["logits"])
    print(f"saved {out}  input (1,3,{hh},{ww}) float32 [0,255] -> "
          f"(1,22) logits [move(9)|aim(9)|shoot(2)|super(2)]")

    try:
        import onnxruntime as ort
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "onnxruntime"])
        import onnxruntime as ort
    sess = ort.InferenceSession(out)
    x = np.random.randint(0, 256, (1, 3, hh, ww)).astype(np.float32)
    y = sess.run(["logits"], {"frame": x})[0]
    print("onnxruntime output shape:", y.shape)
    t0 = time.perf_counter()
    for _ in range(50):
        sess.run(["logits"], {"frame": x})
    print(f"onnx inference: {(time.perf_counter() - t0) / 50 * 1000:.2f} ms/frame")


# ------------------------------------------------------------ eval stage
def evaluate(args):
    bc_path = os.path.join(args.out, "bc.pt")
    rl_path = os.path.join(args.out, "rl_final.zip")
    if os.path.exists(bc_path):
        net = StudentNet()
        net.load_state_dict(torch.load(bc_path, map_location="cpu",
                                       weights_only=True))
        src = bc_path
    else:
        net = load_student_from_ppo(PPO.load(rl_path, device="cpu").policy)
        src = rl_path
    net.eval()
    print(f"evaluating {src} vs scripted bots over {args.eval_episodes} games")
    wins = draws = losses = 0
    gem_diffs = []
    for seed in range(args.eval_episodes):
        env = BrawlArenaEnv(config=Config(mode=args.mode), seed=5000 + seed,
                            include_frame=True)
        obs, _ = env.reset()
        done = False
        while not done:
            x = torch.from_numpy(obs["frame"]).permute(2, 0, 1)[None].float()
            with torch.no_grad():
                logits = net(x)
            a = [lg.argmax(1).item() for lg in split_logits(logits)]
            act = multi_to_action(a)
            obs, _, terminated, _, info = env.step(
                {"move": act.move, "aim": act.aim,
                 "shoot": int(act.shoot), "super": int(act.use_super)})
            done = terminated
        w = info["winner"]
        if w == 0:
            wins += 1
        elif w is None:
            draws += 1
        else:
            losses += 1
        if "gems" in info:
            gem_diffs.append(info["gems"][0] - info["gems"][1])
    extra = f", avg gem diff {np.mean(gem_diffs):.1f}" if gem_diffs else ""
    print(f"W{wins} D{draws} L{losses}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="gem_grab", choices=MODES)
    ap.add_argument("--teacher", type=str,
                    default="runs/teacher_gem_grab/teacher_final.zip")
    ap.add_argument("--out", type=str, default="runs/student")
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu keeps the GPU free for other training runs")
    ap.add_argument("--bc", action="store_true")
    ap.add_argument("--bc-samples", type=int, default=50_000,
                    help="~34 MB RAM per 1000 samples (uint8 frames)")
    ap.add_argument("--bc-epochs", type=int, default=5)
    ap.add_argument("--rl", action="store_true")
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--export-from", choices=["auto", "bc", "rl"],
                    default="auto")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-episodes", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_all = not (args.bc or args.rl or args.export or args.eval)

    if args.bc or run_all:
        print(f"[bc] collecting {args.bc_samples} samples...")
        frames, acts = collect_samples(args.teacher, args.mode,
                                       args.bc_samples, seed=42)
        torch.set_num_threads(os.cpu_count() or 4)   # undo SelfPlayPolicy's =1
        print(f"[bc] training {args.bc_epochs} epochs on {len(frames)} samples...")
        net = train_bc(frames, acts, args.bc_epochs, args.device)
        torch.save(net.state_dict(), os.path.join(args.out, "bc.pt"))
        print("saved to", os.path.join(args.out, "bc.pt"))
    if args.rl or run_all:
        run_rl(args)
    if args.export or run_all:
        export_onnx(args)
    if args.eval or run_all:
        evaluate(args)


if __name__ == "__main__":
    main()
