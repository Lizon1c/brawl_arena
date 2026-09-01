"""Stage 1: train the privileged teacher with PPO + self-play.

The teacher sees only the privileged state vector (full game state,
including stealthed units) -- the vision-only student comes later and
will distill from this teacher.

Opponents (and teammates) are frozen snapshots of the learner itself:
every --snapshot-interval env steps the current policy is saved to a file
(atomic replace); each subprocess env holds a SelfPlayPolicy that reloads
the file when it changes. Until the first snapshot exists the scripted
bot stands in.

SB3's PPO does not support Dict action spaces, so DiscreteActionWrapper
flattens the action to MultiDiscrete([9, 9, 2, 2]):
    move direction (8-way + stay), aim direction (8-way), shoot, super.

Usage:
    .venv/Scripts/python train_teacher.py --mode gem_grab --steps 2000000
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from brawl_arena import Action, BrawlArenaEnv, Config, ScriptedBot
from brawl_arena.core import MODES

_DIRS8 = np.array([[math.cos(a), math.sin(a)] for a in
                   np.linspace(0, 2 * math.pi, 8, endpoint=False)], dtype=np.float32)


class DiscreteActionWrapper(gym.Wrapper):
    """MultiDiscrete([9, 9, 2, 2]) -> the env's Dict action."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.MultiDiscrete([9, 9, 2, 2])

    def step(self, action):
        mv = np.zeros(2, dtype=np.float32) if action[0] == 8 else _DIRS8[action[0]]
        aim = np.array([1.0, 0.0], dtype=np.float32) if action[1] == 8 else _DIRS8[action[1]]
        return self.env.step({"move": mv, "aim": aim,
                              "shoot": int(action[2]), "super": int(action[3])})

    def predict_forward(self, model, obs_dict):
        """Deterministic teacher action for later distillation."""
        a, _ = model.predict(obs_dict, deterministic=True)
        return multi_to_action(a)


def multi_to_action(a) -> Action:
    """MultiDiscrete([9, 9, 2, 2]) -> core Action."""
    mv = np.zeros(2, dtype=np.float32) if a[0] == 8 else _DIRS8[a[0]]
    aim = np.array([1.0, 0.0], dtype=np.float32) if a[1] == 8 else _DIRS8[a[1]]
    return Action(move=mv, aim=aim, shoot=bool(a[2]), use_super=bool(a[3]))


class StateOnlyExtractor(BaseFeaturesExtractor):
    """Ignore the rendered frame; teacher learns from privileged state only."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_in = observation_space["state"].shape[0]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, features_dim), torch.nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs["state"])


class AnchoredPPO(PPO):
    """PPO with an RLHF-style KL anchor to a frozen reference policy.

    Adds `kl_coef * mean KL(pi_ref || pi)` to the loss so the learner can
    never drift far below the BC/scripted baseline: any state where the
    reference puts probability mass must keep mass under the learner too.
    The coefficient decays linearly to 0 over the first `kl_anneal`
    fraction of training, so late self-play is unconstrained and can
    surpass the reference."""

    def __init__(self, *args, anchor_path: str | None = None,
                 kl_coef: float = 0.0, kl_anneal: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor = None
        self.kl_coef = kl_coef
        self.kl_anneal = kl_anneal
        if anchor_path and kl_coef > 0:
            ref = PPO.load(anchor_path, device=self.device)
            self.anchor = ref.policy
            self.anchor.set_training_mode(False)
            for p in self.anchor.parameters():
                p.requires_grad_(False)

    def _kl_coef_now(self) -> float:
        if self.anchor is None or self.kl_coef <= 0:
            return 0.0
        a = self.kl_anneal
        # progress_remaining goes 1 -> 0 over training; full coef until
        # (1 - a), then linear decay to 0
        pr = float(np.clip((self._current_progress_remaining - (1 - a)) / a,
                           0.0, 1.0))
        return self.kl_coef * pr

    def train(self) -> None:
        """SB3 PPO.train plus the anchor KL term (kept in sync with
        stable_baselines3 2.9.0 ppo.py)."""
        import torch.nn.functional as F
        from stable_baselines3.common.utils import explained_variance

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        kl_coef = self._kl_coef_now()

        entropy_losses = []
        pg_losses, value_losses, anchor_losses = [], [], []
        clip_fractions = []
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) \
                        / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss \
                    + self.vf_coef * value_loss

                if kl_coef > 0:
                    dist = self.policy.get_distribution(
                        rollout_data.observations)
                    with torch.no_grad():
                        ref_dist = self.anchor.get_distribution(
                            rollout_data.observations)
                    # MultiDiscrete -> per-head Categoricals; KL(ref||cur)
                    # summed over heads, averaged over the batch
                    kl = sum(
                        torch.distributions.kl_divergence(d_ref, d_cur)
                        for d_ref, d_cur in zip(ref_dist.distribution,
                                                dist.distribution))
                    anchor_loss = kl_coef * kl.mean()
                    anchor_losses.append(anchor_loss.item())
                    loss = loss + anchor_loss

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None \
                        and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to "
                              f"reaching max kl: {approx_kl_div:.2f}")
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                               self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if anchor_losses:
            self.logger.record("train/anchor_loss", np.mean(anchor_losses))
            self.logger.record("train/kl_coef", kl_coef)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std",
                               torch.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates,
                           exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class SelfPlayPolicy:
    """`.act(game, idx)` / `.act_batch(game, indices)` opponent backed by a
    frozen snapshot of the learner. Reloads the snapshot file only when it
    changes on disk, debounced so a flapping mtime (AV/sync tools touching
    the file) can never cause a reload hot-loop. Falls back to the scripted
    bot until the first snapshot exists."""

    def __init__(self, model_path: str, seed: int = 0,
                 check_interval: float = 5.0):
        torch.set_num_threads(1)   # tiny MLP; avoid OMP thrash across workers
        self.model_path = model_path
        self.model = None
        self.mtime = -1.0
        self.check_interval = check_interval
        self._next_check = 0.0
        self.fallback = ScriptedBot(seed=seed)

    def _maybe_reload(self):
        now = time.monotonic()
        if now < self._next_check:
            return
        self._next_check = now + self.check_interval
        try:
            mt = os.path.getmtime(self.model_path)
        except OSError:
            return
        if mt <= self.mtime:
            return
        try:
            self.model = PPO.load(self.model_path, device="cpu")
            self.mtime = mt
        except Exception:
            pass   # snapshot mid-replace or unreadable: keep the old model

    def act(self, game, idx) -> Action:
        return self.act_batch(game, [idx])[0]

    def act_batch(self, game, indices) -> list[Action]:
        """One batched predict for all controlled units of this env step."""
        self._maybe_reload()
        if self.model is None:
            return [self.fallback.act(game, i) for i in indices]
        obs = {"frame": np.zeros((len(indices), 1, 1, 3), dtype=np.uint8),
               "state": np.stack([game.state_vector(i) for i in indices])}
        a, _ = self.model.predict(obs, deterministic=False)
        return [multi_to_action(row) for row in np.asarray(a)]


def save_snapshot(model, path: str):
    """Atomic model save: write to a temp file, then replace."""
    tmp = path + ".tmp.zip"
    model.save(tmp)
    os.replace(tmp, path)


class SnapshotCallback(BaseCallback):
    """Publish a frozen self-play snapshot every `interval` env steps."""

    def __init__(self, path: str, interval: int):
        super().__init__()
        self.path = path
        self.interval = interval
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.interval:
            self._last = self.num_timesteps
            try:
                save_snapshot(self.model, self.path)
            except OSError:
                pass   # a subprocess may be reading the file; retry next time
        return True


def make_env(rank: int, mode: str, policy_path: str, shaped: bool = False,
             scripted_opp: bool = False):
    def _init():
        # include_frame=False: skip rendering, the teacher does not need it
        # one shared snapshot policy per subprocess: halves model copies
        # and lets the env batch all non-learner units into one predict
        sp = SelfPlayPolicy(policy_path, seed=rank)
        # scripted_opp: play against the scripted bot directly (league-style
        # opponent mixing) so the learner gets gradient against the exact
        # eval opponent, not just its own snapshots
        opp = ScriptedBot(seed=3000 + rank) if scripted_opp else sp
        env = BrawlArenaEnv(config=Config(mode=mode, reward_shaping=shaped),
                            seed=1000 + rank,
                            opponent_policy=opp, teammate_policy=sp,
                            include_frame=False)
        return DiscreteActionWrapper(env)
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="gem_grab",
                    choices=list(MODES) + ["all"],
                    help="'all' spreads envs across the four modes")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", type=str, default="runs/teacher")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--snapshot-interval", type=int, default=100_000,
                    help="env steps between self-play snapshot updates")
    ap.add_argument("--init", type=str, default=None,
                    help="warm-start weights, e.g. from pretrain_teacher.py")
    ap.add_argument("--ent-coef", type=float, default=0.0,
                    help="entropy bonus; use ~0.01 after BC warm-start to "
                         "keep the near-deterministic imitator exploring")
    ap.add_argument("--anchor", type=str, default=None,
                    help="frozen reference policy for the KL anchor, "
                         "e.g. runs/teacher_bc/bc_init.zip")
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="peak weight of the anchor KL term (0 disables it)")
    ap.add_argument("--kl-anneal", type=float, default=0.3,
                    help="fraction of training over which the anchor weight "
                         "decays to 0")
    ap.add_argument("--shaped", action="store_true",
                    help="enable behaviour-prior reward shaping "
                         "(hits/kills/survival/gem-holding/range-keeping)")
    ap.add_argument("--scripted-frac", type=float, default=0.0,
                    help="fraction of envs facing scripted bots instead of "
                         "self-play snapshots (league-style mixing)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    policy_path = os.path.join(args.out, "selfplay_latest.zip")
    n_scripted = int(round(args.scripted_frac * args.n_envs))
    modes = MODES if args.mode == "all" else [args.mode]
    env = VecMonitor(SubprocVecEnv(
        [make_env(i, modes[i % len(modes)], policy_path, args.shaped,
                  scripted_opp=(i < n_scripted))
         for i in range(args.n_envs)]))
    if n_scripted:
        print(f"{n_scripted}/{args.n_envs} envs face scripted bots")
    if args.mode == "all":
        print("env modes:", [modes[i % len(modes)] for i in range(args.n_envs)])

    model = AnchoredPPO(
        "MultiInputPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=StateOnlyExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        ),
        learning_rate=3e-4, n_steps=1024, batch_size=512,
        gamma=0.995, ent_coef=args.ent_coef, verbose=1, device=args.device,
        tensorboard_log=args.out,
        anchor_path=args.anchor, kl_coef=args.kl_coef,
        kl_anneal=args.kl_anneal,
    )
    if args.init:
        model.set_parameters(args.init)
        print(f"warm-started from {args.init}")
    # publish the initial snapshot so self-play starts immediately
    save_snapshot(model, policy_path)
    model.learn(total_timesteps=args.steps,
                callback=SnapshotCallback(policy_path, args.snapshot_interval))
    model.save(os.path.join(args.out, "teacher_final"))
    print("saved to", os.path.join(args.out, "teacher_final.zip"))


if __name__ == "__main__":
    main()
