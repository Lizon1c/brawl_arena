"""Gymnasium wrapper around brawl-arena.

The controlled agent is unit 0. All other units are driven by pluggable
policies (scripted bot by default, or a self-play snapshot in training).
In team modes units 1..N use teammate_policy / opponent_policy by side;
in showdown every other unit uses the opponent policy.

Observation: Dict(
    frame: (H, W, 3) uint8  -- full-arena top-down render from team 0's
                               perspective (bush stealth applied). The
                               vision-only student sees only this.
    state: (state_dim,) float32 -- privileged vector, sees everything
                               including stealthed units (teacher input).
)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import ARCHETYPES, Action, Config, Game
from .render import render
from .bots import ScriptedBot


class BrawlArenaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Config | None = None, seed: int | None = None,
                 opponent_policy=None, teammate_policy=None,
                 include_frame: bool = True, n_students: int = 1):
        super().__init__()
        self.cfg = config or Config()
        self.game = Game(self.cfg, seed=seed)
        # seed the default scripted bots so same-seed episodes are
        # reproducible (explicit policies keep their own rng)
        self.opponent_policy = opponent_policy or ScriptedBot(seed=seed)
        self.teammate_policy = teammate_policy or ScriptedBot(
            seed=None if seed is None else seed + 1)
        # units 0..n_students-1 are driven externally (learning students);
        # units from n_students on are driven by the policy hooks
        self.n_students = n_students
        self.include_frame = include_frame
        self._last_mine = 0.0
        self._last_enemy = 0.0
        self._last_carry_ours = False

        h = self.cfg.map_h * self.cfg.render_scale
        w = self.cfg.map_w * self.cfg.render_scale
        frame_shape = (h, w, 3) if include_frame else (1, 1, 3)
        self.observation_space = spaces.Dict({
            "frame": spaces.Box(0, 255, frame_shape, dtype=np.uint8),
            "state": spaces.Box(-10.0, 10.0, (self.game.state_dim,), dtype=np.float32),
        })
        self.action_space = spaces.Dict({
            "move": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
            "aim": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
            "shoot": spaces.Discrete(2),
            "super": spaces.Discrete(2),
        })

    def _obs(self):
        frame = render(self.game, viewer_team=0) if self.include_frame \
            else np.zeros((1, 1, 3), dtype=np.uint8)
        return {"frame": frame, "state": self.game.state_vector(0)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game = Game(self.cfg, seed=seed)
        else:
            self.game.reset()
        self._last_mine = 0.0
        self._last_enemy = 0.0
        self._last_carry_ours = False
        return self._obs(), {}

    def set_game(self, game: Game):
        """Swap in a restored mid-game state (snapshot resume).

        The differential-reward baselines are re-seeded from the restored
        state so the first step after the swap does not see a spurious
        delta, and the observation is regenerated from the new game."""
        self.game = game
        self.cfg = game.cfg   # deepcopied snapshots carry their own cfg copy
        if game.mode == "gem_grab":
            self._last_mine, self._last_enemy = game._relative_score(0)
        else:
            self._last_mine = 0.0
            self._last_enemy = 0.0
        self._last_carry_ours = game.mode == "brawl_ball" \
            and game.ball_carrier is not None \
            and game.units[game.ball_carrier].team == 0
        return self._obs(), {}

    def _reward(self):
        """Scalar reward for n_students == 1 (legacy behaviour, unchanged);
        a list of per-student rewards for multi-student envs (duo): own
        hits/kills/deaths are per-unit, round/game outcomes are shared."""
        g = self.game
        ns = self.n_students
        rs = [-0.001] * ns
        if g.mode == "gem_grab":
            mine, enemy = g._relative_score(0)
            rs[0] += 2.0 * (mine - self._last_mine) - 2.0 * (enemy - self._last_enemy)
            self._last_mine, self._last_enemy = mine, enemy

        for ev in g.events:
            if g.mode in ("knockout", "duel"):
                if ev[0] == "hit":
                    _, atk, victim = ev
                    for i in range(ns):
                        if atk == i:
                            rs[i] += 0.05      # landed a shot
                        if victim == i:
                            rs[i] -= 0.05      # got hit
                elif ev[0] == "kill":
                    _, atk, victim = ev
                    if ns == 1:
                        if g.units[victim].team == 0:
                            rs[0] -= 1.0
                        elif atk is not None and g.units[atk].team == 0:
                            rs[0] += 1.0
                    else:
                        for i in range(ns):
                            if victim == i:
                                rs[i] -= 1.0
                            elif g.units[victim].team == 0:
                                rs[i] -= 0.3   # student teammate went down
                            if atk is not None and atk == i \
                                    and g.units[victim].team != 0:
                                rs[i] += 1.0
                elif ev[0] == "round_end":
                    if ev[1] is not None:
                        for i in range(ns):
                            rs[i] += 2.0 if ev[1] == 0 else -2.0
            elif g.mode == "showdown":
                if ev[0] == "kill":
                    _, atk, victim = ev
                    if victim == 0:
                        rs[0] -= 1.5
                    elif atk == 0:
                        rs[0] += 1.5
            elif g.mode == "brawl_ball":
                if ev[0] == "hit":
                    _, atk, victim = ev
                    if atk == 0:
                        rs[0] += 0.05      # landed a shot
                    elif victim == 0:
                        rs[0] -= 0.05      # got hit
                elif ev[0] == "goal":
                    rs[0] += 1.0 if ev[1] == 0 else -1.0
            elif g.mode == "gem_grab" and self.cfg.reward_shaping:
                if ev[0] == "hit":
                    _, atk, victim = ev
                    if atk == 0:
                        rs[0] += 0.05      # landed a shot
                    elif victim == 0:
                        rs[0] -= 0.05      # got hit
                elif ev[0] == "kill":
                    _, atk, victim = ev
                    if victim == 0:
                        rs[0] -= 2.0       # died: the big penalty
                    elif atk == 0:
                        rs[0] += 1.0       # own kill
                    elif g.units[victim].team == 0:
                        rs[0] -= 0.3       # teammate died
                    elif atk is not None and g.units[atk].team == 0:
                        rs[0] += 0.3       # teammate's kill
        g.events = []

        if g.mode == "brawl_ball":
            carry_ours = g.ball_carrier is not None \
                and g.units[g.ball_carrier].team == 0
            if carry_ours and not self._last_carry_ours:
                rs[0] += 0.05
            self._last_carry_ours = carry_ours
        elif g.mode == "showdown" and g.units[0].alive:
            rs[0] += 0.002

        if g.mode == "gem_grab" and self.cfg.reward_shaping:
            u0 = g.units[0]
            if u0.alive:
                rs[0] += 0.002              # survival, ~+0.03/s
                rs[0] += 0.002 * u0.gems    # gem holding, ~+0.15/s at 5
                team = sum(u.gems for u in g.units if u.team == 0)
                if team >= g.cfg.gem_carry_to_win:
                    rs[0] += 0.02           # countdown turtle, ~+0.3/s
                st = ARCHETYPES[u0.archetype]
                rng_eff = st["melee_range"] if st["melee"] \
                    else st["proj_range"]
                d = min((float(np.linalg.norm(u.pos - u0.pos))
                         for u in g.units if u.team != 0 and u.alive),
                        default=np.inf)
                if rng_eff * 0.6 <= d <= rng_eff:
                    rs[0] += 0.003          # fighting at proper range

        if g.done and g.winner is not None:
            for i in range(ns):
                rs[i] += 5.0 if g.winner == 0 else -5.0
        return rs[0] if ns == 1 else rs

    def step(self, action):
        my_team = self.game.units[0].team
        # one action dict per student unit (0..n_students-1); a bare dict is
        # the legacy single-student form
        acts = list(action) if isinstance(action, (list, tuple)) else [action]
        # group the remaining units by policy so each policy does one batched call
        by_policy = {}
        for i in range(self.n_students, len(self.game.units)):
            u = self.game.units[i]
            policy = self.teammate_policy if u.team == my_team else self.opponent_policy
            by_policy.setdefault(id(policy), (policy, []))[1].append(i)
        pol_actions = {}
        for policy, idxs in by_policy.values():
            if hasattr(policy, "act_batch"):
                batch = policy.act_batch(self.game, idxs)
            else:
                batch = [policy.act(self.game, i) for i in idxs]
            for i, a in zip(idxs, batch):
                pol_actions[i] = a
        actions = [Action(move=a["move"], aim=a["aim"],
                          shoot=bool(a["shoot"]), use_super=bool(a["super"]))
                   for a in acts]
        actions += [pol_actions[i]
                    for i in range(self.n_students, len(self.game.units))]
        self.game.step(actions)

        reward = self._reward()
        terminated = self.game.done
        info = {"winner": self.game.winner, "t": self.game.t,
                "mode": self.game.mode}
        if self.game.mode == "gem_grab":
            info["gems"] = [sum(u.gems for u in self.game.units if u.team == t)
                            for t in (0, 1)]
        elif self.game.mode == "brawl_ball":
            info["scores"] = list(self.game.scores)
        elif self.game.mode == "knockout":
            info["round_wins"] = list(self.game.round_wins)
        return self._obs(), reward, terminated, False, info

    def render(self):
        return render(self.game, viewer_team=0)
