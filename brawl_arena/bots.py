"""Scripted baseline bot: objective-seeking, range-keeping, uses super,
retreats when hurt. Uses only information visible to its own team
(respects bush stealth), so it plays like a real player, not a wallhack.

Handles all four modes (gem_grab, brawl_ball, knockout, showdown).

Good enough to be a non-trivial opponent/teammate; replaceable by the
trained teacher policy later via the same `.act(game, idx) -> Action` API.
"""
from __future__ import annotations

import math

import numpy as np

from .core import ARCHETYPES, Action, Game


def _attack_range(st: dict) -> float:
    """Effective attack range: melee reach for the tank, else projectile range."""
    return st["melee_range"] if st["melee"] else st["proj_range"]


class ScriptedBot:
    def __init__(self, retreat_hp_frac: float = 0.35, seed: int | None = None):
        self.retreat_hp_frac = retreat_hp_frac
        self.rng = np.random.default_rng(seed)

    def act(self, game: Game, idx: int) -> Action:
        me = game.units[idx]
        if not me.alive:
            return Action(move=np.zeros(2), aim=np.array([1.0, 0.0]))
        if game.mode == "brawl_ball":
            return self._act_ball(game, idx)
        return self._act_combat(game, idx)

    def act_batch(self, game: Game, indices) -> list[Action]:
        return [self.act(game, i) for i in indices]

    def _steer(self, game: Game, u, mv: np.ndarray) -> np.ndarray:
        """Obstacle avoidance: if the desired heading is blocked, rotate it
        in growing steps until a free direction is found. Prevents bots from
        pressing into corners forever."""
        mv = mv / (np.linalg.norm(mv) + 1e-6)
        r = game.cfg.unit_radius
        probe = r + 0.75
        base = math.atan2(mv[1], mv[0])
        for off in (0.0, 0.6, -0.6, 1.2, -1.2, 1.9, -1.9, 2.6, -2.6, math.pi):
            a = base + off
            cand = np.array([math.cos(a), math.sin(a)])
            qx = u.pos[0] + cand[0] * probe
            qy = u.pos[1] + cand[1] * probe
            if not game.is_solid_for_unit(qx, qy):
                return cand
        return np.zeros(2)

    def _dodge(self, game: Game, me) -> np.ndarray | None:
        """Projectile dodge: find the enemy shot that will hit us soonest
        and return a unit vector perpendicular to its trajectory (toward
        the side we are already on, which is the shortest escape)."""
        cfg = game.cfg
        best = None
        for p in game.projectiles:
            if p.team == me.team:
                continue
            sp2 = float(np.dot(p.vel, p.vel))
            if sp2 < 1e-6:
                continue
            rel = me.pos - p.pos
            t = -float(np.dot(rel, p.vel)) / sp2   # time to closest approach
            if t < 0.0 or t > 0.45:
                continue
            miss = np.linalg.norm(rel + p.vel * t)
            if miss < cfg.unit_radius + p.radius + 0.25:
                perp = np.array([-p.vel[1], p.vel[0]]) / math.sqrt(sp2)
                side = np.sign(np.dot(rel, perp))
                cand = perp * (side if side != 0 else 1.0)
                if best is None or t < best[0]:
                    best = (t, cand)
        return best[1] if best is not None else None

    # --------------------------------------------------------- combat modes
    def _act_combat(self, game: Game, idx: int) -> Action:
        me = game.units[idx]
        st = ARCHETYPES[me.archetype]

        visible_enemies = [u for u in game.units
                           if u.team != me.team and u.alive
                           and game.is_visible_to(u, me.team)]
        target = min(visible_enemies, key=lambda u: np.linalg.norm(u.pos - me.pos),
                     default=None)
        nearest_gem = min(game.gems, key=lambda g: np.linalg.norm(g - me.pos),
                          default=None)
        enemy_dist = np.linalg.norm(target.pos - me.pos) if target is not None else np.inf
        gem_dist = np.linalg.norm(nearest_gem - me.pos) if nearest_gem is not None else np.inf

        # intent: grab nearby gems unless an enemy is right on top of us
        seeking_gem = nearest_gem is not None and gem_dist < enemy_dist - 1.0
        if seeking_gem:
            to_t = nearest_gem - me.pos
        elif target is not None:
            to_t = target.pos - me.pos
        else:
            to_t = np.array([game.cfg.map_w / 2.0, game.cfg.map_h / 2.0]) - me.pos
            seeking_gem = True
        dist = np.linalg.norm(to_t)

        # showdown: never linger outside the poison ring
        if game.mode == "showdown":
            center = np.array([game.cfg.map_w / 2.0, game.cfg.map_h / 2.0])
            if np.linalg.norm(me.pos - center) > game.poison_radius - 1.5:
                to_t = center - me.pos
                seeking_gem = True
                dist = np.linalg.norm(to_t)

        preferred = _attack_range(st) * 0.8
        # movement: turtle with gems only while OUR countdown is running
        # (or about to); otherwise a gem carrier parking at spawn forever is
        # just a free kill, so keep collecting/fighting instead
        team_gems = sum(u.gems for u in game.units if u.team == me.team)
        turtling = (me.gems >= 3 and not seeking_gem
                    and (game.countdown_team == me.team
                         or team_gems >= game.cfg.gem_carry_to_win))
        if turtling:
            home = me.spawn + np.array([1.5 if me.team == 0 else -1.5, 0.0])
            mv = home - me.pos
            if np.linalg.norm(mv) < 0.6:   # arrived: strafe instead of freezing
                mv = np.array([-mv[1], mv[0]])
        elif seeking_gem:
            mv = to_t
        elif me.hp < self.retreat_hp_frac * st["hp"]:
            mv = -to_t
        elif dist > preferred + 0.5:
            mv = to_t
        elif dist < preferred - 0.5:
            mv = -to_t
        else:
            perp = np.array([-to_t[1], to_t[0]])
            mv = perp if self.rng.random() < 0.9 else -perp
        dodge = self._dodge(game, me)
        if dodge is not None:
            mv = mv + 1.6 * dodge
        mv = self._steer(game, me, mv)

        aim = to_t + self.rng.normal(0, 0.05, 2)
        rng_eff = _attack_range(st)
        in_range = target is not None and enemy_dist < rng_eff * 0.95 + game.cfg.unit_radius
        shoot = bool(in_range and me.ammo >= 1.0)
        use_super = bool(me.super_charge >= 1.0 and in_range)
        return Action(move=mv, aim=aim, shoot=shoot, use_super=use_super)

    # ----------------------------------------------------------- brawl_ball
    def _act_ball(self, game: Game, idx: int) -> Action:
        me = game.units[idx]
        st = ARCHETYPES[me.archetype]
        goal = np.array([game.cfg.map_w - 0.5 if me.team == 0 else 0.5,
                         game.cfg.map_h / 2.0])

        if game.ball_carrier == idx:
            # carry toward the enemy goal, shoot when close enough
            to_goal = goal - me.pos
            d = np.linalg.norm(to_goal)
            mv = self._steer(game, me, to_goal / (d + 1e-6))
            shoot = bool(d < 5.0)   # pass kick flies ~4.7 tiles; walk it close
            use_super = bool(me.super_charge >= 1.0 and d < 12.0)
            return Action(move=mv, aim=to_goal, shoot=shoot, use_super=use_super)

        visible_enemies = [u for u in game.units
                           if u.team != me.team and u.alive
                           and game.is_visible_to(u, me.team)]
        target = min(visible_enemies, key=lambda u: np.linalg.norm(u.pos - me.pos),
                     default=None)

        if game.ball_carrier is None:
            # chase the loose ball, take potshots at enemies on the way
            to_t = game.ball_pos - me.pos
            mv = self._steer(game, me, to_t / (np.linalg.norm(to_t) + 1e-6))
        else:
            carrier = game.units[game.ball_carrier]
            if carrier.team != me.team and game.is_visible_to(carrier, me.team):
                target = carrier   # hunt the enemy carrier
            # support: drift toward the enemy goal side
            to_t = goal - me.pos if target is None else target.pos - me.pos
            d = np.linalg.norm(to_t)
            preferred = _attack_range(st) * 0.8
            if target is None or d > preferred + 0.5:
                mv = to_t
            elif d < preferred - 0.5:
                mv = -to_t
            else:
                perp = np.array([-to_t[1], to_t[0]])
                mv = perp if self.rng.random() < 0.9 else -perp
            mv = mv / (np.linalg.norm(mv) + 1e-6)
            dodge = self._dodge(game, me)
            if dodge is not None:
                mv = mv + 1.6 * dodge
            mv = self._steer(game, me, mv)

        if target is not None:
            aim_to = target.pos - me.pos
            enemy_dist = np.linalg.norm(aim_to)
        else:
            aim_to = to_t
            enemy_dist = np.inf
        aim = aim_to + self.rng.normal(0, 0.05, 2)
        rng_eff = _attack_range(st)
        in_range = target is not None and enemy_dist < rng_eff * 0.95 + game.cfg.unit_radius
        shoot = bool(in_range and me.ammo >= 1.0)
        use_super = bool(me.super_charge >= 1.0 and in_range)
        return Action(move=mv, aim=aim, shoot=shoot, use_super=use_super)
