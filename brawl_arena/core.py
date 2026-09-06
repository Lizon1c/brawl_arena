"""Core game logic for brawl-arena: a top-down arena shooter.

Original implementation modeled on publicly documented genre mechanics
(ammo-regen attacks, pellet spread, lobbed shots over walls, bush stealth,
destructible crates). No code, art or maps from any existing game are used.

Supported modes (all original designs):
    gem_grab   -- 3v3, collect gems from the central mine, hold 10 to win.
    brawl_ball -- 3v3, carry/kick the ball into the enemy goal zone.
    knockout   -- 3v3 rounds, no respawn, first to 2 round wins.
    showdown   -- 6-way free-for-all with a shrinking poison ring.
    duel       -- 1v1, mirrored archetype, knockout round machinery; the
                  pure-combat teacher: fast games, clean credit assignment.

World units are tiles (float coordinates). Map is a grid of tiles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TILE_EMPTY = 0
TILE_WALL = 1          # indestructible
TILE_BUSH = 2
TILE_CRATE = 3         # destructible by supers
TILE_GOAL = 4          # brawl_ball goal zone (blocks units, not the ball)
TILE_FENCE = 5         # blocks unit movement, projectiles fly over it

MODES = ("gem_grab", "brawl_ball", "knockout", "showdown", "duel")

MAX_UNITS = 6          # fixed unit slots so state_dim is constant across modes

# ---------------------------------------------------------------- archetypes
# Stats are our own tuning inspired by public genre conventions.
ARCHETYPES = {
    "shotgun": dict(
        hp=110.0, speed=3.4, ammo_regen=0.5, attack_cd=0.8,
        pellets=5, spread=0.28, proj_speed=10.0, proj_range=5.0,
        proj_damage=9.0, proj_radius=0.16,
        super_pellets=9, super_spread=0.55, super_damage=11.0,
        super_range=5.5, super_breaks_walls=True, lobbed=False,
        aoe=0.0, aoe_super=0.0,
        super_charge_per_damage=0.018,
        melee=False, melee_range=0.0, melee_arc=0.0, super_knockback=0.0,
    ),
    "sniper": dict(
        hp=70.0, speed=3.1, ammo_regen=0.3, attack_cd=1.0,
        pellets=1, spread=0.0, proj_speed=14.0, proj_range=9.5,
        proj_damage=34.0, proj_radius=0.12,
        super_pellets=1, super_spread=0.0, super_damage=60.0,
        super_range=12.0, super_breaks_walls=True, lobbed=False,
        aoe=0.0, aoe_super=0.0,
        super_charge_per_damage=0.016,
        melee=False, melee_range=0.0, melee_arc=0.0, super_knockback=0.0,
    ),
    "thrower": dict(
        hp=80.0, speed=3.2, ammo_regen=0.45, attack_cd=1.1,
        pellets=1, spread=0.0, proj_speed=7.0, proj_range=6.5,
        proj_damage=22.0, proj_radius=0.2,
        super_pellets=1, super_spread=0.0, super_damage=30.0,
        super_range=7.0, super_breaks_walls=True, lobbed=True,
        aoe=1.1, aoe_super=1.8,
        super_charge_per_damage=0.015,
        melee=False, melee_range=0.0, melee_arc=0.0, super_knockback=0.0,
    ),
    # tank: high hp, fast, short-range melee fan swing; super is a
    # point-blank AoE that damages and knocks enemies back.
    "tank": dict(
        hp=170.0, speed=3.6, ammo_regen=0.55, attack_cd=0.9,
        pellets=0, spread=0.0, proj_speed=0.0, proj_range=0.0,
        proj_damage=18.0, proj_radius=0.0,
        super_pellets=0, super_spread=0.0, super_damage=42.0,
        super_range=0.0, super_breaks_walls=True, lobbed=False,
        aoe=0.0, aoe_super=3.0,
        super_charge_per_damage=0.014,
        melee=True, melee_range=2.3, melee_arc=1.0, super_knockback=9.0,
    ),
}
ARCHETYPE_NAMES = list(ARCHETYPES.keys())


@dataclass
class Config:
    mode: str = "gem_grab"
    map_w: int = 21            # overridden from the sampled map at reset
    map_h: int = 15
    units_per_team: int = 3
    unit_radius: float = 0.42
    max_ammo: float = 3.0
    reveal_radius: float = 2.2        # bush reveal distance
    reveal_after_attack: float = 1.0  # s visible after firing
    respawn_time: float = 3.0
    hp_regen_delay: float = 3.0
    hp_regen_rate: float = 8.0
    time_limit: float = 180.0
    dt: float = 1.0 / 15.0
    render_scale: int = 6
    # gem_grab
    gem_spawn_interval: float = 8.0
    gem_carry_to_win: int = 10
    gem_countdown: float = 15.0
    # brawl_ball
    goals_to_win: int = 2
    ball_pass_speed: float = 7.5
    ball_super_speed: float = 12.0
    ball_friction: float = 6.0
    ball_carry_slow: float = 0.85
    ball_pickup_max_speed: float = 6.0
    ball_pickup_cd: float = 0.6
    # knockout
    rounds_to_win: int = 2
    max_rounds: int = 5
    round_time_limit: float = 60.0
    # handicap matches (e.g. (2, 1): two units vs one -- the duo curriculum);
    # None = units_per_team on both sides. Only used by team-based modes.
    team_sizes: tuple | None = None
    # showdown
    n_showdown_units: int = 6
    poison_start_delay: float = 15.0
    poison_shrink_interval: float = 8.0
    poison_shrink_step: float = 1.5
    poison_min_radius: float = 1.5
    poison_dps: float = 10.0
    knockback_decay: float = 8.0
    # reward shaping (gym_env): extra behaviour-prior rewards on top of the
    # sparse win/loss signal; off by default to keep env semantics stable
    reward_shaping: bool = False


@dataclass
class Unit:
    team: int
    archetype: str
    pos: np.ndarray
    spawn: np.ndarray
    hp: float
    ammo: float = 3.0
    super_charge: float = 0.0
    attack_cd: float = 0.0
    alive: bool = True
    respawn_t: float = 0.0
    since_damage: float = 99.0
    since_attack: float = 99.0
    gems: int = 0
    kb: np.ndarray = field(default_factory=lambda: np.zeros(2))  # knockback vel
    look: dict = field(default_factory=dict)  # procedural appearance, set at reset


@dataclass
class Projectile:
    team: int
    owner: int
    pos: np.ndarray
    vel: np.ndarray
    damage: float
    range_left: float
    radius: float
    aoe: float = 0.0
    lobbed: bool = False
    breaks_walls: bool = False
    is_super: bool = False
    range_total: float = 0.0     # initial range, for the render arc only


@dataclass
class Action:
    move: np.ndarray                # (2,) in [-1, 1]
    aim: np.ndarray                 # (2,) direction
    shoot: bool = False
    use_super: bool = False


class Game:
    def __init__(self, config: Config | None = None, seed: int | None = None):
        self.cfg = config or Config()
        assert self.cfg.mode in MODES, f"unknown mode {self.cfg.mode!r}"
        self.mode = self.cfg.mode
        self.rng = np.random.default_rng(seed)
        self.tiles: np.ndarray
        self.units: list[Unit] = []
        self.projectiles: list[Projectile] = []
        self.gems: list[np.ndarray] = []
        self.events: list[tuple] = []   # ("kill", atk_idx|None, victim) etc.
        self.gem_timer = 0.0
        self.countdown_team: int | None = None
        self.countdown_t = 0.0
        # brawl_ball
        self.scores = [0, 0]
        self.ball_pos = np.zeros(2)
        self.ball_vel = np.zeros(2)
        self.ball_carrier: int | None = None
        self.ball_cd: list[float] = []
        # knockout
        self.round_wins = [0, 0]
        self.round_num = 1
        self.round_t = 0.0
        # showdown
        self.poison_radius = 1.0
        self.poison_max = 1.0
        self.poison_timer = 0.0
        self.t = 0.0
        self.done = False
        self.winner: int | None = None
        self.map_spawns: list | None = None
        self.reset()

    # ------------------------------------------------------------------ map
    def _load_map(self) -> np.ndarray:
        from .maps import sample_map
        tiles, spawns = sample_map(self.mode, self.rng)
        self.map_spawns = spawns   # custom showdown spawn tiles, or None
        return tiles

    # ---------------------------------------------------------------- reset
    def reset(self):
        self.tiles = self._load_map()
        self.cfg.map_h, self.cfg.map_w = self.tiles.shape
        cfg = self.cfg
        w, h = cfg.map_w, cfg.map_h
        self.t = 0.0
        self.projectiles = []
        self.gems = []
        self.events = []
        self.gem_timer = 0.0
        self.countdown_team = None
        self.countdown_t = 0.0
        self.scores = [0, 0]
        self.round_wins = [0, 0]
        self.round_num = 1
        self.round_t = 0.0
        self.done = False
        self.winner = None
        self.units = []

        if self.mode == "showdown":
            n = cfg.n_showdown_units
            if self.map_spawns:
                spots = [(x + 0.5, y + 0.5) for x, y in self.map_spawns]
            else:
                # symmetric about the true map centre (w/2, h/2)
                spots = [(2.5, 2.5), (w - 2.5, 2.5), (2.5, h - 2.5),
                         (w - 2.5, h - 2.5), (w / 2.0, 2.5), (w / 2.0, h - 2.5)]
            for i in range(n):
                x, y = spots[i % len(spots)]
                self._clear_area(x, y, 1)
                arch = ARCHETYPE_NAMES[i % len(ARCHETYPE_NAMES)]
                spawn = np.array([x, y], dtype=np.float64)
                self.units.append(Unit(team=i, archetype=arch, pos=spawn.copy(),
                                       spawn=spawn, hp=ARCHETYPES[arch]["hp"],
                                       ammo=cfg.max_ammo))
            cx, cy = w / 2.0, h / 2.0
            self.poison_max = math.hypot(cx, cy) + 1.0
            self.poison_radius = self.poison_max
            self.poison_timer = cfg.poison_start_delay
        elif self.mode == "duel":
            # 1v1 mirror match: one unit per team, SAME archetype on both
            # sides (pure skill, no counter-pick), knockout round machinery.
            if self.map_spawns:
                sides = [[], []]
                for sx, sy in self.map_spawns:
                    sides[0 if sx < w / 2 else 1].append((sx + 0.5, sy + 0.5))
                spots = [sides[0][0], sides[1][0]]
            else:
                spots = [(2.5, h / 2.0), (w - 2.5, h / 2.0)]
            arch = ARCHETYPE_NAMES[int(self.rng.integers(len(ARCHETYPE_NAMES)))]
            for team in (0, 1):
                x, y = spots[team]
                self._clear_area(x, y, 1)
                spawn = np.array([x, y], dtype=np.float64)
                self.units.append(Unit(team=team, archetype=arch,
                                       pos=spawn.copy(), spawn=spawn,
                                       hp=ARCHETYPES[arch]["hp"],
                                       ammo=cfg.max_ammo))
        else:
            n0, n1 = (cfg.team_sizes if cfg.team_sizes
                      else (cfg.units_per_team, cfg.units_per_team))
            n = max(n0, n1)
            # same random roster for both teams keeps the matchup fair
            roster = [ARCHETYPE_NAMES[i] for i in
                      self.rng.permutation(len(ARCHETYPE_NAMES))[:n]]
            if self.map_spawns and self.mode in ("brawl_ball", "knockout"):
                # custom spawn markers: 3 per side, team by map half
                sides = [[], []]
                for sx, sy in self.map_spawns:
                    sides[0 if sx < w / 2 else 1].append((sx + 0.5, sy + 0.5))
                for team in (0, 1):
                    spots = sorted(sides[team], key=lambda p: p[1])
                    for i in range((n0, n1)[team]):
                        x, y = spots[i % len(spots)]
                        self._clear_area(x, y, 1)
                        spawn = np.array([x, y], dtype=np.float64)
                        self.units.append(Unit(team=team, archetype=roster[i],
                                               pos=spawn.copy(), spawn=spawn,
                                               hp=ARCHETYPES[roster[i]]["hp"],
                                               ammo=cfg.max_ammo))
            else:
                # symmetric about the true map centre (w/2, h/2): ball and
                # gem mine sit at (w/2, h/2), so spawns must too, otherwise
                # one side is permanently closer to the mid objective
                # clear the spawn columns (keep bushes). Units spawn at x=2 /
                # x=w-3 with radius ~0.42, so their body reaches one column
                # further out; clear 3 columns per side to avoid spawning
                # embedded in a wall.
                for cols in ((1, 4), (w - 4, w - 1)):
                    sub = self.tiles[:, cols[0]:cols[1]]
                    sub[(sub == TILE_WALL) | (sub == TILE_CRATE)
                        | (sub == TILE_FENCE)] = TILE_EMPTY
                for team in (0, 1):
                    nt = (n0, n1)[team]
                    x = 2.5 if team == 0 else w - 2.5
                    ys = np.linspace(2.5, h - 2.5, nt + 2)[1:-1]
                    for i in range(nt):
                        spawn = np.array([x, ys[i]], dtype=np.float64)
                        self.units.append(Unit(team=team, archetype=roster[i],
                                               pos=spawn.copy(), spawn=spawn,
                                               hp=ARCHETYPES[roster[i]]["hp"],
                                               ammo=cfg.max_ammo))

        # brawl_ball: ball starts at the centre of a cleared area
        self.ball_pos = np.array([w / 2.0, h / 2.0], dtype=np.float64)
        self.ball_vel = np.zeros(2)
        self.ball_carrier = None
        self.ball_cd = [0.0] * MAX_UNITS
        if self.mode == "brawl_ball":
            cx, cy = w // 2, h // 2
            self.tiles[cy - 1:cy + 2, cx - 1:cx + 2] = TILE_EMPTY

        # procedural per-unit appearance, fixed for this episode
        for u in self.units:
            u.look = {"radius": cfg.unit_radius * float(self.rng.uniform(0.88, 1.12)),
                      "color": float(self.rng.random())}  # palette fraction

    def _clear_area(self, x: float, y: float, r: int):
        xi, yi = math.floor(x), math.floor(y)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                xx, yy = xi + dx, yi + dy
                if 0 <= xx < self.cfg.map_w and 0 <= yy < self.cfg.map_h:
                    if self.tiles[yy, xx] in (TILE_WALL, TILE_CRATE, TILE_FENCE):
                        self.tiles[yy, xx] = TILE_EMPTY

    def _reset_positions(self, ball: bool = False):
        """Round/goal reset: everyone back to spawn, full hp, clear shots."""
        cfg = self.cfg
        for u in self.units:
            u.pos = u.spawn.copy()
            u.hp = ARCHETYPES[u.archetype]["hp"]
            u.ammo = cfg.max_ammo
            u.alive = True
            u.respawn_t = 0.0
            u.super_charge = 0.0
            u.attack_cd = 0.0
            u.since_damage = 99.0
            u.since_attack = 99.0
            u.kb = np.zeros(2)
        self.projectiles = []
        if ball:
            self.ball_pos = np.array([cfg.map_w / 2.0, cfg.map_h / 2.0])
            self.ball_vel = np.zeros(2)
            self.ball_carrier = None
            self.ball_cd = [0.0] * MAX_UNITS

    # --------------------------------------------------------------- helpers
    def is_solid_for_unit(self, x: float, y: float) -> bool:
        """Movement collision: walls, crates, goal zones and fences."""
        xi, yi = math.floor(x), math.floor(y)
        if xi < 0 or yi < 0 or xi >= self.cfg.map_w or yi >= self.cfg.map_h:
            return True
        return self.tiles[yi, xi] in (TILE_WALL, TILE_CRATE, TILE_GOAL, TILE_FENCE)

    def blocks_projectile(self, x: float, y: float) -> bool:
        """Projectile collision: walls, crates, goal zones (fences let
        projectiles fly over)."""
        xi, yi = math.floor(x), math.floor(y)
        if xi < 0 or yi < 0 or xi >= self.cfg.map_w or yi >= self.cfg.map_h:
            return True
        return self.tiles[yi, xi] in (TILE_WALL, TILE_CRATE, TILE_GOAL)

    def _break_tile(self, x: float, y: float):
        xi, yi = math.floor(x), math.floor(y)
        if 0 <= xi < self.cfg.map_w and 0 <= yi < self.cfg.map_h:
            if self.tiles[yi, xi] == TILE_CRATE:
                self.tiles[yi, xi] = TILE_EMPTY

    def is_visible_to(self, unit: Unit, team: int) -> bool:
        """Stealth rule: hidden in bush unless an enemy is close or it fired recently."""
        xi, yi = math.floor(unit.pos[0]), math.floor(unit.pos[1])
        if self.tiles[yi, xi] != TILE_BUSH:
            return True
        if unit.since_attack < self.cfg.reveal_after_attack:
            return True
        for u in self.units:
            if u.team == team and u.alive:
                if np.linalg.norm(u.pos - unit.pos) < self.cfg.reveal_radius:
                    return True
        return False

    def _move_unit(self, u: Unit, delta: np.ndarray):
        r = self.cfg.unit_radius
        nx = u.pos[0] + delta[0]
        if not (self.is_solid_for_unit(nx - r, u.pos[1] - r)
                or self.is_solid_for_unit(nx + r, u.pos[1] - r)
                or self.is_solid_for_unit(nx - r, u.pos[1] + r)
                or self.is_solid_for_unit(nx + r, u.pos[1] + r)):
            u.pos[0] = nx
        ny = u.pos[1] + delta[1]
        if not (self.is_solid_for_unit(u.pos[0] - r, ny - r)
                or self.is_solid_for_unit(u.pos[0] + r, ny - r)
                or self.is_solid_for_unit(u.pos[0] - r, ny + r)
                or self.is_solid_for_unit(u.pos[0] + r, ny + r)):
            u.pos[1] = ny

    # ----------------------------------------------------------------- step
    def step(self, actions: list[Action]):
        if self.done:
            return
        self.t += self.cfg.dt
        self._step_units(actions)
        self._step_projectiles()
        if self.mode == "gem_grab":
            self._step_gem_grab()
        elif self.mode == "brawl_ball":
            self._step_brawl_ball()
        elif self.mode in ("knockout", "duel"):
            self._step_knockout()
        elif self.mode == "showdown":
            self._step_showdown()

    def _step_units(self, actions: list[Action]):
        cfg = self.cfg
        dt = cfg.dt
        can_respawn = self.mode in ("gem_grab", "brawl_ball")
        for i, (u, a) in enumerate(zip(self.units, actions)):
            if not u.alive:
                if can_respawn:
                    u.respawn_t -= dt
                    if u.respawn_t <= 0:
                        u.alive = True
                        u.hp = ARCHETYPES[u.archetype]["hp"]
                        u.pos = u.spawn.copy()
                continue
            st = ARCHETYPES[u.archetype]
            mv = np.asarray(a.move, dtype=np.float64)
            norm = np.linalg.norm(mv)
            if norm > 1e-6:
                mv = mv / max(norm, 1.0)
                # carrying the ball slows you down so defenders can catch up
                speed = st["speed"]
                if self.mode == "brawl_ball" and self.ball_carrier == i:
                    speed *= cfg.ball_carry_slow
                self._move_unit(u, mv * speed * dt)
            # knockback impulse decays exponentially
            kb_norm = np.linalg.norm(u.kb)
            if kb_norm > 1e-3:
                self._move_unit(u, u.kb * dt)
                u.kb *= max(0.0, 1.0 - cfg.knockback_decay * dt)
            else:
                u.kb[:] = 0.0
            u.attack_cd = max(0.0, u.attack_cd - dt)
            u.ammo = min(cfg.max_ammo, u.ammo + st["ammo_regen"] * dt)
            u.since_damage += dt
            u.since_attack += dt
            if u.since_damage > cfg.hp_regen_delay:
                u.hp = min(st["hp"], u.hp + cfg.hp_regen_rate * dt)
            self.ball_cd[i] = max(0.0, self.ball_cd[i] - dt)

            aim = np.asarray(a.aim, dtype=np.float64)
            anorm = np.linalg.norm(aim)
            if anorm <= 1e-6:
                continue
            # ball carrier cannot attack; shoot = pass, super = power shot
            if self.mode == "brawl_ball" and self.ball_carrier == i:
                if a.use_super and u.super_charge >= 1.0:
                    self._kick_ball(i, aim / anorm, cfg.ball_super_speed)
                    u.super_charge = 0.0
                elif a.shoot:
                    self._kick_ball(i, aim / anorm, cfg.ball_pass_speed)
                continue
            if u.attack_cd <= 0 and u.ammo >= 1.0:
                if a.use_super and u.super_charge >= 1.0:
                    self._fire(u, i, aim, st, is_super=True)
                elif a.shoot:
                    self._fire(u, i, aim, st, is_super=False)

    def _step_projectiles(self):
        cfg = self.cfg
        dt = cfg.dt
        alive_proj = []
        for p in self.projectiles:
            step = np.linalg.norm(p.vel) * dt
            p.pos = p.pos + p.vel * dt
            p.range_left -= step
            landed = p.range_left <= 0
            if self.blocks_projectile(p.pos[0], p.pos[1]):
                if p.lobbed and not landed:
                    alive_proj.append(p)      # lobs fly over walls
                elif p.breaks_walls:
                    self._break_tile(p.pos[0], p.pos[1])
                    if p.aoe > 0:
                        self._aoe_damage(p)
                elif p.aoe > 0:
                    self._aoe_damage(p)
                continue
            if landed:
                if p.aoe > 0:
                    self._aoe_damage(p)
                continue
            hit = False
            for u in self.units:
                if not u.alive or u.team == p.team:
                    continue
                if np.linalg.norm(u.pos - p.pos) < cfg.unit_radius + p.radius:
                    self._damage(u, p.damage, p.owner)
                    hit = True
                    if not p.is_super:
                        break
            if not hit or p.is_super:
                alive_proj.append(p)
        self.projectiles = alive_proj

    # -------------------------------------------------------------- combat
    def _fire(self, u: Unit, idx: int, aim: np.ndarray, st: dict, is_super: bool):
        base = math.atan2(aim[1], aim[0])
        if is_super and st["super_knockback"] > 0:
            # tank super: point-blank AoE damage + knockback, breaks crates
            u.super_charge = 0.0
            u.ammo -= 1.0
            u.attack_cd = st["attack_cd"]
            u.since_attack = 0.0
            r = int(math.ceil(st["aoe_super"]))
            cx, cy = math.floor(u.pos[0]), math.floor(u.pos[1])
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        self._break_tile(cx + dx, cy + dy)
            for v in self.units:
                if not v.alive or v.team == u.team:
                    continue
                d = v.pos - u.pos
                dist = np.linalg.norm(d)
                if dist < st["aoe_super"]:
                    self._damage(v, st["super_damage"], idx)
                    if dist > 1e-6:
                        v.kb += d / dist * st["super_knockback"]
            return
        if st["melee"] and not is_super:
            # tank swing: instant fan-shaped melee hit, no projectile
            u.ammo -= 1.0
            u.attack_cd = st["attack_cd"]
            u.since_attack = 0.0
            for v in self.units:
                if not v.alive or v.team == u.team:
                    continue
                d = v.pos - u.pos
                dist = np.linalg.norm(d)
                if dist > st["melee_range"] + self.cfg.unit_radius:
                    continue
                if dist < 1e-6 or abs(_ang_diff(math.atan2(d[1], d[0]), base)) < st["melee_arc"]:
                    self._damage(v, st["proj_damage"], idx)
            return
        u.super_charge = 0.0 if is_super else u.super_charge
        u.ammo -= 1.0
        u.attack_cd = st["attack_cd"]
        u.since_attack = 0.0
        n = st["super_pellets"] if is_super else st["pellets"]
        spread = st["super_spread"] if is_super else st["spread"]
        dmg = st["super_damage"] if is_super else st["proj_damage"]
        rng = st["super_range"] if is_super else st["proj_range"]
        aoe = st["aoe_super"] if is_super else st["aoe"]
        for k in range(n):
            ang = base + (k - (n - 1) / 2.0) * spread + float(self.rng.normal(0, 0.02))
            d = np.array([math.cos(ang), math.sin(ang)])
            self.projectiles.append(Projectile(
                team=u.team, owner=idx, pos=u.pos.copy(),
                vel=d * st["proj_speed"], damage=dmg, range_left=rng,
                radius=st["proj_radius"], aoe=aoe, lobbed=st["lobbed"],
                breaks_walls=is_super and st["super_breaks_walls"],
                is_super=is_super, range_total=rng))

    def _damage(self, u: Unit, dmg: float, attacker_idx: int | None):
        u.hp -= dmg
        u.since_damage = 0.0
        if attacker_idx is not None:
            self.events.append(("hit", attacker_idx,
                                self.units.index(u)))
            attacker = self.units[attacker_idx]
            st = ARCHETYPES[attacker.archetype]
            attacker.super_charge = min(
                1.0, attacker.super_charge + dmg * st["super_charge_per_damage"])
        if u.hp <= 0 and u.alive:
            u.alive = False
            u.respawn_t = self.cfg.respawn_time
            u.super_charge = 0.0
            u.kb[:] = 0.0
            victim_idx = self.units.index(u)
            self.events.append(("kill", attacker_idx, victim_idx))
            if self.mode == "gem_grab":
                # drop carried gems on death
                for _ in range(u.gems):
                    jitter = self.rng.normal(0, 0.5, 2)
                    self.gems.append(u.pos + jitter)
            u.gems = 0

    def _aoe_damage(self, p: Projectile):
        if p.breaks_walls:
            r = int(math.ceil(p.aoe))
            cx, cy = math.floor(p.pos[0]), math.floor(p.pos[1])
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        self._break_tile(cx + dx, cy + dy)
        for u in self.units:
            if not u.alive or u.team == p.team:
                continue
            if np.linalg.norm(u.pos - p.pos) < p.aoe:
                self._damage(u, p.damage, p.owner)

    # ------------------------------------------------------------ gem_grab
    def _step_gem_grab(self):
        cfg = self.cfg
        dt = cfg.dt
        self.gem_timer += dt
        if self.gem_timer >= cfg.gem_spawn_interval:
            self.gem_timer = 0.0
            self.gems.append(np.array([cfg.map_w / 2.0, cfg.map_h / 2.0],
                                      dtype=np.float64))
        for u in self.units:
            if not u.alive:
                continue
            keep = []
            for g in self.gems:
                if np.linalg.norm(g - u.pos) < cfg.unit_radius + 0.3:
                    u.gems += 1
                else:
                    keep.append(g)
            self.gems = keep

        held = [sum(u.gems for u in self.units if u.team == t) for t in (0, 1)]
        leader = None
        if held[0] >= cfg.gem_carry_to_win:
            leader = 0
        elif held[1] >= cfg.gem_carry_to_win:
            leader = 1
        if leader is not None:
            if self.countdown_team != leader:
                self.countdown_team = leader
                self.countdown_t = cfg.gem_countdown
            else:
                self.countdown_t -= dt
                if self.countdown_t <= 0:
                    self.done = True
                    self.winner = leader
        else:
            self.countdown_team = None
            self.countdown_t = 0.0

        if not self.done and self.t >= cfg.time_limit:
            self.done = True
            if held[0] != held[1]:
                self.winner = 0 if held[0] > held[1] else 1

    # ----------------------------------------------------------- brawl_ball
    def _kick_ball(self, idx: int, direction: np.ndarray, speed: float):
        u = self.units[idx]
        self.ball_carrier = None
        self.ball_pos = u.pos + direction * (self.cfg.unit_radius + 0.3)
        self.ball_vel = direction * speed
        self.ball_cd[idx] = self.cfg.ball_pickup_cd
        u.since_attack = 0.0

    def _step_brawl_ball(self):
        cfg = self.cfg
        dt = cfg.dt
        if self.ball_carrier is not None:
            carrier = self.units[self.ball_carrier]
            if not carrier.alive:
                self.ball_carrier = None   # ball drops where the carrier died
            else:
                self.ball_pos = carrier.pos.copy()
        else:
            sp = np.linalg.norm(self.ball_vel)
            if sp > 1e-6:
                new = self.ball_pos + self.ball_vel * dt
                xi, yi = math.floor(new[0]), math.floor(new[1])
                tile = self.tiles[yi, xi] if (0 <= xi < cfg.map_w
                                              and 0 <= yi < cfg.map_h) else TILE_WALL
                if tile == TILE_GOAL:
                    self.ball_pos = new
                elif tile in (TILE_WALL, TILE_CRATE):
                    self.ball_vel = np.zeros(2)
                else:
                    self.ball_pos = new
                    sp_new = max(0.0, sp - cfg.ball_friction * dt)
                    self.ball_vel *= sp_new / sp
            # goal?
            xi, yi = int(self.ball_pos[0]), int(self.ball_pos[1])
            if 0 <= xi < cfg.map_w and 0 <= yi < cfg.map_h \
                    and self.tiles[yi, xi] == TILE_GOAL:
                team = 1 if self.ball_pos[0] < cfg.map_w / 2.0 else 0
                self.scores[team] += 1
                self.events.append(("goal", team))
                if self.scores[team] >= cfg.goals_to_win:
                    self.done = True
                    self.winner = team
                else:
                    self._reset_positions(ball=True)
                return
            # pickup of a slow free ball: closest unit wins; exact ties
            # (mirrored positions) are broken at random, not by unit index
            if np.linalg.norm(self.ball_vel) < cfg.ball_pickup_max_speed:
                cands = []
                for i, u in enumerate(self.units):
                    if not u.alive or self.ball_cd[i] > 0:
                        continue
                    d = np.linalg.norm(self.ball_pos - u.pos)
                    if d < cfg.unit_radius + 0.35:
                        cands.append((d, i))
                if cands:
                    cands.sort()
                    d0 = cands[0][0]
                    tied = [i for d, i in cands if d <= d0 + 1e-9]
                    self.ball_carrier = tied[int(self.rng.integers(len(tied)))]
                    self.ball_vel = np.zeros(2)

        if not self.done and self.t >= cfg.time_limit:
            self.done = True
            if self.scores[0] != self.scores[1]:
                self.winner = 0 if self.scores[0] > self.scores[1] else 1

    # ------------------------------------------------------------ knockout
    def _step_knockout(self):
        cfg = self.cfg
        dt = cfg.dt
        self.round_t += dt
        alive = [sum(1 for u in self.units if u.team == t and u.alive)
                 for t in (0, 1)]
        decided = False
        round_winner = None
        if alive[0] == 0 or alive[1] == 0:
            decided = True
            round_winner = 1 if alive[0] == 0 else (0 if alive[1] == 0 else None)
        elif self.round_t >= cfg.round_time_limit:
            decided = True
            if alive[0] != alive[1]:
                round_winner = 0 if alive[0] > alive[1] else 1
            else:
                hp = [sum(u.hp for u in self.units if u.team == t) for t in (0, 1)]
                if hp[0] != hp[1]:
                    round_winner = 0 if hp[0] > hp[1] else 1
        if not decided:
            return
        self.events.append(("round_end", round_winner))
        if round_winner is not None:
            self.round_wins[round_winner] += 1
        game_over = (round_winner is not None
                     and self.round_wins[round_winner] >= cfg.rounds_to_win) \
            or self.round_num >= cfg.max_rounds
        if game_over:
            self.done = True
            if self.round_wins[0] != self.round_wins[1]:
                self.winner = 0 if self.round_wins[0] > self.round_wins[1] else 1
        else:
            self.round_num += 1
            self.round_t = 0.0
            self._reset_positions()

    # ------------------------------------------------------------ showdown
    def _step_showdown(self):
        cfg = self.cfg
        dt = cfg.dt
        self.poison_timer -= dt
        if self.poison_timer <= 0:
            self.poison_timer = cfg.poison_shrink_interval
            self.poison_radius = max(cfg.poison_min_radius,
                                     self.poison_radius - cfg.poison_shrink_step)
        center = np.array([cfg.map_w / 2.0, cfg.map_h / 2.0])
        for u in self.units:
            if u.alive and np.linalg.norm(u.pos - center) > self.poison_radius:
                self._damage(u, cfg.poison_dps * dt, None)
        alive = [u for u in self.units if u.alive]
        if len(alive) <= 1:
            self.done = True
            self.winner = alive[0].team if alive else None
        elif self.t >= cfg.time_limit:
            self.done = True
            best = max(alive, key=lambda u: u.hp)
            self.winner = best.team

    # --------------------------------------------------------- observations
    def state_vector(self, idx: int) -> np.ndarray:
        """Privileged full-state vector from the perspective of unit `idx`
        (sees everything, including stealthed units -- that is the point of
        the privileged teacher). Fixed length across all modes."""
        cfg = self.cfg
        me = self.units[idx]
        out = [self.t / cfg.time_limit, me.team / (MAX_UNITS - 1.0)]
        out += [me.pos[0] / cfg.map_w, me.pos[1] / cfg.map_h,
                me.hp / 180.0, me.ammo / cfg.max_ammo,
                me.super_charge, float(me.alive), me.gems / 10.0,
                self.countdown_t / cfg.gem_countdown]
        for name in ARCHETYPE_NAMES:
            out.append(1.0 if me.archetype == name else 0.0)
        for m in MODES:
            out.append(1.0 if self.mode == m else 0.0)
        # other units, ego-relative, zero-padded to MAX_UNITS - 1 slots
        others = [u for j, u in enumerate(self.units) if j != idx]
        for u in others[:MAX_UNITS - 1]:
            rel = (u.pos - me.pos) / np.array([cfg.map_w, cfg.map_h])
            out += [rel[0], rel[1], u.hp / 180.0, u.ammo / cfg.max_ammo,
                    u.super_charge, float(u.alive), float(u.team == me.team),
                    u.gems / 10.0]
            for name in ARCHETYPE_NAMES:
                out.append(1.0 if u.archetype == name else 0.0)
        out += [0.0] * (MAX_UNITS - 1 - len(others[:MAX_UNITS - 1])) \
            * (8 + len(ARCHETYPE_NAMES))
        for p in self.projectiles[:16]:
            rel = (p.pos - me.pos) / np.array([cfg.map_w, cfg.map_h])
            out += [rel[0], rel[1], p.vel[0] / 10.0, p.vel[1] / 10.0,
                    float(p.team == me.team)]
        out += [0.0] * (16 - len(self.projectiles[:16])) * 5
        for g in self.gems[:8]:
            rel = (g - me.pos) / np.array([cfg.map_w, cfg.map_h])
            out += [rel[0], rel[1]]
        out += [0.0] * (8 - len(self.gems[:8])) * 2
        out += self._mode_fields(idx)
        mine, enemy = self._relative_score(me.team)
        out += [mine, enemy]
        return np.asarray(out, dtype=np.float32)

    def _relative_score(self, team: int) -> tuple[float, float]:
        """(mine, enemy) score in [0, 1]-ish, interpreted per mode."""
        cfg = self.cfg
        if self.mode == "gem_grab":
            held = [sum(u.gems for u in self.units if u.team == t) for t in (0, 1)]
            mine = held[team] / cfg.gem_carry_to_win
            enemy = held[1 - team] / cfg.gem_carry_to_win
        elif self.mode == "brawl_ball":
            mine = self.scores[team] / cfg.goals_to_win
            enemy = self.scores[1 - team] / cfg.goals_to_win
        elif self.mode in ("knockout", "duel"):
            mine = self.round_wins[team] / cfg.rounds_to_win
            enemy = self.round_wins[1 - team] / cfg.rounds_to_win
        else:
            mine = enemy = 0.0
        return mine, enemy

    def _mode_fields(self, idx: int) -> list[float]:
        """12 mode-specific floats (zero-padded); mode is one-hot elsewhere."""
        cfg = self.cfg
        me = self.units[idx]
        mf = [0.0] * 12
        if self.mode == "gem_grab":
            mf[0] = self.countdown_t / cfg.gem_countdown
            mf[1] = float(self.countdown_team is not None)
            mf[2] = len(self.gems) / 8.0
        elif self.mode == "brawl_ball":
            rel = (self.ball_pos - me.pos) / np.array([cfg.map_w, cfg.map_h])
            mf[0], mf[1] = rel[0], rel[1]
            mf[2], mf[3] = self.ball_vel[0] / 16.0, self.ball_vel[1] / 16.0
            mf[4] = float(self.ball_carrier is not None)
            if self.ball_carrier is not None:
                mf[5] = float(self.ball_carrier == idx)
                mf[6] = float(self.units[self.ball_carrier].team == me.team)
        elif self.mode in ("knockout", "duel"):
            mf[0] = self.round_num / cfg.max_rounds
            mf[1] = self.round_t / cfg.round_time_limit
            n_team = max(1, sum(1 for u in self.units if u.team == me.team))
            n_foe = max(1, sum(1 for u in self.units if u.team != me.team))
            mf[2] = sum(1 for u in self.units
                        if u.team == me.team and u.alive) / n_team
            mf[3] = sum(1 for u in self.units
                        if u.team != me.team and u.alive) / n_foe
        elif self.mode == "showdown":
            center = np.array([cfg.map_w / 2.0, cfg.map_h / 2.0])
            mf[0] = self.poison_radius / self.poison_max
            mf[1] = self.poison_timer / cfg.poison_shrink_interval
            mf[2] = sum(1 for u in self.units if u.alive) / MAX_UNITS
            rel = (center - me.pos) / np.array([cfg.map_w, cfg.map_h])
            mf[3], mf[4] = rel[0], rel[1]
        return mf

    @property
    def state_dim(self) -> int:
        return (2 + 8 + len(ARCHETYPE_NAMES) + len(MODES)
                + (MAX_UNITS - 1) * (8 + len(ARCHETYPE_NAMES))
                + 16 * 5 + 8 * 2 + 12 + 2)


def _ang_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi
