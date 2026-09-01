"""Numpy top-down renderer for brawl-arena.

Produces an RGB uint8 frame of shape (map_h*scale, map_w*scale, 3).
Rendered from `viewer_team`'s perspective: enemies hidden in bushes
(per the stealth rule) are not drawn, which is what the vision-only
student has to work with.

Unit appearance is procedurally randomized per episode (radius jitter,
random body colour from a palette, class marker above the head) so the
vision student cannot overfit to fixed colours. The team ring around each
unit still identifies friend/foe.
"""
from __future__ import annotations

import numpy as np

from .core import (ARCHETYPES, TILE_WALL, TILE_BUSH, TILE_CRATE, TILE_EMPTY,
                   TILE_FENCE, TILE_GOAL, Game)

COLORS = {
    "floor": (28, 32, 38),
    "wall": (120, 120, 128),
    "crate": (150, 100, 60),
    "bush": (36, 96, 44),
    "goal": (86, 76, 130),
    "proj0": (140, 190, 255),
    "proj1": (255, 150, 140),
    "shadow": (12, 14, 16),
    "gem": (200, 60, 220),
    "ball": (240, 240, 240),
    "ball_edge": (90, 90, 90),
    "hp_bg": (60, 60, 60),
    "hp_fg": (90, 220, 90),
    "super_ready": (255, 230, 120),
    "marker": (245, 245, 245),
    "poison": (120, 30, 50),
    "poison_edge": (220, 70, 100),
}

# random body colours (team is shown by the ring, so these can be anything)
PALETTE = [
    (235, 90, 80), (80, 160, 240), (240, 200, 70), (150, 100, 230),
    (90, 200, 140), (240, 140, 60), (200, 90, 180), (120, 200, 220),
]

# ring colour by team: fixed blue/red for team modes, per-index for showdown
TEAM_RINGS = [(70, 140, 240), (240, 80, 70), (240, 200, 70),
              (150, 100, 230), (90, 200, 140), (240, 140, 60)]

# class marker shapes above the head
MARKERS = {"shotgun": "square", "sniper": "triangle",
           "thrower": "circle", "tank": "cross"}


# --------------------------------------------------------- tile textures
_TEX_CACHE: dict[int, dict[int, np.ndarray]] = {}


def tile_textures(scale: int) -> dict[int, np.ndarray]:
    """Pre-generated scale x scale RGB tiles, one per tile type.

    Generated once per scale with a fixed seed, so frames never flicker."""
    if scale in _TEX_CACHE:
        return _TEX_CACHE[scale]
    rng = np.random.default_rng(20240601)
    s = scale

    def noisy(color, amp):
        t = np.clip(np.array(color) + rng.integers(-amp, amp + 1, (s, s, 1)),
                    0, 255).astype(np.uint8)
        return np.broadcast_to(t, (s, s, 3)).copy()

    # floor: dark with subtle grain
    floor = noisy((28, 32, 38), 4)

    # wall: silver metal plate with rivets in the corners
    wall = noisy((122, 122, 130), 8)
    wall[0, :] = (88, 88, 96)          # top shading
    wall[-1, :] = (146, 146, 154)      # bottom highlight
    rv = max(1, s // 5)
    for py in (max(1, s // 4), min(s - 2, 3 * s // 4)):
        for px in (max(1, s // 4), min(s - 2, 3 * s // 4)):
            wall[py:py + rv, px:px + rv] = (70, 70, 78)

    # crate: brown planks with horizontal seams
    crate = noisy((150, 100, 60), 10)
    plank = max(1, s // 3)
    for y0 in range(0, s, plank):
        if (y0 // plank) % 2:
            crate[y0:y0 + plank] = (crate[y0:y0 + plank] * 0.8).astype(np.uint8)
        crate[y0, :] = (96, 62, 36)    # seam line

    # bush: green base with darker and lighter speckles
    bush = noisy((36, 96, 44), 10)
    speck = rng.random((s, s))
    bush[speck < 0.18] = (22, 62, 28)
    bush[speck > 0.88] = (66, 138, 74)

    # fence: dark-brown vertical slats with gaps
    fence = np.empty((s, s, 3), dtype=np.uint8)
    fence[:] = (44, 28, 18)            # gap colour
    slat = max(2, s // 3)
    for x0 in range(0, s, slat * 2):
        fence[:, x0:x0 + slat] = noisy((88, 56, 34), 6)[:, x0:x0 + slat]
        fence[:, x0] = (60, 38, 22)    # slat edge

    # goal: bright horizontal bands
    goal = np.empty((s, s, 3), dtype=np.uint8)
    band = max(1, s // 4)
    for y0 in range(0, s, band):
        goal[y0:y0 + band] = (232, 200, 88) if (y0 // band) % 2 == 0 \
            else (52, 46, 76)

    _TEX_CACHE[scale] = {TILE_EMPTY: floor, TILE_WALL: wall, TILE_CRATE: crate,
                         TILE_BUSH: bush, TILE_FENCE: fence, TILE_GOAL: goal}
    return _TEX_CACHE[scale]


def _paint_tiles(tiles: np.ndarray, scale: int) -> np.ndarray:
    """Tile the map grid into an RGB image using the cached textures."""
    h, w = tiles.shape
    tex = tile_textures(scale)
    img = np.tile(tex[TILE_EMPTY], (h, w, 1))
    for tid in (TILE_WALL, TILE_BUSH, TILE_CRATE, TILE_GOAL, TILE_FENCE):
        ys, xs = np.nonzero(tiles == tid)
        for y, x in zip(ys, xs):
            img[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = tex[tid]
    return img


def render_map_preview(tiles: np.ndarray, scale: int = 12) -> np.ndarray:
    """Terrain-only RGB image of a tile grid (used by the map editor)."""
    return _paint_tiles(tiles, scale)


def _draw_disc(img: np.ndarray, cx: float, cy: float, r: float, color, scale: int):
    h, w = img.shape[:2]
    px, py = cx * scale, cy * scale
    pr = max(1, int(r * scale))
    x0, x1 = max(0, int(px - pr)), min(w, int(px + pr) + 1)
    y0, y1 = max(0, int(py - pr)), min(h, int(py + pr) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = (xs - px) ** 2 + (ys - py) ** 2 <= pr ** 2
    img[y0:y1, x0:x1][mask] = color


def _draw_marker(img: np.ndarray, shape: str, cx: float, cy: float,
                 color, scale: int):
    h, w = img.shape[:2]
    px, py = int(cx * scale), int(cy * scale)
    r = max(2, int(0.16 * scale))
    if shape == "circle":
        _draw_disc(img, cx, cy, 0.16, color, scale)
        return
    x0, x1 = max(0, px - r), min(w, px + r + 1)
    y0, y1 = max(0, py - r), min(h, py + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    if shape == "square":
        img[y0:y1, x0:x1] = color
    elif shape == "cross":
        t = max(1, r // 2)
        img[max(0, py - t):min(h, py + t + 1), x0:x1] = color
        img[y0:y1, max(0, px - t):min(w, px + t + 1)] = color
    elif shape == "triangle":  # thin upward triangle
        for i in range(y1 - y0):
            half = max(1, int((i + 1) * r / (y1 - y0)))
            img[y0 + i, max(0, px - half):min(w, px + half + 1)] = color


def render(game: "Game", viewer_team: int = 0) -> np.ndarray:
    cfg = game.cfg
    s = cfg.render_scale
    h, w = game.tiles.shape
    img = _paint_tiles(game.tiles, s)

    # showdown poison ring: darken everything outside the safe zone
    if game.mode == "showdown":
        ys, xs = np.mgrid[0:h * s, 0:w * s]
        cx, cy = cfg.map_w / 2.0 * s, cfg.map_h / 2.0 * s
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / s
        outside = dist > game.poison_radius
        img[outside] = (img[outside] * 0.4 +
                        np.array(COLORS["poison"]) * 0.6).astype(np.uint8)
        edge = np.abs(dist - game.poison_radius) < 0.12
        img[edge] = COLORS["poison_edge"]

    for g in game.gems:
        _draw_disc(img, g[0], g[1], 0.25, COLORS["gem"], s)

    if game.mode == "brawl_ball" and game.ball_carrier is None:
        _draw_disc(img, game.ball_pos[0], game.ball_pos[1], 0.3,
                   COLORS["ball_edge"], s)
        _draw_disc(img, game.ball_pos[0], game.ball_pos[1], 0.22,
                   COLORS["ball"], s)

    for p in game.projectiles:
        c = COLORS["proj0"] if p.team == 0 else COLORS["proj1"]
        if p.lobbed and p.range_total > 0:
            # fake ballistic arc: parabolic height 0->1->0 over the flight,
            # shadow stays at ground pos, disc rises (y-offset), grows and
            # brightens with height so lobs read as "flying over things"
            t = 1.0 - max(0.0, min(1.0, p.range_left / p.range_total))
            h = 4.0 * t * (1.0 - t)
            _draw_disc(img, p.pos[0], p.pos[1], max(p.radius, 0.2) * 0.9,
                       COLORS["shadow"], s)
            lift = h * 0.9                      # peak ~0.9 tile above ground
            r = max(p.radius, 0.2) * (1.0 + 0.7 * h)
            c_hi = tuple(min(255, int(v * (1.0 + 0.5 * h))) for v in c)
            _draw_disc(img, p.pos[0], p.pos[1] - lift, r, c_hi, s)
        else:
            _draw_disc(img, p.pos[0], p.pos[1], max(p.radius, 0.2), c, s)

    for i, u in enumerate(game.units):
        if not u.alive:
            continue
        if u.team != viewer_team and not game.is_visible_to(u, viewer_team):
            continue   # stealthed in a bush: not drawn
        radius = u.look.get("radius", cfg.unit_radius)
        body = PALETTE[int(u.look.get("color", 0.0) * len(PALETTE)) % len(PALETTE)]
        ring = TEAM_RINGS[u.team % len(TEAM_RINGS)]
        _draw_disc(img, u.pos[0], u.pos[1], radius + 0.1, ring, s)
        _draw_disc(img, u.pos[0], u.pos[1], radius, body, s)
        # class marker above the head
        _draw_marker(img, MARKERS.get(u.archetype, "circle"),
                     u.pos[0], u.pos[1] - radius - 0.32, COLORS["marker"], s)
        # hp bar above the unit
        bar_w = int(radius * 2 * s)
        frac = max(0.0, u.hp / ARCHETYPES[u.archetype]["hp"])
        cx, cy = int(u.pos[0] * s), int(u.pos[1] * s)
        y0 = max(0, cy - int(radius * s) - 3)
        x0 = max(0, cx - bar_w // 2)
        img[y0:y0 + 2, x0:x0 + bar_w] = COLORS["hp_bg"]
        img[y0:y0 + 2, x0:x0 + int(bar_w * frac)] = COLORS["hp_fg"]
        # gem count pips under the unit
        for k in range(min(u.gems, 10)):
            gx = x0 + k * 3
            if 0 <= y0 + 4 < img.shape[0] and 0 <= gx < img.shape[1]:
                img[y0 + 3:y0 + 5, gx:gx + 2] = COLORS["gem"]
        # yellow ring when super is ready
        if u.super_charge >= 1.0:
            _draw_disc(img, u.pos[0], u.pos[1], radius + 0.22,
                       COLORS["super_ready"], s)
            _draw_disc(img, u.pos[0], u.pos[1], radius + 0.1, ring, s)
            _draw_disc(img, u.pos[0], u.pos[1], radius, body, s)

    # carried ball rides on top of its carrier
    if game.mode == "brawl_ball" and game.ball_carrier is not None:
        _draw_disc(img, game.ball_pos[0], game.ball_pos[1], 0.24,
                   COLORS["ball"], s)

    return img
