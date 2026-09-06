"""Original hand-designed maps for brawl-arena.

Each map is written as the top-left QUADRANT (top-half rows, left-half
columns including the centre column/row) and mirrored across both axes at
load time, which guarantees top-bottom and left-right symmetry.

Design rules followed (original layouts, public genre design guidelines):
  * 3v3 maps use a three-lane layout: two side lanes + an open contested mid
  * gem_grab: mine ringed by cover but flankable; centre 3x3 fully clear,
    no hard cover within 2 of the mine (bush allowed); no large contiguous
    central bush
  * brawl_ball: no long horizontal wall sealing the goal; U-walls / pillars
    only; mid is direct but dangerous, flanks offer detour shot angles
  * showdown: mixed open/cover/bush centre; crate hotspots equidistant from
    the spawn points
  * every corridor is at least 2 tiles wide, no diagonal-only wall seams,
    no enclosed regions (all verified by _validate() at import time)

Characters:
    '.'  empty floor
    '#'  indestructible wall
    'b'  bush (stealth)
    'c'  destructible crate
    'g'  brawl_ball goal opening (border middle 3 tiles, both sides)
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np

from .core import (TILE_BUSH, TILE_CRATE, TILE_EMPTY, TILE_FENCE, TILE_GOAL,
                   TILE_WALL)

_CHAR_TO_TILE = {".": TILE_EMPTY, "#": TILE_WALL, "b": TILE_BUSH,
                 "c": TILE_CRATE, "g": TILE_GOAL, "f": TILE_FENCE}

_MODE_SIZE = {"gem_grab": (21, 15), "brawl_ball": (21, 15),
              "knockout": (17, 13), "showdown": (25, 19), "duel": (17, 13)}


def _expand(quadrant: list[str]) -> np.ndarray:
    """Mirror a top-left quadrant across the vertical and horizontal axes."""
    rows = [list(q + q[-2::-1]) for q in quadrant]
    rows = rows + rows[-2::-1]
    return np.array([[_CHAR_TO_TILE[c] for c in r] for r in rows], dtype=np.int8)


# ---------------------------------------------------------- gem_grab (21x15)
# open_field: three wide-open lanes, one pillar pair anchors the top/bottom
# lane, small bush patches ring the mine top/bottom; sides fully open.
_GEM_OPEN = [
    "###########",
    "#..........",
    "#..........",
    "#....##....",
    "#......bb..",
    "#..##......",
    "#..........",
    "#..........",
]

# ambush_bush: side lanes are heavy bush for ambushes; the mine is ringed
# left/right by bush columns at distance 3; the mid lane stays open.
_GEM_BUSH = [
    "###########",
    "#..bbb..bb.",
    "#..bbb..bb.",
    "#..........",
    "#......b...",
    "#......b...",
    "#......b...",
    "#......b...",
]

# bunker: 2x2 crate blocks guard the side lanes, wall blocks extend the
# cover downward; a single crate rings the mine top/bottom at distance 3.
_GEM_BUNKER = [
    "###########",
    "#..b....b..",
    "#..........",
    "#..cc..cc..",
    "#..cc.....c",
    "#..##......",
    "#..........",
    "#..........",
]

# flank_maze: a full-width wall band splits off top/bottom detour lanes
# with 2-wide connectors; pillars fused to the band shape the mid lane.
_GEM_FLANK = [
    "###########",
    "#..........",
    "#..........",
    "#..###..###",
    "#..##..bb..",
    "#..........",
    "#..........",
    "#..........",
]

# --------------------------------------------------------- brawl_ball (21x15)
# open_mid: direct open mid lane, small L-shaped goal guards (no long
# horizontal wall in front of the goal), flank pillars for angled shots.
_BALL_OPEN = [
    "###########",
    "#..........",
    "#..........",
    "#..#....#..",
    "#...##.....",
    "#...#......",
    "g..........",
    "g..........",
]

# bush_flanks: bushy side lanes offer safe detours and angled shots; mid is
# short and direct; single crate pillars guard each goal approach.
_BALL_BUSH = [
    "###########",
    "#.bbb..bbb.",
    "#.bbb..bbb.",
    "#..........",
    "#....b.b...",
    "#...c...c..",
    "g..........",
    "g..........",
]

# center_cover: staggered 2-wide wall blocks around the midfield force the
# ball carrier to pick an angle; the goal approach itself stays open.
_BALL_COVER = [
    "###########",
    "#..........",
    "#..b....b..",
    "#..........",
    "#..##..##..",
    "#....##....",
    "g..........",
    "g..........",
]

# ---------------------------------------------------------- knockout (17x13)
# duel_ring: open arena; single crate pillars plus one central crate pair
# are the only cover.
_KO_RING = [
    "#########",
    "#........",
    "#........",
    "#..c..c..",
    "#...cc...",
    "#........",
    "#........",
]

# quad_cover: a 2x2 wall block plus a staggered 2x1 wall pair per quadrant
# (well separated, no diagonal seams); corner bushes for ambushes.
_KO_COVER = [
    "#########",
    "#........",
    "#..b...b.",
    "#..##..b.",
    "#....##..",
    "#........",
    "#........",
]

# ---------------------------------------------------------- showdown (25x19)
# (built-in showdown map removed; showdown maps come from maps_custom/)

MAPS = {
    "gem_grab": {"open_field": _GEM_OPEN, "ambush_bush": _GEM_BUSH,
                 "bunker": _GEM_BUNKER, "flank_maze": _GEM_FLANK},
    "brawl_ball": {"open_mid": _BALL_OPEN, "bush_flanks": _BALL_BUSH,
                   "center_cover": _BALL_COVER},
    "knockout": {"duel_ring": _KO_RING, "quad_cover": _KO_COVER},
    "showdown": {},
    "duel": {},          # 1v1 falls back to the knockout arenas in sample_map
}


def sample_map(mode: str, rng: np.random.Generator):
    """Pick a random map for the mode (built-in or custom).

    Returns (tiles, spawns): tiles is the full int8 grid; spawns is a list
    of (x, y) tile coords from custom 's' markers, or None to use the
    mode's default spawn layout."""
    pool = [(None, _expand(q)) for q in MAPS[mode].values()]
    pool += [(e["spawns"], e["tiles"]) for e in CUSTOM_MAPS.get(mode, [])]
    if mode == "duel":
        # 1v1 plays on its own custom maps plus the knockout arenas
        pool += [(None, _expand(q)) for q in MAPS["knockout"].values()]
    spawns, tiles = pool[int(rng.integers(len(pool)))]
    return tiles.copy(), spawns


# ------------------------------------------------------------ custom maps
CUSTOM_MAPS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps_custom")

CUSTOM_MAPS: dict[str, list[dict]] = {}


def parse_ascii_map(text: str):
    """Parse a full-map ASCII file. Returns (tiles, spawns).

    's' marks a showdown spawn point (stored as floor); all other chars
    follow the editor palette."""
    rows = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    assert rows, "empty map"
    w = len(rows[0])
    bad = [(i, len(r)) for i, r in enumerate(rows) if len(r) != w]
    assert not bad, f"ragged rows: row(s) {bad}, expected width {w}"
    spawns = []
    tiles = np.zeros((len(rows), w), dtype=np.int8)
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c == "s":
                spawns.append((x, y))
                tiles[y, x] = TILE_EMPTY
            else:
                assert c in _CHAR_TO_TILE, f"unknown char {c!r}"
                tiles[y, x] = _CHAR_TO_TILE[c]
    return tiles, spawns


def load_custom_maps(directory: str | None = None) -> dict[str, list[dict]]:
    """Scan maps_custom/ for '<mode>_<name>.txt' files; valid ones join the
    sampling pool, invalid ones are skipped with a warning."""
    found: dict[str, list[dict]] = {}
    d = directory or CUSTOM_MAPS_DIR
    if not os.path.isdir(d):
        return found
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".txt"):
            continue
        stem = fn[:-4]
        mode = next((m for m in _MODE_SIZE if stem.startswith(m + "_")), None)
        if mode is None:
            print(f"[maps] skipping {fn}: unknown mode prefix")
            continue
        name = stem
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                tiles, spawns = parse_ascii_map(f.read())
            _check_map(mode, name, tiles, spawns=spawns or None)
            found.setdefault(mode, []).append(
                {"name": name, "tiles": tiles, "spawns": spawns or None})
        except Exception as e:
            print(f"[maps] skipping {fn}: {e}")
    return found


# -------------------------------------------------------------- validation
def _components(mask: np.ndarray) -> list[int]:
    """Sizes of the 4-connected components of `mask`."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    sizes = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            n = 0
            q = deque([(sx, sy)])
            seen[sy, sx] = True
            while q:
                x, y = q.popleft()
                n += 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] \
                            and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((nx, ny))
            sizes.append(n)
    return sizes


def _check_map(mode: str, name: str, tiles: np.ndarray,
               spawns: list | None = None):
    tag = f"{mode}/{name}"
    h, w = tiles.shape
    ew, eh = _MODE_SIZE[mode]
    assert (w, h) == (ew, eh), f"{tag}: bad size {(w, h)}, expected {(ew, eh)}"

    # fairness: require either double-mirror symmetry or 180-degree
    # (centre) rotational symmetry
    double_mirror = (tiles == tiles[::-1]).all() and (tiles == tiles[:, ::-1]).all()
    center_sym = (tiles == tiles[::-1, ::-1]).all()
    assert double_mirror or center_sym, \
        f"{tag}: must be double-mirror symmetric or 180-degree center symmetric"

    # unit-passable = floor or bush (walls, crates, goal zones, fences block)
    free = (tiles == TILE_EMPTY) | (tiles == TILE_BUSH)

    # connectivity: a single region, no enclosed areas
    n_regions = len(_components(free))
    assert n_regions == 1, f"{tag}: {n_regions} disconnected regions"

    # corridor width: every passable tile must lie in a fully-passable 2x2
    # block (kills 1-wide gaps and diagonal-only seams)
    bl = free[:-1, :-1] & free[1:, :-1] & free[:-1, 1:] & free[1:, 1:]
    covered = np.zeros_like(free)
    covered[:-1, :-1] |= bl
    covered[1:, :-1] |= bl
    covered[:-1, 1:] |= bl
    covered[1:, 1:] |= bl
    bad = np.argwhere(free & ~covered)
    assert len(bad) == 0, f"{tag}: narrow passage at {bad[:5].tolist()}"

    # no oversized single bush field
    biggest_bush = max(_components(tiles == TILE_BUSH), default=0)
    assert biggest_bush <= 16, f"{tag}: bush blob of {biggest_bush} tiles"

    cy, cx = h // 2, w // 2
    if mode == "gem_grab":
        # our maps are smaller than the genre reference, so the mine rule is
        # scaled down: the centre 3x3 must be fully clear; the 5x5 ring may
        # hold bushes but no hard cover (walls/crates/fences)
        core = tiles[cy - 1:cy + 2, cx - 1:cx + 2]
        assert (core == TILE_EMPTY).all(), f"{tag}: obstacle within 1 of mine"
        ring = tiles[cy - 2:cy + 3, cx - 2:cx + 3]
        bad = (ring == TILE_WALL) | (ring == TILE_CRATE) | (ring == TILE_FENCE)
        assert not bad.any(), f"{tag}: hard cover within 2 of mine (bush ok)"
    elif mode == "brawl_ball":
        assert tiles[cy, cx] == TILE_EMPTY, f"{tag}: ball spawn blocked"
        for gx, inward in ((0, 1), (w - 1, w - 2)):
            goals = [y for y in range(h) if tiles[y, gx] == TILE_GOAL]
            assert goals == [cy - 1, cy, cy + 1], f"{tag}: bad goals at x={gx}"
            for y in goals:
                assert free[y, inward], f"{tag}: goal sealed at ({inward}, {y})"
    elif mode == "showdown":
        assert int((tiles == TILE_CRATE).sum()) >= 8, f"{tag}: too few crates"
        if spawns:
            assert len(spawns) >= 2, f"{tag}: too few spawn markers"
        else:
            spawns = [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3),
                      (w // 2, 2), (w // 2, h - 3)]
            for sx, sy in spawns:
                assert free[sy, sx], f"{tag}: spawn blocked at ({sx}, {sy})"
    if mode != "showdown":
        if spawns:
            # custom 3v3 spawns (brawl_ball / knockout): 3 per side, each
            # free, and every marker must have its 180-degree twin (a
            # double-mirror-symmetric set automatically satisfies this).
            # duel is 1v1: exactly 1 per side.
            n_per_side = 1 if mode == "duel" else 3
            left = [s for s in spawns if s[0] < w / 2]
            right = [s for s in spawns if s[0] > w / 2]
            assert len(left) == n_per_side and len(right) == n_per_side, \
                f"{tag}: need {n_per_side} spawn markers per side, " \
                f"got {len(left)}/{len(right)}"
            sset = set(spawns)
            for sx, sy in spawns:
                assert free[sy, sx], f"{tag}: spawn blocked at ({sx}, {sy})"
                assert (w - 1 - sx, h - 1 - sy) in sset, \
                    f"{tag}: spawn at ({sx}, {sy}) lacks 180-degree twin"
        else:
            for sx in (2, w - 3):
                for sy in range(2, h - 2):
                    assert free[sy, sx], f"{tag}: spawn column blocked at ({sx}, {sy})"


def _validate():
    for mode, maps in MAPS.items():
        for name, quadrant in maps.items():
            _check_map(mode, name, _expand(quadrant))


_validate()

# pick up user-designed maps from maps_custom/ (invalid ones are skipped)
CUSTOM_MAPS.update(load_custom_maps())
