"""Control group: run balance analysis on the built-in maps."""
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from balance_check import _beta, _run_map
from brawl_arena.maps import MAPS, _expand


def main():
    jobs = []
    for mode in ("gem_grab", "brawl_ball", "knockout", "showdown"):
        for name, quad in MAPS[mode].items():
            jobs.append((mode, f"builtin/{name}", _expand(quad), None, 200, 777))
    with mp.Pool(8) as pool:
        results = pool.map(_run_map, jobs)
    for mode, name, sw, sl, n, margins, arch in results:
        if mode != "showdown":
            m_, lo, hi = _beta(sw, sl)
            flag = "  <-- SIDE BIAS" if lo > 0.5 or hi < 0.5 else ""
            print(f"{name}: team0 win {m_:.3f} [{lo:.3f},{hi:.3f}]{flag}")
        else:
            parts = []
            for a, (w, g) in sorted(arch.items()):
                m_, lo, hi = _beta(w, g - w)
                flag = "!" if lo > 0.25 or hi < 0.11 else ""
                parts.append(f"{a}:{m_:.3f}{flag}")
            print(f"{name}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
