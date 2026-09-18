"""Step 0: settle the stage before any content goes on it.

Arena size moves the standoff optimum and the difficulty at once, so every
survival number is only comparable within one size. Pick the size first, with
the cheapest pilot available, and never move it again.

What a good stage needs:
  * the ceiling must not hit the safety cap, or survival is censored
  * there must be headroom between the fly and the ceiling, or improvements
    are invisible
  * runs must be short enough to measure a dozen versions against

Timed waves throughout, because the clear-the-board rule lets a bad pilot stall
the difficulty and makes survival meaningless.
"""
import json, numpy as np
import game
from pilots import KiterPilot
from survive import run_until_death

SEEDS = [7, 11, 23, 41, 57, 63, 88, 101]
SIZES = [(720, 480), (960, 640), (1280, 854), (1600, 1067)]
REACH = [100, 130, 200, 300]
CAP = 900.0

out = {}
print(f"{'arena':>12s}{'area x':>8s}{'reach':>7s}{'survived':>16s}{'wave':>7s}"
      f"{'kills':>8s}{'hit cap':>9s}")
base = 960 * 640
for w, h in SIZES:
    game.set_arena(w, h)
    for reach in REACH:
        rows = [run_until_death(KiterPilot(reach=reach), sd, 7, CAP, "timed")
                for sd in SEEDS]
        g = lambda k: np.array([r[k] for r in rows], float)
        s = g("survived")
        print(f"{w}x{h:<6d}{w*h/base:8.2f}{reach:7d}{s.mean():9.1f} ± {s.std():4.1f}"
              f"{g('wave').mean():7.1f}{g('kills').mean():8.1f}"
              f"{sum(r['capped'] for r in rows):6d}/{len(rows)}", flush=True)
        out[f"{w}x{h}@{reach}"] = rows
game.set_arena(960, 640)
json.dump({"mode": "arena x standoff sweep", "waves": "timed", "cap_s": CAP,
           "seeds": SEEDS, "runs": out}, open("results/arena_sweep.json", "w"), indent=1)
print("\nwrote results/arena_sweep.json")
