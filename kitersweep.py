"""Where is the ceiling really? Sweep the kiter's standoff radius.

420 px was a first guess and every comparison in the project has been made
against it. The connectome versions found that standing closer helps, so the
ceiling this was all measured against may simply have been set in the wrong
place. The kiter costs nothing to run, so this sweeps it properly.
"""
import json, numpy as np
from pilots import KiterPilot
from survive import run_until_death, SEEDS

print(f"{'reach':>7s}{'survived':>17s}{'median':>9s}{'worst':>8s}{'best':>8s}"
      f"{'wave':>7s}{'kills':>8s}{'acc':>8s}")
out = {}
for reach in (100, 130, 160, 200, 240, 280, 330, 420, 520):
    rows = [run_until_death(KiterPilot(reach=reach), sd, 7) for sd in SEEDS]
    g = lambda k: np.array([r[k] for r in rows], float)
    s = g("survived")
    print(f"{reach:7d}{s.mean():10.1f} ± {s.std():4.1f}{np.median(s):9.1f}"
          f"{s.min():8.1f}{s.max():8.1f}{g('wave').mean():7.1f}"
          f"{g('kills').mean():8.1f}{g('acc').mean():7.1f}%")
    out[reach] = rows
json.dump({"mode": "kiter standoff sweep", "seeds": SEEDS, "n_barrels": 7,
           "by_reach": out}, open("results/kiter_reach_sweep.json", "w"), indent=1)
print("\nwrote results/kiter_reach_sweep.json")
