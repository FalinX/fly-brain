"""Same 12 seeds as multibench: does the trained readout show up in the game?"""
import sys
import numpy as np
from bench import run
from pilots import FlyPilot

import versions
V = versions.get(sys.argv[1] if len(sys.argv) > 1 else "")
RO = V["id"]

SEEDS = [7, 11, 23, 41, 57, 63, 88, 101, 119, 144, 177, 203]
fly = FlyPilot(readout=V["readout"], encoder=V["encoder"],
               barrel_gate=V.get("barrel_gate", "none"),
               escape_mode=V.get("escape_mode", "hand"), label=V["id"], quiet=False)
rows = []
for sd in SEEDS:
    fly.v[:] = 0.0
    fly.fired = np.zeros(0, np.int64)
    fly.dn_trace[:] = 0.0
    fly.hist.clear()
    fly.steer_ema = 0.0
    fly.fleeing = False
    r = run(lambda: fly, 90.0, seed=sd)[0]
    rows.append(r)
    print(f"seed {sd:3d}  {r['survived']:5.1f}s  {r['kills']:3d}k  {r['acc']:5.1f}% acc  "
          f"barrel-dmg {r['dmg_barrel']:4.0f}  contact-dmg {r['dmg_contact']:4.0f}  "
          f"barrel-hits {r['barrel_shots']:3d}", flush=True)
g = lambda k: np.array([r[k] for r in rows], float)
print(f"\nfly-trained  survived {g('survived').mean():5.1f}±{g('survived').std():4.1f}  "
      f"kills {g('kills').mean():5.1f}±{g('kills').std():4.1f}  "
      f"acc {g('acc').mean():5.1f}±{g('acc').std():4.1f}  "
      f"wave {g('wave').mean():4.1f}  ms {g('ms_med').mean():5.2f}")
