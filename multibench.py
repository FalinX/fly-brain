"""Five seeds, 90 s each: the fly against the three-rule script."""
import numpy as np
from bench import run
from pilots import FlyPilot, ScriptPilot

SEEDS = [7, 11, 23, 41, 57, 63, 88, 101, 119, 144, 177, 203]
SECS = 90.0

fly = FlyPilot(quiet=False)
res = {"script": [], "fly": []}
for sd in SEEDS:
    res["script"].append(run(ScriptPilot, SECS, seed=sd)[0])
    fly.v[:] = 0.0
    fly.fired = np.zeros(0, np.int64)
    fly.hist.clear()
    fly.steer_ema = 0.0
    fly.fleeing = False
    res["fly"].append(run(lambda: fly, SECS, seed=sd)[0])
    print(f"seed {sd:3d}  script {res['script'][-1]['survived']:5.1f}s "
          f"{res['script'][-1]['kills']:3d}k   "
          f"fly {res['fly'][-1]['survived']:5.1f}s {res['fly'][-1]['kills']:3d}k", flush=True)

print(f"\n{'':10s}{'survived s':>12s}{'kills':>10s}{'accuracy %':>13s}{'wave':>7s}{'ms/frame':>11s}")
for k, rows in res.items():
    surv = np.array([r["survived"] for r in rows])
    kill = np.array([r["kills"] for r in rows])
    acc = np.array([r["acc"] for r in rows])
    wav = np.array([r["wave"] for r in rows])
    ms = np.array([r["ms_med"] for r in rows])
    print(f"{k:10s}{surv.mean():7.1f}±{surv.std():4.1f}{kill.mean():7.1f}±{kill.std():4.1f}"
          f"{acc.mean():9.1f}±{acc.std():4.1f}{wav.mean():7.1f}{ms.mean():11.2f}")
print(f"\nfull 90 s survived — script {sum(r['survived']>=89.9 for r in res['script'])}/5   "
      f"fly {sum(r['survived']>=89.9 for r in res['fly'])}/5")
