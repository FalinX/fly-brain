"""Which descending channels actually carry the encoder's signal?

Runs the fly inside the game, logs what went in and what came out, and
correlates them. Anything near zero is noise and must not be wired to a key.
"""
from __future__ import annotations

import sys
import numpy as np

from game import Game, DT
from pilots import FlyPilot

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
WINDOW = int(sys.argv[2]) if len(sys.argv) > 2 else 9

fly = FlyPilot(window=WINDOW, quiet=False)
g = Game(n_players=1, seed=11)

log = {k: [] for k in ("loom_L", "loom_R", "threat_L", "threat_R",
                       "shot_L", "shot_R", "chase_L", "chase_R",
                       "forward", "steer_L", "steer_R", "backward",
                       "escape", "punch", "kick",
                       "rel", "dist", "n_close")}

frames = int(SECS / DT)
for i in range(frames):
    obs = g.observe(0)
    drive = fly.encode(obs)
    fly.brain_step(drive)
    for k, v in fly.see.items():
        log[k].append(v)
    log["forward"].append(0.5 * (fly.rate("forward_L") + fly.rate("forward_R")))
    log["steer_L"].append(fly.rate("steer_L"))
    log["steer_R"].append(fly.rate("steer_R"))
    log["backward"].append(0.5 * (fly.rate("backward_L") + fly.rate("backward_R")))
    log["escape"].append(0.5 * (fly.rate("escape_L") + fly.rate("escape_R")))
    log["punch"].append(0.5 * (fly.rate("punch_L") + fly.rate("punch_R")))
    log["kick"].append(0.5 * (fly.rate("kick_L") + fly.rate("kick_R")))
    zs = obs["zombies"]
    z = min(zs, key=lambda z: z["dist"]) if zs else None
    log["rel"].append(z["rel"] if z else 0.0)
    log["dist"].append(z["dist"] if z else 999.0)
    log["n_close"].append(sum(1 for q in zs if q["dist"] < 220))
    # keep it alive so the log covers real situations: drive it with the script
    p = g.players[0]
    if z:
        p.cmd_turn = max(-6.5, min(6.5, z["rel"] * 6.0))
        p.cmd_move = (-1.0, 0.0) if z["dist"] < 160 else (0.0, 0.0)
        p.cmd_fire = abs(z["rel"]) < 0.22
    g.step()
    if g.over:
        print("game over at", round(g.t, 1), "s — restarting")
        g = Game(n_players=1, seed=11 + i)

A = {k: np.array(v, float) for k, v in log.items()}
ins = ["loom_L", "loom_R", "threat_L", "threat_R", "shot_L", "shot_R", "chase_L", "chase_R"]
outs = ["forward", "steer_L", "steer_R", "backward", "escape", "punch", "kick"]

print(f"\nframes {frames}   window {WINDOW} ({WINDOW*DT*1000:.0f} ms)\n")
print("--- encoder drive actually seen (non-zero fraction / mean when on) ---")
for k in ins:
    a = A[k]
    on = a > 0
    print(f"{k:10s} on {100*on.mean():5.1f}%   mean-when-on {a[on].mean() if on.any() else 0:.3f}"
          f"   max {a.max():.3f}")

print("\n--- readout rates (Hz) ---")
print(f"{'group':10s}{'med':>7s}{'p90':>7s}{'p99':>7s}{'max':>7s}{'zero%':>8s}")
for k in outs:
    a = A[k]
    print(f"{k:10s}{np.median(a):7.1f}{np.percentile(a,90):7.1f}"
          f"{np.percentile(a,99):7.1f}{a.max():7.1f}{100*(a==0).mean():8.1f}")

print("\n--- correlation: encoder in  ->  descending out ---")
hdr = "".join(f"{o[:8]:>9s}" for o in outs)
print(f"{'':11s}{hdr}")
for k in ins:
    row = ""
    for o in outs:
        x, y = A[k], A[o]
        if x.std() < 1e-9 or y.std() < 1e-9:
            row += f"{'-':>9s}"
        else:
            row += f"{np.corrcoef(x, y)[0,1]:9.2f}"
    print(f"{k:11s}{row}")

print("\n--- does steering point at the target? ---")
d = A["steer_R"] - A["steer_L"]
m = np.abs(A["rel"]) > 0.05
if d[m].std() > 1e-9:
    print(f"corr(steer_R - steer_L, angle to nearest zombie) = "
          f"{np.corrcoef(d[m], A['rel'][m])[0,1]:+.3f}")
    left = A["rel"] < -0.15
    right = A["rel"] > 0.15
    print(f"target on the LEFT : steer_L {A['steer_L'][left].mean():5.2f} Hz  "
          f"steer_R {A['steer_R'][left].mean():5.2f} Hz")
    print(f"target on the RIGHT: steer_L {A['steer_L'][right].mean():5.2f} Hz  "
          f"steer_R {A['steer_R'][right].mean():5.2f} Hz")

print("\n--- does escape fire when something is actually rushing in? ---")
e = A["escape"]
hi = e > np.percentile(e, 90)
print(f"nearest-zombie distance when escape is in its top 10%: {A['dist'][hi].mean():6.1f} px")
print(f"                                          other frames: {A['dist'][~hi].mean():6.1f} px")
print(f"zombies within 220 px, escape top 10%: {A['n_close'][hi].mean():.2f}   "
      f"other frames: {A['n_close'][~hi].mean():.2f}")

np.savez("diag.npz", **A)
print("\nwrote diag.npz")
