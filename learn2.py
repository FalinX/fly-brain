"""DAgger round 2, aimed at the version that is actually shipping.

`learn.py` fitted round 1 on states produced by the round-0 readout, which was
still steering with the trained escape heading. v6 and v7 threw that heading
away and went back to the hand rule, so the states they visit are not the
states the current readout was fitted on. This closes that gap: let the
shipping version drive, label what it produces, aggregate with fresh
mixed-policy coverage, refit.

    python learn2.py 60 v7

Writes readout_v8.npz. Held out by SEED, never by frame.
"""
from __future__ import annotations

import sys
import time
import numpy as np

import versions
from pilots import FlyPilot
from learn import (TRAIN_SEEDS, TEST_SEEDS, TARGETS, collect, fit_all, save,
                   report, apply_model, side_score, auc)

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
DRIVE = sys.argv[2] if len(sys.argv) > 2 else "v7"
OUT = sys.argv[3] if len(sys.argv) > 3 else "readout_v8.npz"


def configure(fly, v):
    """Make this brain behave exactly like the named version."""
    fly.encoder = v["encoder"]
    fly.barrel_gate = v.get("barrel_gate", "none")
    fly.escape_mode = v.get("escape_mode", "hand")
    if v["readout"]:
        r = np.load(v["readout"])
        fly.ro = {k: r[k] for k in r.files}
    else:
        fly.ro = None
    fly.name = v["id"]


if __name__ == "__main__":
    v = versions.get(DRIVE)
    print(f"driving policy for the on-policy half: {v['id']} — {v['headline']}")
    fly = FlyPilot(quiet=False)
    t0 = time.time()

    configure(fly, v)
    print("\ncollecting TEST (mixed)...")
    Xte, Yte = collect(fly, TEST_SEEDS, SECS, "mixed")

    print("\ncoverage half: script + wander...")
    Xa, Ya = collect(fly, TRAIN_SEEDS, SECS, "mixed")

    print(f"\non-policy half: {v['id']} drives...")
    Xb, Yb = collect(fly, TRAIN_SEEDS, SECS, "self")

    X = np.concatenate([Xa, Xb])
    Y = {k: np.concatenate([Ya[k], Yb[k]]) for k in TARGETS}
    m = fit_all(X, Y, f"round 2 on {v['id']}")
    save(m, OUT)

    old = np.load(v["readout"])
    old_m = {k: {"mu": old[k + "_mu"], "P": old[k + "_P"],
                 "w": old[k + "_w"], "b": float(old[k + "_b"])} for k in TARGETS}

    print(f"\n=== held out, {len(Xte)} frames   ({time.time()-t0:.0f}s) ===")
    print(f"{'readout':22s}{'aim side':>12s}{'aim err':>10s}{'escape err':>12s}"
          f"{'flee AUC':>10s}{'barrel AUC':>11s}")
    report(old_m, Xte, Yte, f"shipped ({v['readout']})")
    report(m, Xte, Yte, "round 2")
    print(f"\ntrained on {len(X):,} frames ({len(Xa):,} mixed + {len(Xb):,} on-policy)")
    print(f"wrote {OUT}")
