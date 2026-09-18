"""Calibrate a new zombie mix so it is as hard as the one it replaces.

Adding kinds changed two things at once: the roster, and the overall
difficulty. A tank at 34 px/s cannot catch a player at 185, so it replaced
dangerous devils with harmless bullet sponges — every evasion-only pilot got
*better* (the blind control went 160.8 s to 257.8) while the pilot that tries
to clear the board got worse. That confounds the only question the new kinds
were added to answer.

The fix is to hold difficulty fixed and let the roster be the only change.
Difficulty here is defined by what the no-brain kiter scores, because it has no
readout to confound and costs nothing to run: the 2-kind world gives it
294.6 +/- 47.9 s, so a candidate mix is calibrated to land on that.

    python calibrate.py
"""
from __future__ import annotations

import json
import numpy as np

import game
from game import KINDS, Kind
from pilots import KiterPilot
from survive import run_until_death

SEEDS = [7, 11, 23, 41, 57, 63, 88, 101]
TARGET = 294.6          # what the 2-kind world gives the tuned kiter
CAP = 900.0


def score(label):
    rows = [run_until_death(KiterPilot(reach=130), sd, 7, CAP, "timed") for sd in SEEDS]
    s = np.array([r["survived"] for r in rows], float)
    k = np.array([r["kills"] for r in rows], float)
    print(f"  {label:34s}{s.mean():7.1f} ± {s.std():5.1f} s   kills {k.mean():6.1f}"
          f"   {'TOO EASY' if s.mean() > TARGET + 35 else 'too hard' if s.mean() < TARGET - 35 else 'in range'}")
    return s.mean(), rows


CANDIDATES = [
    ("as first written", dict(tank=(34.0, 0.22), runner=(135.0, 0.30))),
    ("tank keeps up (44), fewer", dict(tank=(44.0, 0.14), runner=(135.0, 0.34))),
    ("tank 52, fewer still", dict(tank=(52.0, 0.12), runner=(140.0, 0.36))),
    ("tank 58, runners common", dict(tank=(58.0, 0.12), runner=(145.0, 0.40))),
]

if __name__ == "__main__":
    print(f"target: the 2-kind world scores the kiter {TARGET} s\n")
    out = {}
    for label, cfg in CANDIDATES:
        for name, (speed, share) in cfg.items():
            k = KINDS[name]
            KINDS[name] = Kind(k.name, k.hp, k.hp_per_wave, speed, k.radius,
                               k.dps, k.score, k.from_wave, share)
        mean, rows = score(label)
        out[label] = {"config": {n: list(v) for n, v in cfg.items()},
                      "mean": round(mean, 1),
                      "runs": rows}
    json.dump({"target_s": TARGET, "seeds": SEEDS, "candidates": out},
              open("results/kind_calibration.json", "w"), indent=1)
    print("\nwrote results/kind_calibration.json")
