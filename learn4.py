"""Does the aim readout have the same training-set problem as escape did?

v10 showed that every readout in this project was fitted on data that is ~88%
situations where escaping does not matter, and rebalancing onto the frames that
decide runs really did move the offline number.

Aim might be the same story — or the exact opposite, and that is worth saying
before the fit rather than after.

Escape and aim are limited by different frames:

  escape  the frames that matter are the rare ones (surrounded). Any direction
          works when nothing is near, so the common frames teach nothing.
  aim     when the target is far off-axis, being roughly right is enough — the
          fly just turns. When the target is nearly centred, precision is the
          whole game, because that is the frame where a bullet lands or misses.

So upweighting off-axis frames, which is what "balance the bearings" means,
could be exactly the wrong move for aim. This script fits both and compares
them on the metric that decides kills — angular error while the target is
already near the centre — before anything goes near the game.

    python learn4.py 45 v7
"""
from __future__ import annotations

import sys
import time
import numpy as np

import versions
from pilots import FlyPilot
from learn import TARGETS, TRAIN_SEEDS, TEST_SEEDS, Basis, apply_model
from learn2 import configure
from learn3 import collect, ang_err

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
DRIVE = sys.argv[2] if len(sys.argv) > 2 else "v7"
BANDS = [(0.0, 0.15), (0.15, 0.40), (0.40, 0.90), (0.90, np.pi)]


def bearing(Y):
    return np.arctan2(Y["aim_sin"], Y["aim_cos"])


def resample(X, Y, weights_of, rng, tag):
    """Draw a fit set of the same size with the given per-frame weight."""
    w = weights_of(bearing(Y))
    w = w / w.sum()
    idx = rng.choice(len(X), size=len(X), replace=True, p=w)
    ab = np.abs(bearing(Y)[idx])
    print(f"  {tag:18s} centred<0.15 rad {100*np.mean(ab < 0.15):5.1f}%   "
          f"off-axis>0.9 {100*np.mean(ab > 0.9):5.1f}%")
    return X[idx], {k: Y[k][idx] for k in TARGETS}


def fit_aim(Xf, Yf, Xva, Yva, select_mask):
    """Rank and L2 picked on `select_mask` frames of the validation split."""
    B = Basis(Xf)
    best = None
    for rank in (50, 120, 300, 600):
        for lam in (1.0, 10.0, 100.0, 1000.0):
            ms = B.fit(Yf["aim_sin"], rank, lam)
            mc = B.fit(Yf["aim_cos"], rank, lam)
            e = ang_err(apply_model(ms, Xva), apply_model(mc, Xva),
                        Yva["aim_sin"], Yva["aim_cos"])
            sc = e[select_mask].mean()
            if best is None or sc < best[0]:
                best = (sc, rank, lam, ms, mc)
    return best


def band_report(name, ms, mc, Xte, Yte):
    e = ang_err(apply_model(ms, Xte), apply_model(mc, Xte),
                Yte["aim_sin"], Yte["aim_cos"])
    ab = np.abs(bearing(Yte))
    row = f"{name:22s}"
    for lo, hi in BANDS:
        m = (ab >= lo) & (ab < hi)
        row += f"{e[m].mean():10.1f}°" if m.any() else f"{'-':>11s}"
    side = np.sign(np.arctan2(apply_model(ms, Xte), apply_model(mc, Xte)))
    t = np.sign(bearing(Yte))
    inf = ab > 0.25
    sc = np.mean(np.where(side[inf] == 0, 0.5, (side[inf] == t[inf]).astype(float)))
    print(row + f"{100*sc:11.1f}%")
    return e


if __name__ == "__main__":
    v = versions.get(DRIVE)
    fly = FlyPilot(quiet=False)
    configure(fly, v)
    rng = np.random.default_rng(0)
    t0 = time.time()

    print("\ncollecting TEST...")
    parts = [collect(fly, TEST_SEEDS, SECS, "mixed", 0),
             collect(fly, TEST_SEEDS, SECS, "mixed", 6)]
    Xte = np.concatenate([p[0] for p in parts])
    Yte = {k: np.concatenate([p[1][k] for p in parts]) for k in TARGETS}

    print("collecting TRAIN...")
    parts = []
    for pol, sw in (("mixed", 0), ("self", 0), ("self", 5), ("mixed", 7)):
        parts.append(collect(fly, TRAIN_SEEDS, SECS, pol, sw))
    X = np.concatenate([p[0] for p in parts])
    Y = {k: np.concatenate([p[1][k] for p in parts]) for k in TARGETS}
    cut = int(0.8 * len(X))
    Xva, Yva = X[cut:], {k: Y[k][cut:] for k in TARGETS}
    Xtr, Ytr = X[:cut], {k: Y[k][:cut] for k in TARGETS}
    ab_va = np.abs(bearing(Yva))
    print(f"  {len(X)} frames; natural distribution: "
          f"centred<0.15 rad {100*np.mean(np.abs(bearing(Y)) < 0.15):.1f}%, "
          f"off-axis>0.9 {100*np.mean(np.abs(bearing(Y)) > 0.9):.1f}%")

    print("\nresampled fit sets:")
    # flat over |bearing|: every band equally represented
    edges = np.linspace(0, np.pi, 9)
    def flat_w(b):
        idx = np.clip(np.digitize(np.abs(b), edges) - 1, 0, len(edges) - 2)
        cnt = np.bincount(idx, minlength=len(edges) - 1).astype(float)
        cnt[cnt == 0] = 1.0
        return 1.0 / cnt[idx]
    Xa, Ya = resample(Xtr, Ytr, flat_w, rng, "bearing-uniform")
    # the opposite: weight toward the centre, where a shot lands or misses
    Xb, Yb = resample(Xtr, Ytr, lambda b: np.exp(-(b / 0.35) ** 2) + 0.15, rng,
                      "centre-weighted")

    old = np.load(v["readout"])
    om = {k: {"mu": old[k + "_mu"], "P": old[k + "_P"], "w": old[k + "_w"],
              "b": float(old[k + "_b"])} for k in ("aim_sin", "aim_cos")}

    fits = {}
    for tag, (Xf, Yf) in (("bearing-uniform", (Xa, Ya)),
                          ("centre-weighted", (Xb, Yb)),
                          ("as collected", (Xtr, Ytr))):
        sel = ab_va < 0.15 if tag == "centre-weighted" else ab_va > 0.25
        sc, rank, lam, ms, mc = fit_aim(Xf, Yf, Xva, Yva, sel)
        fits[tag] = (ms, mc, rank, lam)
        print(f"  {tag:18s} rank {rank:3d}  L2 {lam:g}")

    print(f"\n=== aim error by how far off-axis the target is "
          f"({len(Xte)} held-out frames) ===")
    hdr = "".join(f"{f'{lo:.2f}-{hi:.2f}':>11s}" for lo, hi in BANDS)
    print(f"{'':22s}{hdr}{'side ok':>12s}")
    band_report("shipped (v7)", om["aim_sin"], om["aim_cos"], Xte, Yte)
    for tag, (ms, mc, _, _) in fits.items():
        band_report(tag, ms, mc, Xte, Yte)

    print("\nthe band that decides kills is the first one: a shot only lands "
          "when the target is already near the centre.")

    for tag, out in (("bearing-uniform", "readout_v11.npz"),
                     ("centre-weighted", "readout_v12.npz")):
        ms, mc, rank, lam = fits[tag]
        d = {}
        for k in TARGETS:
            src = ms if k == "aim_sin" else mc if k == "aim_cos" else None
            if src is None:
                for suf in ("_mu", "_P", "_w", "_b"):
                    d[k + suf] = old[k + suf]
            else:
                d[k + "_mu"], d[k + "_P"] = src["mu"], src["P"]
                d[k + "_w"], d[k + "_b"] = src["w"], src["b"]
        d["rank"], d["lam"] = rank, lam
        np.savez(out, **d)
        print(f"wrote {out} ({tag}; everything except aim copied from {v['readout']})")
    print(f"({time.time()-t0:.0f}s)")
