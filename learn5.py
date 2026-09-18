"""Retrain at tonic 0.10, with and without the olfactory channel.

At the 0.14 operating point the antennal lobe sits pinned near its 30 Hz
ceiling and the descending population cannot tell left-smell from right-smell
at all (AUC 0.494, chance). Lowering tonic frees it — 0.567 at 0.10, 0.671 at
0.07 — but also quietens everything else, and at 0.07 the descending
population is silent (0.1 Hz) so there is nothing left to read. 0.10 is the
only point where both might work: smell partly legible, descending still
firing at ~0.8 Hz.

Two readouts are fitted from two separate collections at that tonic, one with
the olfactory channel and one without, because changing the operating point
and adding an input at the same time is exactly the mistake v4 made.

A new target comes with it: `rear`, whether the nearest zombie is behind the
fly. Vision cannot answer that — the encoder's bearings stop at 167° and the
whole point of smell is that it has no blind spot.

    python learn5.py 45 olf     ->  readout_v14.npz
    python learn5.py 45 noolf   ->  readout_v14n.npz
"""
from __future__ import annotations

import math
import sys
import time
import numpy as np

from game import Game, DT
from pilots import FlyPilot
from learn import TARGETS as BASE_TARGETS, WARM, labels, MixedDriver, Basis, apply_model
from learn import TRAIN_SEEDS, TEST_SEEDS
from learn3 import ang_err

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
MODE = sys.argv[2] if len(sys.argv) > 2 else "olf"
OUT = "readout_v14.npz" if MODE == "olf" else "readout_v14n.npz"
TONIC = 0.10
TARGETS = BASE_TARGETS + ["rear"]
EXPLORE = 0.20


def all_labels(obs):
    y = labels(obs)
    zs = obs["zombies"]
    z = min(zs, key=lambda q: q["dist"]) if zs else None
    y["rear"] = 1.0 if (z is not None and abs(z["rel"]) > math.pi / 2) else 0.0
    return y


def collect(fly, seeds, secs, policy, start_wave=0):
    X, Y = [], {k: [] for k in TARGETS}
    deaths = 0
    for sd in seeds:
        g = Game(n_players=1, seed=sd, start_wave=start_wave)
        drv = MixedDriver(seed=sd)
        rng = np.random.default_rng(sd + 7 * start_wave)
        fly.v[:] = 0.0
        fly.fired = np.zeros(0, np.int64)
        fly.dn_trace[:] = 0.0
        fly.hist.clear()
        fly.steer_ema = 0.0
        fly.fleeing = False
        ex_left, ex_turn = 0, 0.0
        for i in range(int(secs / DT)):
            obs = g.observe(0)
            if policy == "self":
                cmd = fly.act(obs)
            else:
                fly.brain_step(fly.encode(obs))
                cmd = drv.act(obs)
            if i >= WARM and obs["zombies"]:
                X.append(fly.dn_trace.copy())
                for k, v in all_labels(obs).items():
                    Y[k].append(v)
            if policy == "self":
                if ex_left <= 0 and rng.random() < EXPLORE * DT:
                    ex_left, ex_turn = int(1.0 / DT), float(rng.uniform(-5, 5))
                if ex_left > 0:
                    ex_left -= 1
                    cmd = dict(cmd, turn=ex_turn)
            p = g.players[0]
            p.cmd_move, p.cmd_turn = cmd["move"], cmd["turn"]
            p.cmd_fire, p.cmd_swap = cmd["fire"], cmd["swap"]
            g.step()
            if g.over:
                deaths += 1
                g = Game(n_players=1, seed=sd + 1000 + i, start_wave=start_wave)
    print(f"    {len(X):6d} frames, {deaths} deaths", flush=True)
    f32 = lambda v: np.array(v, np.float32)
    return f32(X), {k: f32(v) for k, v in Y.items()}


def auc(score, label):
    o = np.argsort(score)
    lab = label[o]
    n1 = lab.sum()
    n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((np.arange(len(lab))[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    use_olf = MODE == "olf"
    print(f"tonic {TONIC}, olfaction {'ON' if use_olf else 'OFF'}")
    fly = FlyPilot(tonic20=TONIC, olfaction=use_olf, encoder="retino",
                   barrel_gate="tight", escape_mode="hand", move_mode="kite",
                   readout=None, quiet=False)
    t0 = time.time()

    print("  test:")
    pt = [collect(fly, TEST_SEEDS, SECS, "mixed", 0),
          collect(fly, TEST_SEEDS, SECS, "mixed", 6)]
    Xte = np.concatenate([p[0] for p in pt])
    Yte = {k: np.concatenate([p[1][k] for p in pt]) for k in TARGETS}

    print("  train:")
    pr = [collect(fly, TRAIN_SEEDS, SECS, "mixed", 0),
          collect(fly, TRAIN_SEEDS, SECS, "mixed", 5),
          collect(fly, TRAIN_SEEDS, SECS, "mixed", 7)]
    X = np.concatenate([p[0] for p in pr])
    Y = {k: np.concatenate([p[1][k] for p in pr]) for k in TARGETS}
    print(f"  {len(X)} train / {len(Xte)} test frames  "
          f"(descending trace mean {X.mean():.3f}, nonzero "
          f"{100*np.mean(X > 0):.1f}%)")

    B = Basis(X)
    cut = int(0.8 * len(X))
    Bi = Basis(X[:cut])
    best = None
    for rank in (50, 120, 300):
        for lam in (1.0, 10.0, 100.0, 1000.0):
            ms = Bi.fit(Y["aim_sin"][:cut], rank, lam)
            mc = Bi.fit(Y["aim_cos"][:cut], rank, lam)
            e = ang_err(apply_model(ms, X[cut:]), apply_model(mc, X[cut:]),
                        Y["aim_sin"][cut:], Y["aim_cos"][cut:])
            if best is None or e.mean() < best[0]:
                best = (e.mean(), rank, lam)
    _, rank, lam = best
    m = {k: B.fit(Y[k], rank, lam) for k in TARGETS}

    aim = np.arctan2(Yte["aim_sin"], Yte["aim_cos"])
    pa = np.arctan2(apply_model(m["aim_sin"], Xte), apply_model(m["aim_cos"], Xte))
    msk = np.abs(aim) > 0.25
    sgn = np.sign(pa[msk])
    side_ok = np.mean(np.where(sgn == 0, 0.5, (sgn == np.sign(aim[msk])).astype(float)))
    print(f"\n=== tonic {TONIC}, olfaction {'ON' if use_olf else 'OFF'}  "
          f"rank {rank}, L2 {lam:g} ===")
    print(f"  aim side correct   {100*side_ok:5.1f}%")
    print(f"  aim mean error     {np.degrees(np.abs((pa-aim+np.pi)%(2*np.pi)-np.pi)[msk].mean()):5.1f}°")
    print(f"  flee AUC           {auc(apply_model(m['flee'], Xte), Yte['flee']):.3f}")
    print(f"  barrel AUC         {auc(apply_model(m['blocked'], Xte), Yte['blocked']):.3f}")
    print(f"  REAR AUC           {auc(apply_model(m['rear'], Xte), Yte['rear']):.3f}"
          f"   (positives {100*Yte['rear'].mean():.1f}%)")

    d = {}
    for k in BASE_TARGETS:
        d[k + "_mu"], d[k + "_P"] = m[k]["mu"], m[k]["P"]
        d[k + "_w"], d[k + "_b"] = m[k]["w"], m[k]["b"]
    d["rank"], d["lam"] = rank, lam
    np.savez(OUT, **d)
    print(f"\n({time.time()-t0:.0f}s) wrote {OUT}")
