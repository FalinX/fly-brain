"""One DAgger round.

Round 0 fits the readout on states produced by someone else — half script, half
wander. Once that readout starts steering, it visits states it never saw while
training, which is why the offline gain shrank in the game.

DAgger fixes exactly that: let the trained pilot drive, label those states with
the ground truth (the true bearing is available from the game whoever is
steering), add them to the pile, refit.

Round 1 keeps a fifth of its frames on a random turn so the new data still
covers bearings the trained pilot would otherwise never produce.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

from game import Game, DT
from pilots import FlyPilot
from train import (TRAIN_SEEDS, TEST_SEEDS, WARM, MixedDriver, Basis,
                   apply_model, r2, side_score, auc)

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 75.0
EXPLORE = 0.20            # fraction of frames driven by a random turn


def reset(fly):
    fly.v[:] = 0.0
    fly.fired = np.zeros(0, np.int64)
    fly.dn_trace[:] = 0.0
    fly.hist.clear()
    fly.steer_ema = 0.0
    fly.fleeing = False


def collect(fly, seeds, secs, policy):
    """policy: 'mixed' = script + wander, 'self' = the fly's current readout."""
    X, ang, flee, hand_steer, hand_esc = [], [], [], [], []
    deaths = 0
    for sd in seeds:
        g = Game(n_players=1, seed=sd)
        drv = MixedDriver(seed=sd)
        rng = np.random.default_rng(sd)
        reset(fly)
        explore_left, explore_turn = 0, 0.0
        for i in range(int(secs / DT)):
            obs = g.observe(0)
            if policy == "self":
                cmd = fly.act(obs)          # steps the brain itself
            else:
                fly.encode_last = fly.encode(obs)
                fly.brain_step(fly.encode_last)
                cmd = drv.act(obs)
            zs = obs["zombies"]
            if i >= WARM and zs:
                z = min(zs, key=lambda q: q["dist"])
                X.append(fly.dn_trace.copy())
                ang.append(z["rel"])
                flee.append(1.0 if z["dist"] < 170 else 0.0)
                hand_steer.append(fly.rate("steer_R") - fly.rate("steer_L"))
                hand_esc.append(0.5 * (fly.rate("escape_L") + fly.rate("escape_R")))
            if policy == "self":
                if explore_left <= 0 and rng.random() < EXPLORE * DT / 1.0:
                    explore_left = int(1.0 / DT)
                    explore_turn = float(rng.uniform(-5.0, 5.0))
                if explore_left > 0:
                    explore_left -= 1
                    cmd = dict(cmd, turn=explore_turn)
            p = g.players[0]
            p.cmd_move, p.cmd_turn = cmd["move"], cmd["turn"]
            p.cmd_fire, p.cmd_swap = cmd["fire"], cmd["swap"]
            g.step()
            if g.over:
                deaths += 1
                g = Game(n_players=1, seed=sd + 1000 + i)
                reset(fly)
        print(f"  seed {sd:3d}: {len(X):6d} frames", flush=True)
    f32 = lambda v: np.array(v, np.float32)
    print(f"  ({deaths} deaths during collection)")
    return f32(X), f32(ang), f32(flee), f32(hand_steer), f32(hand_esc)


def fit(X, A, F, tag):
    B = Basis(X)
    cut = int(0.75 * len(X))
    Bi = Basis(X[:cut])
    best = None
    for rank in (20, 50, 120, 300, 600):
        for lam in (1.0, 10.0, 100.0, 1000.0, 10000.0):
            ms = Bi.fit(np.sin(A[:cut]), rank, lam)
            mc = Bi.fit(np.cos(A[:cut]), rank, lam)
            pr = np.arctan2(apply_model(ms, X[cut:]), apply_model(mc, X[cut:]))
            err = np.abs((pr - A[cut:] + np.pi) % (2 * np.pi) - np.pi).mean()
            if best is None or err < best[0]:
                best = (err, rank, lam)
    _, rank, lam = best
    m = {"sin": B.fit(np.sin(A), rank, lam),
         "cos": B.fit(np.cos(A), rank, lam),
         "flee": B.fit(F, rank, lam), "rank": rank, "lam": lam}
    print(f"  [{tag}] n={len(X)}  rank {rank}  L2 {lam:g}")
    return m


def save(m, path):
    np.savez(path,
             sin_mu=m["sin"]["mu"], sin_P=m["sin"]["P"], sin_w=m["sin"]["w"],
             sin_b=m["sin"]["b"],
             cos_mu=m["cos"]["mu"], cos_P=m["cos"]["P"], cos_w=m["cos"]["w"],
             cos_b=m["cos"]["b"],
             flee_mu=m["flee"]["mu"], flee_P=m["flee"]["P"], flee_w=m["flee"]["w"],
             flee_b=m["flee"]["b"], rank=m["rank"], lam=m["lam"])


def report(m, Xte, Ate, Fte, Hte, Ete, tag):
    pred = np.arctan2(apply_model(m["sin"], Xte), apply_model(m["cos"], Xte))
    err = np.abs((pred - Ate + np.pi) % (2 * np.pi) - np.pi)
    msk = np.abs(Ate) > 0.25
    a_side = 100 * side_score(pred, Ate, msk)
    a_err = np.degrees(err[msk].mean())
    a_auc = auc(apply_model(m["flee"], Xte), Fte)
    print(f"{tag:26s}{a_side:13.1f}%{a_err:11.1f}°{a_auc:12.3f}")
    return a_side, a_err, a_auc


if __name__ == "__main__":
    fly = FlyPilot(quiet=False)

    # ---------- held-out test set, collected once, never trained on --------
    print("collecting TEST (mixed policy)...")
    Xte, Ate, Fte, Hte, Ete = collect(fly, TEST_SEEDS, SECS, "mixed")

    # ---------- round 0 ---------------------------------------------------
    print("\nround 0: collecting with script + wander...")
    X0, A0, F0, H0, E0 = collect(fly, TRAIN_SEEDS, SECS, "mixed")
    t = time.time()
    m0 = fit(X0, A0, F0, "round 0")
    save(m0, "readout.npz")
    print(f"  fitted in {time.time()-t:.0f}s")

    # ---------- round 1: the trained pilot generates its own states --------
    fly1 = FlyPilot(readout="readout.npz", quiet=True)
    print("\nround 1: collecting with the round-0 readout driving...")
    X1, A1, F1, H1, E1 = collect(fly1, TRAIN_SEEDS, SECS, "self")

    Xa = np.concatenate([X0, X1])
    Aa = np.concatenate([A0, A1])
    Fa = np.concatenate([F0, F1])
    t = time.time()
    m1 = fit(Xa, Aa, Fa, "round 1 (aggregated)")
    save(m1, "readout_dagger.npz")
    print(f"  fitted in {time.time()-t:.0f}s")

    # ---------- compare ---------------------------------------------------
    print(f"\n=== held out: {len(Xte)} frames, "
          f"{int((np.abs(Ate) > 0.25).sum())} informative ===")
    print(f"{'readout':26s}{'side correct':>14s}{'mean err':>11s}{'flee AUC':>12s}")
    print(f"{'hand-coded, 2 cells':26s}{100*side_score(Hte, Ate, np.abs(Ate)>0.25):13.1f}%"
          f"{'-':>11s}{auc(Ete, Fte):12.3f}")
    report(m0, Xte, Ate, Fte, Hte, Ete, "round 0, 1,314 DNs")
    report(m1, Xte, Ate, Fte, Hte, Ete, "round 1 DAgger, 1,314 DNs")
    print(f"\nround 0 trained on {len(X0):,} frames, round 1 on {len(Xa):,}")
    print("wrote readout_dagger.npz")
