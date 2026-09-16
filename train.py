"""Can a trained readout beat the 2-neuron hand-coded one?

The connectome never changes. What gets fitted is a linear layer on top of the
leaky spike traces of all 1,314 descending neurons — the reservoir-computing
setup flybrain.Readout uses (PCA, then ridge, rank and L2 chosen by holding out
part of the training seeds).

Two things this script is careful about, because the first version got both
wrong:

1. The data collector must NOT keep the target centred. The script pilot always
   faces the nearest zombie, so 75% of its frames have a bearing inside ±0.25
   rad and "predict zero" scores well while learning nothing. Half the frames
   here come from a wandering driver that sweeps the bearing across the whole
   circle.

2. The hand-coded readout is recorded on the SAME frames and scored with the
   SAME rule. When the two DNa02 cells tie, that is no information, so a tie
   counts as half a point for both readouts.

Held out by SEED, never by frame, because frames 33 ms apart are near-copies.
"""
from __future__ import annotations

import sys
import time
import numpy as np

from game import Game, DT
from pilots import FlyPilot, ScriptPilot

TRAIN_SEEDS = [7, 11, 23, 41, 57, 63]
TEST_SEEDS = [88, 101, 119]
SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 75.0
WARM = 30


class MixedDriver:
    """Alternates between playing properly and wandering, so the bearing of the
    nearest zombie covers the whole circle instead of sitting at zero."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.script = ScriptPilot()
        self.mode = 0
        self.left = 0
        self.turn = 0.0
        self.mv = (0.0, 0.0)

    def act(self, obs):
        if self.left <= 0:
            self.mode ^= 1
            self.left = int(self.rng.uniform(1.5, 3.5) / DT)
            self.turn = float(self.rng.uniform(-3.0, 3.0))
            self.mv = (float(self.rng.uniform(-1, 1)), float(self.rng.uniform(-1, 1)))
        self.left -= 1
        if self.mode == 0:
            return self.script.act(obs)
        return {"move": self.mv, "turn": self.turn, "fire": True, "swap": False}


def collect(fly, seeds, secs):
    X, ang, flee, hand_steer, hand_esc = [], [], [], [], []
    for sd in seeds:
        g = Game(n_players=1, seed=sd)
        drv = MixedDriver(seed=sd)
        fly.v[:] = 0.0
        fly.fired = np.zeros(0, np.int64)
        fly.dn_trace[:] = 0.0
        fly.hist.clear()
        for i in range(int(secs / DT)):
            obs = g.observe(0)
            fly.encode_last = fly.encode(obs)
            fly.brain_step(fly.encode_last)
            zs = obs["zombies"]
            if i >= WARM and zs:
                z = min(zs, key=lambda q: q["dist"])
                X.append(fly.dn_trace.copy())
                ang.append(z["rel"])
                flee.append(1.0 if z["dist"] < 170 else 0.0)
                hand_steer.append(fly.rate("steer_R") - fly.rate("steer_L"))
                hand_esc.append(0.5 * (fly.rate("escape_L") + fly.rate("escape_R")))
            cmd = drv.act(obs)
            p = g.players[0]
            p.cmd_move, p.cmd_turn = cmd["move"], cmd["turn"]
            p.cmd_fire, p.cmd_swap = cmd["fire"], cmd["swap"]
            g.step()
            if g.over:
                g = Game(n_players=1, seed=sd + 1000 + i)
        print(f"  seed {sd:3d}: {len(X):6d} frames", flush=True)
    f32 = lambda v: np.array(v, np.float32)
    return f32(X), f32(ang), f32(flee), f32(hand_steer), f32(hand_esc)


class Basis:
    """One SVD of the training features, reused for every rank and every L2."""

    def __init__(self, X):
        self.mu = X.mean(0)
        Xc = X - self.mu
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.Vt, self.S = Vt, S
        self.Z = Xc @ Vt.T

    def fit(self, y, rank, lam):
        r = min(rank, self.Vt.shape[0])
        Z = self.Z[:, :r]
        w = np.linalg.solve(Z.T @ Z + lam * np.eye(r), Z.T @ (y - y.mean()))
        return {"mu": self.mu, "P": self.Vt[:r].T, "w": w, "b": float(y.mean())}


def apply_model(m, X):
    return (X - m["mu"]) @ m["P"] @ m["w"] + m["b"]


def r2(y, p):
    return 1.0 - float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def side_score(pred_signed, true_ang, mask):
    """Fraction of informative frames where the readout points the right way.
    A readout that outputs exactly zero carries no information, so it scores
    half — the same as a coin flip."""
    s = np.sign(pred_signed[mask])
    t = np.sign(true_ang[mask])
    return float(np.mean(np.where(s == 0, 0.5, (s == t).astype(float))))


def auc(score, label):
    order = np.argsort(score)
    lab = label[order]
    n1 = lab.sum()
    n0 = len(lab) - n1
    return float((np.arange(len(lab))[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    fly = FlyPilot(quiet=False)
    print(f"descending population used as features: {len(fly.dn_idx)}")

    t0 = time.time()
    print("collecting train...")
    Xtr, Atr, Ftr, Htr, Etr = collect(fly, TRAIN_SEEDS, SECS)
    print("collecting test...")
    Xte, Ate, Fte, Hte, Ete = collect(fly, TEST_SEEDS, SECS)
    print(f"train {Xtr.shape}  test {Xte.shape}  ({time.time()-t0:.0f}s)")
    print(f"bearing coverage: |rel| > 0.25 rad on "
          f"{100*np.mean(np.abs(Ate) > 0.25):.1f}% of test frames")

    print("fitting basis (one SVD)...", flush=True)
    t1 = time.time()
    B = Basis(Xtr)
    print(f"  {time.time()-t1:.0f}s", flush=True)

    cut = int(0.75 * len(Xtr))
    Bi = Basis(Xtr[:cut])
    best = None
    for rank in (20, 50, 120, 300, 600):
        for lam in (1.0, 10.0, 100.0, 1000.0, 10000.0):
            ms = Bi.fit(np.sin(Atr[:cut]), rank, lam)
            mc = Bi.fit(np.cos(Atr[:cut]), rank, lam)
            pr = np.arctan2(apply_model(ms, Xtr[cut:]), apply_model(mc, Xtr[cut:]))
            err = np.abs((pr - Atr[cut:] + np.pi) % (2 * np.pi) - np.pi).mean()
            if best is None or err < best[0]:
                best = (err, rank, lam)
    _, rank, lam = best
    ms = B.fit(np.sin(Atr), rank, lam)
    mc = B.fit(np.cos(Atr), rank, lam)
    pred = np.arctan2(apply_model(ms, Xte), apply_model(mc, Xte))
    err = np.abs((pred - Ate + np.pi) % (2 * np.pi) - np.pi)

    m = np.abs(Ate) > 0.25
    print("\n=== AIM: which way is the nearest zombie? ===")
    print(f"picked rank {rank}, L2 {lam:g}   informative frames {m.sum()}/{len(Ate)}")
    print(f"{'readout':34s}{'side correct':>14s}{'mean err':>11s}{'median err':>12s}")
    print(f"{'trained, all 1,314 DNs':34s}{100*side_score(pred, Ate, m):13.1f}%"
          f"{np.degrees(err[m].mean()):10.1f}°{np.degrees(np.median(err[m])):11.1f}°")
    hand_err = np.full(m.sum(), np.nan)
    print(f"{'hand-coded, DNa02 only (2 cells)':34s}"
          f"{100*side_score(Hte, Ate, m):13.1f}%{'-':>11s}{'-':>12s}")
    rnd = np.random.default_rng(0).uniform(-np.pi, np.pi, len(Ate))
    print(f"{'random':34s}{50.0:13.1f}%"
          f"{np.degrees(np.abs((rnd-Ate+np.pi)%(2*np.pi)-np.pi)[m].mean()):10.1f}°{'-':>12s}")
    print(f"R2 held out:  sin {r2(np.sin(Ate), apply_model(ms, Xte)):+.3f}"
          f"   cos {r2(np.cos(Ate), apply_model(mc, Xte)):+.3f}")

    mf = B.fit(Ftr, rank, lam)
    pf = apply_model(mf, Xte)
    print("\n=== FLEE: is a zombie within 170 px? ===")
    print(f"positives {100*Fte.mean():.1f}% of test frames")
    print(f"{'trained, all 1,314 DNs':34s} AUC {auc(pf, Fte):.3f}")
    print(f"{'hand-coded, DNp01 only (2 cells)':34s} AUC {auc(Ete, Fte):.3f}")
    print(f"{'random':34s} AUC 0.500")

    np.savez("readout.npz",
             sin_mu=ms["mu"], sin_P=ms["P"], sin_w=ms["w"], sin_b=ms["b"],
             cos_mu=mc["mu"], cos_P=mc["P"], cos_w=mc["w"], cos_b=mc["b"],
             flee_mu=mf["mu"], flee_P=mf["P"], flee_w=mf["w"], flee_b=mf["b"],
             rank=rank, lam=lam)
    print("\nwrote readout.npz")
