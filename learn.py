"""Teach the readout to dodge, and to stop shooting the barrels.

Four things are fitted on top of the frozen connectome, all linear, all reading
the same 1,314 descending-neuron traces:

    aim       sin/cos of the bearing of the nearest zombie
    flee      is something close enough that I should move
    escape    sin/cos of the direction to move to get away from everything
    blocked   is an explosive barrel sitting in my line of fire

`escape` is the new one that matters. The old pilot fled by walking straight
backwards along its body axis, which is not what a fly does — a real escape is
aimed away from the looming stimulus. With the retinotopic encoder the brain
now knows where the threats are, so the direction can be read back out.

Round 0 collects with script + wander. Round 1 lets the round-0 readout drive
and adds what it sees to the pile — the states it produces are the ones it has
to be right about.
"""
from __future__ import annotations

import math
import sys
import time
import numpy as np

from game import Game, DT, ARENA_W, ARENA_H, WALL
from pilots import FlyPilot, ScriptPilot

TRAIN_SEEDS = [7, 11, 23, 41, 57, 63]
TEST_SEEDS = [88, 101, 119]
SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
WARM = 30
EXPLORE = 0.20

TARGETS = ["aim_sin", "aim_cos", "flee", "esc_sin", "esc_cos", "blocked"]


# --------------------------------------------------------------------- labels
def labels(obs):
    """Everything we want the readout to be able to say, computed from the
    game. Only used while training — at run time the readout sees spikes."""
    p = obs["self"]
    zs = obs["zombies"]
    out = {}
    z = min(zs, key=lambda q: q["dist"]) if zs else None
    a = z["rel"] if z else 0.0
    out["aim_sin"], out["aim_cos"] = math.sin(a), math.cos(a)
    out["flee"] = 1.0 if (z and z["dist"] < 170) else 0.0

    # where to go: away from every zombie nearby, and away from the walls,
    # expressed in the fly's own frame
    vx = vy = 0.0
    for q in zs:
        if q["dist"] < 340:
            w = (1.0 - q["dist"] / 340.0) ** 2
            vx -= math.cos(q["rel"]) * w
            vy -= math.sin(q["rel"]) * w
    wx = (max(0.0, 1 - (p.x - WALL) / 150.0) - max(0.0, 1 - (ARENA_W - WALL - p.x) / 150.0))
    wy = (max(0.0, 1 - (p.y - WALL) / 150.0) - max(0.0, 1 - (ARENA_H - WALL - p.y) / 150.0))
    ca, sa = math.cos(p.aim), math.sin(p.aim)
    vx += 1.2 * (wx * ca + wy * sa)
    vy += 1.2 * (-wx * sa + wy * ca)
    if math.hypot(vx, vy) < 1e-6:
        vx, vy = -1.0, 0.0
    e = math.atan2(vy, vx)
    out["esc_sin"], out["esc_cos"] = math.sin(e), math.cos(e)

    blocked = 0.0
    for b in obs["barrels"]:
        if abs(b["rel"]) < 0.20 and b["dist"] < 320 and (z is None or b["dist"] < z["dist"] + 40):
            blocked = 1.0
            break
    out["blocked"] = blocked
    return out


class MixedDriver:
    """Half playing properly, half wandering, so bearings cover the circle."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.script = ScriptPilot()
        self.mode, self.left = 0, 0
        self.turn, self.mv = 0.0, (0.0, 0.0)

    def act(self, obs):
        if self.left <= 0:
            self.mode ^= 1
            self.left = int(self.rng.uniform(1.5, 3.5) / DT)
            self.turn = float(self.rng.uniform(-3.5, 3.5))
            self.mv = (float(self.rng.uniform(-1, 1)), float(self.rng.uniform(-1, 1)))
        self.left -= 1
        if self.mode == 0:
            return self.script.act(obs)
        return {"move": self.mv, "turn": self.turn, "fire": True, "swap": False}


def reset(fly):
    fly.v[:] = 0.0
    fly.fired = np.zeros(0, np.int64)
    fly.dn_trace[:] = 0.0
    fly.hist.clear()
    fly.steer_ema = 0.0
    fly.fleeing = False


def collect(fly, seeds, secs, policy):
    X, Y = [], {k: [] for k in TARGETS}
    deaths = 0
    for sd in seeds:
        g = Game(n_players=1, seed=sd)
        drv = MixedDriver(seed=sd)
        rng = np.random.default_rng(sd)
        reset(fly)
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
                for k, v in labels(obs).items():
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
                g = Game(n_players=1, seed=sd + 1000 + i)
                reset(fly)
        print(f"  seed {sd:3d}: {len(X):6d} frames", flush=True)
    print(f"  ({deaths} deaths while collecting)")
    return (np.array(X, np.float32),
            {k: np.array(v, np.float32) for k, v in Y.items()})


class Basis:
    def __init__(self, X):
        self.mu = X.mean(0)
        Xc = X - self.mu
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.Vt, self.S, self.Z = Vt, S, Xc @ Vt.T

    def fit(self, y, rank, lam):
        r = min(rank, self.Vt.shape[0])
        Z = self.Z[:, :r]
        w = np.linalg.solve(Z.T @ Z + lam * np.eye(r), Z.T @ (y - y.mean()))
        return {"mu": self.mu, "P": self.Vt[:r].T, "w": w, "b": float(y.mean())}


def apply_model(m, X):
    return (X - m["mu"]) @ m["P"] @ m["w"] + m["b"]


def side_score(pred, true, mask):
    s, t = np.sign(pred[mask]), np.sign(true[mask])
    return float(np.mean(np.where(s == 0, 0.5, (s == t).astype(float))))


def auc(score, label):
    o = np.argsort(score)
    lab = label[o]
    n1 = lab.sum()
    n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((np.arange(len(lab))[lab == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def fit_all(X, Y, tag):
    B = Basis(X)
    cut = int(0.75 * len(X))
    Bi = Basis(X[:cut])
    best = None
    for rank in (20, 50, 120, 300):
        for lam in (1.0, 10.0, 100.0, 1000.0):
            ms = Bi.fit(Y["aim_sin"][:cut], rank, lam)
            mc = Bi.fit(Y["aim_cos"][:cut], rank, lam)
            pr = np.arctan2(apply_model(ms, X[cut:]), apply_model(mc, X[cut:]))
            tr = np.arctan2(Y["aim_sin"][cut:], Y["aim_cos"][cut:])
            err = np.abs((pr - tr + np.pi) % (2 * np.pi) - np.pi).mean()
            if best is None or err < best[0]:
                best = (err, rank, lam)
    _, rank, lam = best
    m = {k: B.fit(Y[k], rank, lam) for k in TARGETS}
    m["rank"], m["lam"] = rank, lam
    print(f"  [{tag}] n={len(X)}  rank {rank}  L2 {lam:g}")
    return m


def save(m, path):
    d = {}
    for k in TARGETS:
        d[k + "_mu"] = m[k]["mu"]
        d[k + "_P"] = m[k]["P"]
        d[k + "_w"] = m[k]["w"]
        d[k + "_b"] = m[k]["b"]
    d["rank"], d["lam"] = m["rank"], m["lam"]
    np.savez(path, **d)


def report(m, Xte, Yte, tag):
    aim = np.arctan2(Yte["aim_sin"], Yte["aim_cos"])
    p_aim = np.arctan2(apply_model(m["aim_sin"], Xte), apply_model(m["aim_cos"], Xte))
    msk = np.abs(aim) > 0.25
    e_true = np.arctan2(Yte["esc_sin"], Yte["esc_cos"])
    p_esc = np.arctan2(apply_model(m["esc_sin"], Xte), apply_model(m["esc_cos"], Xte))
    esc_err = np.degrees(np.abs((p_esc - e_true + np.pi) % (2 * np.pi) - np.pi).mean())
    print(f"{tag:22s}{100*side_score(p_aim, aim, msk):11.1f}%"
          f"{np.degrees(np.abs((p_aim-aim+np.pi)%(2*np.pi)-np.pi)[msk].mean()):10.1f}°"
          f"{esc_err:12.1f}°{auc(apply_model(m['flee'], Xte), Yte['flee']):10.3f}"
          f"{auc(apply_model(m['blocked'], Xte), Yte['blocked']):11.3f}")


if __name__ == "__main__":
    fly = FlyPilot(quiet=False)
    t0 = time.time()

    print("collecting TEST...")
    Xte, Yte = collect(fly, TEST_SEEDS, SECS, "mixed")
    print("\nround 0: script + wander...")
    X0, Y0 = collect(fly, TRAIN_SEEDS, SECS, "mixed")
    m0 = fit_all(X0, Y0, "round 0")
    save(m0, "readout_r0.npz")

    r = np.load("readout_r0.npz")
    fly.ro = {k: r[k] for k in r.files}
    print("\nround 1: the round-0 readout drives...")
    X1, Y1 = collect(fly, TRAIN_SEEDS, SECS, "self")
    fly.ro = None
    Xa = np.concatenate([X0, X1])
    Ya = {k: np.concatenate([Y0[k], Y1[k]]) for k in TARGETS}
    m1 = fit_all(Xa, Ya, "round 1")
    save(m1, "readout_dodge.npz")

    print(f"\n=== held out, {len(Xte)} frames   ({time.time()-t0:.0f}s total) ===")
    print(f"{'readout':22s}{'aim side':>12s}{'aim err':>10s}{'escape err':>12s}"
          f"{'flee AUC':>10s}{'barrel AUC':>11s}")
    report(m0, Xte, Yte, "round 0")
    report(m1, Xte, Yte, "round 1 DAgger")
    print(f"\nbarrel-in-line positives: {100*Yte['blocked'].mean():.1f}% of frames")
    print("wrote readout_dodge.npz")
