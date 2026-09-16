"""Headless run: measure what the descending neurons actually do during play,
how long a brain frame costs, and how the fly scores against the baseline."""
from __future__ import annotations

import sys
import time
import numpy as np

from game import Game, DT
from pilots import FlyPilot, ScriptPilot


def run(pilot_factory, seconds=60.0, seed=7, collect=False, quiet=True):
    g = Game(n_players=1, seed=seed)
    pilot = pilot_factory()
    frames = int(seconds / DT)
    times, rec = [], {k: [] for k in
                      ("forward", "steer_L", "steer_R", "backward", "escape", "punch", "kick")}
    fires = turns = 0
    for _ in range(frames):
        obs = g.observe(0)
        t0 = time.perf_counter()
        cmd = pilot.act(obs)
        times.append(time.perf_counter() - t0)
        p = g.players[0]
        p.cmd_move, p.cmd_turn = cmd["move"], cmd["turn"]
        p.cmd_fire, p.cmd_swap = cmd["fire"], cmd["swap"]
        fires += bool(cmd["fire"])
        turns += abs(cmd["turn"]) > 0.4
        if collect and hasattr(pilot, "trace"):
            for k, v in pilot.trace.items():
                rec[k].append(v)
        g.step()
        if g.over:
            break
    p = g.players[0]
    return {"pilot": pilot.name, "survived": round(g.t, 1), "wave": g.wave,
            "dmg_barrel": round(p.dmg_barrel, 0), "dmg_contact": round(p.dmg_contact, 0),
            "barrel_shots": p.barrel_shots,
            "kills": p.kills, "score": p.score, "hp": round(max(0.0, p.hp), 1),
            "shots": p.shots, "hits": p.hits,
            "acc": round(100 * p.hits / max(1, p.shots), 1),
            "fire_frames_pct": round(100 * fires / max(1, len(times)), 1),
            "turn_frames_pct": round(100 * turns / max(1, len(times)), 1),
            "ms_med": round(1000 * float(np.median(times)), 2),
            "ms_p95": round(1000 * float(np.percentile(times, 95)), 2),
            "budget_ms": round(1000 * DT, 1)}, rec


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0

    print("=== baseline: 3-rule script ===")
    r, _ = run(ScriptPilot, secs, seed=7)
    print(r)

    print("\n=== the fly ===")
    fly = FlyPilot(quiet=False)
    r2, rec = run(lambda: fly, secs, seed=7, collect=True)
    print(r2)

    print("\n--- descending-neuron readout during play (Hz) ---")
    print(f"{'group':10s}{'min':>8s}{'med':>8s}{'p90':>8s}{'p99':>8s}{'max':>8s}")
    for k, v in rec.items():
        if not v:
            continue
        a = np.array(v)
        print(f"{k:10s}{a.min():8.1f}{np.median(a):8.1f}"
              f"{np.percentile(a,90):8.1f}{np.percentile(a,99):8.1f}{a.max():8.1f}")
