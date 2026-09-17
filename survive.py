"""How long does a fly with a gun last?

Every benchmark in this repo stops at 90 s, and from v6 onward the fly hits that
ceiling in almost every arena — so the survival numbers are censored and the
real answer was never measured.

This runs with no time limit. Waves keep escalating: each one spawns
5 + 2.6 x wave zombies, devils appear from wave 3 and reach 45% of spawns, and
zombie health rises by 3 a wave. Nothing in the game heals the player, so death
is certain; the only question is when.

    python survive.py v7            # 12 arenas, run until it dies
    python survive.py v7 3          # a quick probe on three arenas
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np

import versions
import ensemble
from game import Game, DT
from pilots import ScriptPilot, KiterPilot

# the twelve arenas every benchmark in this repo uses, plus a second block
# added only for the survival question, because the headline claim was sitting
# at t = 2.25 on twelve and deserved a bigger sample
SEEDS = [7, 11, 23, 41, 57, 63, 88, 101, 119, 144, 177, 203,
         5, 19, 37, 52, 71, 96, 113, 128, 151, 166, 188, 211]
HARD_CAP = 1800.0        # 30 minutes; a safety net, not an expected outcome


def build(vid):
    if vid == "script":
        return ScriptPilot(), {"id": "script", "label": "3-rule script"}
    if vid == "kiter":
        return KiterPilot(), {"id": "kiter", "label": "hand-written kiter, no brain"}
    if vid == "kiter-nofire":
        return (KiterPilot(shoot=False),
                {"id": "kiter-nofire", "label": "kiter that never shoots"})
    if vid == "script+barrel":
        return (ScriptPilot(barrel_gate=True),
                {"id": "script+barrel", "label": "3-rule script + v7's barrel check"})
    v = versions.get(vid)
    return ensemble.build(v, n=v.get("flies", 1), quiet=False), v


def run_until_death(pilot, seed, n_barrels=7):
    g = Game(n_players=1, seed=seed, n_barrels=n_barrels)
    if hasattr(pilot, "reset"):
        pilot.reset()
    elif hasattr(pilot, "v"):
        pilot.v[:] = 0.0
        pilot.fired = np.zeros(0, np.int64)
        pilot.dn_trace[:] = 0.0
        pilot.hist.clear()
        pilot.steer_ema = 0.0
        pilot.fleeing = False
    hp_at = {}
    times = []
    while not g.over and g.t < HARD_CAP:
        obs = g.observe(0)
        t0 = time.perf_counter()
        cmd = pilot.act(obs)
        times.append(time.perf_counter() - t0)
        p = g.players[0]
        p.cmd_move, p.cmd_turn = cmd["move"], cmd["turn"]
        p.cmd_fire, p.cmd_swap = cmd["fire"], cmd["swap"]
        g.step()
        w = g.wave
        if w not in hp_at:
            hp_at[w] = round(max(0.0, g.players[0].hp), 1)
    p = g.players[0]
    return {"seed": seed, "survived": round(g.t, 1), "wave": g.wave,
            "kills": p.kills, "score": p.score,
            "acc": round(100 * p.hits / max(1, p.shots), 1),
            "dmg_barrel": round(p.dmg_barrel), "dmg_contact": round(p.dmg_contact),
            "barrel_shots": p.barrel_shots,
            "hp_entering_wave": hp_at,
            "ms_med": round(1000 * float(np.median(times)), 2),
            "capped": g.t >= HARD_CAP}


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "v7"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else len(SEEDS)
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    pilot, meta = build(vid)
    seeds = SEEDS[:n]
    rows = []
    t0 = time.time()
    for sd in seeds:
        r = run_until_death(pilot, sd, nb)
        rows.append(r)
        print(f"seed {sd:3d}  {r['survived']:7.1f}s  wave {r['wave']:2d}  "
              f"{r['kills']:4d} kills  {r['acc']:5.1f}% acc  "
              f"barrel {r['dmg_barrel']:3d}  contact {r['dmg_contact']:3d}"
              f"{'  (HIT CAP)' if r['capped'] else ''}", flush=True)

    g = lambda k: np.array([r[k] for r in rows], float)
    print(f"\n{meta['id']}: survived {g('survived').mean():7.1f} ± {g('survived').std():5.1f} s"
          f"   wave {g('wave').mean():4.1f}   kills {g('kills').mean():6.1f}"
          f"   {g('ms_med').mean():5.1f} ms/frame   ({time.time()-t0:.0f}s of wall clock)")
    print(f"median survival {np.median(g('survived')):.1f} s"
          f"   best {g('survived').max():.1f} s   worst {g('survived').min():.1f} s")

    os.makedirs("results", exist_ok=True)
    out = (f"results/survival_{meta['id']}.json" if nb == 7
           else f"results/survival_{meta['id']}_b{nb}.json")
    json.dump({"version": meta["id"], "label": meta.get("label", meta["id"]),
               "mode": "until death", "hard_cap_s": HARD_CAP, "n_barrels": nb,
               "seeds": seeds, "runs": rows}, open(out, "w"), indent=1)
    print("wrote", out)
