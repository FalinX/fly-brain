"""Browser front end for Boxhead 2Play.

The game and the connectome stay in Python — the browser is a renderer and a
keyboard. That keeps the loop tick-locked exactly as it is in the pygame
version: one brain step per game frame, 30 frames a second, no matter how long
the socket takes.

    python server.py --readout readout_dagger.npz
    open http://127.0.0.1:8770

Wire format:
  JSON  {"type":"init", ...}         once, on connect
  bin   Float32Array xyz * 140,638   once, neuron positions
  bin   Uint8Array region id         once
  JSON  {"type":"f", ...}            every frame, game + readout state
  bin   Uint32Array                  every frame, which neurons just spiked
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time

import numpy as np
from aiohttp import web, WSMsgType

from game import Game, DT, ARENA_W, ARENA_H, WALL, WEAPONS
from pilots import FlyPilot, ScriptPilot, HumanPilot
import versions

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("FLY_DATA", os.path.expanduser("~/fly-data"))

REGIONS = [("optic L", "#38bdf8"), ("optic R", "#6366f1"),
           ("central brain", "#fb923c"), ("VNC", "#14b8a6"),
           ("visual proj.", "#ef4444"), ("descending", "#fde047"),
           ("ascending", "#c084fc")]

STATE = {"fly": None, "geom": None}


# ------------------------------------------------------------------ geometry
def load_geometry():
    z = np.load(os.path.join(DATA, "brain.npz"), allow_pickle=True)
    pos = z["positions"]
    sc = z["superclass"].astype(str)
    side = z["side"].astype(str)
    ct = z["cell_type"].astype(str)
    ok = ~np.isnan(pos).any(1)
    idx = np.flatnonzero(ok)

    P = np.column_stack([pos[ok, 0], pos[ok, 2], -pos[ok, 1]]).astype(np.float32)
    P -= (P.max(0) + P.min(0)) / 2
    P /= np.abs(P).max()

    s, sd = sc[ok], side[ok]
    reg = np.full(len(idx), 6, np.uint8)
    ol = np.char.startswith(s, "ol_")
    reg[ol & (sd == "L")] = 0
    reg[ol & (sd == "R")] = 1
    reg[np.char.startswith(s, "cb_")] = 2
    reg[np.char.startswith(s, "vnc_")] = 3
    reg[np.char.startswith(s, "visual_projection")] = 4
    reg[s == "visual_centrifugal"] = 4
    reg[s == "ascending_neuron"] = 6
    reg[s == "descending_neuron"] = 5

    row_of = np.full(len(pos), -1, np.int64)
    row_of[idx] = np.arange(len(idx))

    marks = {}
    for name, types in (("LC4", ["LC4"]), ("LPLC2", ["LPLC2"]), ("LPLC1", ["LPLC1"]),
                        ("LC10a", ["LC10a"]), ("DNp01", ["DNp01"]),
                        ("DNa02", ["DNa02"]), ("MDN", ["MDN"])):
        r = row_of[np.flatnonzero(np.isin(ct, types))]
        r = r[r >= 0]
        if len(r):
            marks[name] = [float(v) for v in P[r].mean(0)]

    sizes = [int((reg == i).sum()) for i in range(len(REGIONS))]
    return {"P": P, "reg": reg, "row_of": row_of, "marks": marks,
            "n": len(idx), "sizes": sizes}


# ------------------------------------------------------------------ session
class Session:
    def __init__(self, p1, p2, seed, version):
        self.n = 1 if p2 == "none" else 2
        self.g = Game(n_players=self.n, seed=seed)
        self.version = version
        self.pilots = [self._make(p1, version)]
        if self.n == 2:
            self.pilots.append(self._make(p2, version))
        self.keys = {"w": 0, "a": 0, "s": 0, "d": 0}
        self.aim = 0.0
        self.fire = False
        self.swap = False
        self.brain_ms = 0.0

    def _make(self, kind, version):
        if kind == "fly":
            fly = STATE["fly"]
            fly.v[:] = 0.0
            fly.fired = np.zeros(0, np.int64)
            fly.dn_trace[:] = 0.0
            fly.hist.clear()
            fly.steer_ema = 0.0
            fly.fleeing = False
            fly.esc_dir = None
            fly.blocked = 0.0
            fly.encoder = version["encoder"]
            fly.barrel_gate = version.get("barrel_gate", "none")
            fly.escape_mode = version.get("escape_mode", "hand")
            if version["readout"]:
                r = np.load(os.path.join(HERE, version["readout"]))
                fly.ro = {k: r[k] for k in r.files}
            else:
                fly.ro = None
            fly.name = version["id"]
            return fly
        if kind == "script":
            return ScriptPilot()
        return HumanPilot()

    def human_cmd(self, p):
        wx = self.keys["d"] - self.keys["a"]
        wy = self.keys["s"] - self.keys["w"]
        da = (self.aim - p.aim + math.pi) % (2 * math.pi) - math.pi
        ca, sa = math.cos(p.aim), math.sin(p.aim)
        return {"move": (wx * ca + wy * sa, -wx * sa + wy * ca),
                "turn": da / DT, "fire": self.fire, "swap": self.swap}

    def step(self):
        for pid, pilot in enumerate(self.pilots):
            p = self.g.players[pid]
            if not p.alive:
                continue
            obs = self.g.observe(pid)
            if isinstance(pilot, HumanPilot):
                cmd = self.human_cmd(p)
            else:
                t0 = time.perf_counter()
                cmd = pilot.act(obs)
                if isinstance(pilot, FlyPilot):
                    self.brain_ms = (0.85 * self.brain_ms
                                     + 0.15 * (time.perf_counter() - t0) * 1000)
            p.cmd_move, p.cmd_turn = cmd["move"], cmd["turn"]
            p.cmd_fire, p.cmd_swap = cmd["fire"], cmd["swap"]
        self.swap = False
        self.g.step()

    # -------------------------------------------------------------- payload
    def frame(self):
        g = self.g
        fly = next((p for p in self.pilots if isinstance(p, FlyPilot)), None)
        out = {
            "type": "f", "t": round(g.t, 2), "wave": g.wave, "over": g.over,
            "players": [{
                "x": round(p.x, 1), "y": round(p.y, 1), "aim": round(p.aim, 3),
                "hp": round(max(0.0, p.hp), 1), "alive": p.alive,
                "kills": p.kills, "score": p.score, "hurt": round(p.hurt_flash, 2),
                "acc": round(100 * p.hits / max(1, p.shots), 1),
                "weapon": WEAPONS[p.weapon].name,
                "pilot": self.pilots[i].name if self.pilots[i] else "-",
            } for i, p in enumerate(g.players)],
            "z": [[round(z.x, 1), round(z.y, 1), 1 if z.kind == "devil" else 0,
                   1 if z.hit_flash > 0 else 0] for z in g.zombies],
            "b": [[round(b.x, 1), round(b.y, 1), round(b.vx, 0), round(b.vy, 0)]
                  for b in g.bullets],
            "bar": [[round(b.x, 1), round(b.y, 1), 1 if b.fuse >= 0 else 0]
                    for b in g.barrels],
            "boom": [[round(e.x, 1), round(e.y, 1), round(e.t / 0.35, 3), e.r]
                     for e in g.booms],
            "ms": round(self.brain_ms, 1),
        }
        if fly is not None:
            d = fly.see or {}
            out["fly"] = {
                "see": {k: round(float(d.get(k, 0.0)), 3) for k in
                        ("loom_L", "loom_R", "threat_L", "threat_R",
                         "chase_L", "chase_R", "shot_L", "shot_R")},
                "dn": {k: round(float(v), 2) for k, v in fly.trace.items()},
                "flee": bool(fly.fleeing),
                "trained": fly.ro is not None,
                "version": fly.name,
                "encoder": fly.encoder,
                "esc": round(fly.esc_dir, 3) if fly.esc_dir is not None else None,
                "blocked": round(float(fly.blocked), 3),
                "has_dodge": bool(fly.ro is not None and "esc_sin_w" in fly.ro),
            }
        return out

    def fired_rows(self):
        fly = next((p for p in self.pilots if isinstance(p, FlyPilot)), None)
        if fly is None or not len(fly.fired):
            return np.zeros(0, np.uint32)
        r = STATE["geom"]["row_of"][fly.fired]
        return r[r >= 0].astype(np.uint32)


# ------------------------------------------------------------------ handlers
async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    q = request.query
    sess = Session(q.get("p1", "human"), q.get("p2", "fly"),
                   int(q.get("seed", "7")), versions.get(q.get("fly", "")))
    geom = STATE["geom"]

    await ws.send_json({
        "type": "init", "arena": [ARENA_W, ARENA_H, WALL], "n": geom["n"],
        "regions": [{"name": n, "color": c, "size": s}
                    for (n, c), s in zip(REGIONS, geom["sizes"])],
        "marks": geom["marks"], "dt": DT,
        "players": [p.name if p else "-" for p in sess.pilots],
        "versions": [{k: v[k] for k in
                      ("id", "label", "headline", "note", "score", "encoder")}
                     for v in versions.available()],
        "version": sess.version["id"],
    })
    await ws.send_bytes(geom["P"].tobytes())
    await ws.send_bytes(geom["reg"].tobytes())

    async def pump():
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                m = json.loads(msg.data)
            except Exception:
                continue
            if m.get("t") == "k":
                sess.keys.update({k: int(v) for k, v in m["keys"].items()})
                sess.aim = float(m.get("aim", sess.aim))
                sess.fire = bool(m.get("fire", False))
                if m.get("swap"):
                    sess.swap = True

    reader = asyncio.create_task(pump())
    next_t = time.perf_counter()
    try:
        while not ws.closed:
            sess.step()
            await ws.send_json(sess.frame())
            await ws.send_bytes(sess.fired_rows().tobytes())
            next_t += DT
            delay = next_t - time.perf_counter()
            if delay < -0.25:            # fell far behind; resync
                next_t = time.perf_counter()
                delay = 0
            await asyncio.sleep(max(0.0, delay))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        reader.cancel()
    return ws


async def index(request):
    return web.FileResponse(os.path.join(HERE, "web", "index.html"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--fly", default="", help="which version to start on")
    a = ap.parse_args()

    print("loading the connectome...")
    STATE["fly"] = FlyPilot(quiet=False)
    STATE["geom"] = load_geometry()
    print("fly versions available: " +
          ", ".join(v["id"] for v in versions.available()))
    print(f"{STATE['geom']['n']:,} neurons have a position and will be drawn")

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/web", os.path.join(HERE, "web"))
    print(f"\n  open  http://127.0.0.1:{a.port}\n")
    web.run_app(app, host="127.0.0.1", port=a.port, print=lambda *_: None)


if __name__ == "__main__":
    main()
