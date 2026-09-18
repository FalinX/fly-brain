"""Run several copies of the same fly and let them vote.

Same wiring, same encoder, same readout — the only difference is the noise
stream each copy gets. Whatever the connectome is really computing is shared
by every copy; the noise is not, so averaging cancels it. fly.ai measured this
on SSH Fighter: moves toward the opponent went from 63% with one fly to 72%
with eight voting.

The copies share the weight arrays by reference, so N flies cost N times the
compute but only one copy of the 205 MB matrix. The encoder is deterministic
given the same observation, so it runs once and every copy is handed the same
drive — only `brain_step` is duplicated.

Frame budget is the real limit. v7 costs about 10 ms of a 33.3 ms frame, so
two copies fit and four do not.
"""
from __future__ import annotations

import numpy as np

from game import DT
from pilots import FlyPilot


class EnsemblePilot:
    def __init__(self, base: FlyPilot, n=2, base_seed=64, label=None):
        self.flies = [base] + [base.clone(base_seed + 7919 * i) for i in range(1, n)]
        self.n_flies = n
        self.name = label or f"{base.name}x{n}"
        # things the HUD reads off a pilot
        self.n = base.n
        self.weights = base.weights
        self.ro = base.ro
        self.encoder = base.encoder
        self.dn_idx = base.dn_idx
        self.trace = dict(base.trace)
        self.see = {}
        self.encode_last = {}
        self.fleeing = False
        self.esc_dir = None
        self.blocked = 0.0
        self.disagree = 0.0        # how often the copies want different things

    # the live 3D view wants one spike set to draw
    @property
    def fired(self):
        return self.flies[0].fired

    def reset(self):
        for f in self.flies:
            f.v[:] = 0.0
            f.fired = np.zeros(0, np.int64)
            f.dn_trace[:] = 0.0
            f.hist.clear()
            f.steer_ema = 0.0
            f.fleeing = False

    def act(self, obs):
        drive = self.flies[0].encode(obs)      # deterministic, so share it
        self.see = dict(self.flies[0].see)
        self.encode_last = drive
        cmds = []
        for f in self.flies:
            f.see = self.see
            f.encode_last = drive
            f.brain_step(drive)
            cmds.append(f.decode(obs))

        turn = float(np.mean([c["turn"] for c in cmds]))
        fwd = float(np.mean([c["move"][0] for c in cmds]))
        strafe = float(np.mean([c["move"][1] for c in cmds]))
        votes = sum(f.fleeing for f in self.flies)
        self.fleeing = votes * 2 >= self.n_flies
        # the barrel gate is geometry, identical in every copy
        self.blocked = self.flies[0].blocked
        fire = cmds[0]["fire"]

        for k in self.trace:
            self.trace[k] = float(np.mean([f.trace[k] for f in self.flies]))
        signs = [np.sign(c["turn"]) for c in cmds]
        self.disagree = 0.0 if len(set(signs)) == 1 else 1.0

        return {"move": (fwd, strafe), "turn": turn, "fire": fire,
                "swap": cmds[0]["swap"]}


def build(version, n=2, quiet=False):
    """An ensemble of `n` copies configured exactly like `version`."""
    base = FlyPilot(readout=version["readout"], encoder=version["encoder"],
                    barrel_gate=version.get("barrel_gate", "none"),
                    escape_mode=version.get("escape_mode", "hand"),
                    move_mode=version.get("move_mode", "reverse"),
                    tonic20=version.get("tonic20"),
                    olfaction=version.get("olfaction", False),
                    reach=version.get("reach", (260.0, 26.0)),
                    blind=version.get("blind", False),
                    label=version["id"], quiet=quiet)
    if n <= 1:
        return base
    e = EnsemblePilot(base, n=n, label=f"{version['id']}x{n}")
    if not quiet:
        print(f"[Ensemble] {n} copies of {version['id']}, one shared weight matrix")
    return e
