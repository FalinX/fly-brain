"""Second way to get a cleaner signal without training anything: run several
copies of the same fly.

Same wiring, same encoder, different noise stream. Whatever the connectome
really computes is shared by every copy; the noise is not, so averaging cancels
it. fly.ai measured this on SSH Fighter — chase went from 63% with one fly to
72% with eight voting.

The copies share the weight arrays, so N flies cost N x compute but only one
copy of the 205 MB matrix.
"""
from __future__ import annotations

import numpy as np

from game import DT
from pilots import FlyPilot


class EnsembleFlyPilot:
    name = "fly-ensemble"

    def __init__(self, n=2, dt=DT, base_seed=64, **kw):
        self.flies = []
        first = FlyPilot(dt=dt, seed=base_seed, quiet=False, **kw)
        self.flies.append(first)
        for i in range(1, n):
            f = FlyPilot.__new__(FlyPilot)
            f.__dict__.update({k: v for k, v in first.__dict__.items()})
            # shared, read-only: indptr / indices / weights / det / grp / dn_idx
            f.rng = np.random.default_rng(base_seed + 1000 * i)
            f.v = np.zeros(first.n, np.float32)
            f.fired = np.zeros(0, np.int64)
            f.hist = []
            f.dn_trace = np.zeros(len(first.dn_idx), np.float32)
            f.steer_ema = 0.0
            f.fleeing = False
            f.trace = dict(first.trace)
            self.flies.append(f)
        self.n_flies = n
        self.fleeing = False
        self.trace = dict(first.trace)
        self.n = first.n
        self.weights = first.weights
        self.encode_last = {}
        print(f"[Ensemble] {n} copies of the same connectome, "
              f"one shared weight matrix")

    def act(self, obs):
        cmds = [f.act(obs) for f in self.flies]
        self.encode_last = self.flies[0].encode_last
        # average the readouts, not the keystrokes
        turn = float(np.mean([c["turn"] for c in cmds]))
        votes = sum(f.fleeing for f in self.flies)
        self.fleeing = votes * 2 >= self.n_flies      # majority
        fwd = float(np.mean([c["move"][0] for c in cmds]))
        strafe = float(np.mean([c["move"][1] for c in cmds]))
        for k in self.trace:
            self.trace[k] = float(np.mean([f.trace[k] for f in self.flies]))
        return {"move": (fwd, strafe), "turn": turn,
                "fire": True, "swap": cmds[0]["swap"]}
