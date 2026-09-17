"""Controllers for a Boxhead player.

FlyPilot is the real one: the MaleCNS connectome runs one leaky
integrate-and-fire step per game frame, tick-locked, and the only things a
human wrote are the encoder (game -> voltage on the fly's own detectors) and
the decoder (descending-neuron spikes -> keys).

ScriptPilot is the honest baseline. Every claim about the fly is measured
against it.
"""
from __future__ import annotations

import math
import os
import numpy as np
from scipy import sparse

import numba

from game import DT, WEAPONS

DATA = os.environ.get("FLY_DATA", os.path.expanduser("~/fly-data"))


# --------------------------------------------------------------- baseline
class ScriptPilot:
    """Three rules: face the nearest zombie, shoot it, walk away if it is close.

    `barrel_gate=True` adds the same line-of-fire check v7 uses, so the
    baseline can be compared against a fly that has one. Without it the script
    shoots its own barrels and several runs end that way, which flatters the
    fly.
    """

    def __init__(self, turn_rate=6.5, barrel_gate=False):
        self.turn_rate = turn_rate
        self.barrel_gate = barrel_gate
        self.name = "script+barrel" if barrel_gate else "script"

    def act(self, obs):
        zs = obs["zombies"]
        if not zs:
            return {"move": (0.0, 0.0), "turn": 0.0, "fire": False, "swap": False}
        z = min(zs, key=lambda z: z["dist"])
        turn = max(-self.turn_rate, min(self.turn_rate, z["rel"] * 6.0))
        aimed = abs(z["rel"]) < 0.22
        back = 1.0 if z["dist"] < 150 else 0.0
        fire = aimed
        if fire and self.barrel_gate:
            spread = WEAPONS[obs["self"].weapon].spread
            for b in obs["barrels"]:
                if b["dist"] > 520 or abs(b["rel"]) > 1.2:
                    continue
                if b["dist"] * abs(math.sin(b["rel"])) < 11.0 + b["dist"] * spread + 6.0:
                    fire = False
                    break
        return {"move": (-back, 0.0), "turn": turn, "fire": fire, "swap": False}


# --------------------------------------------------------------- human
class HumanPilot:
    name = "human"

    def __init__(self):
        self.state = {"move": (0.0, 0.0), "turn": 0.0, "fire": False, "swap": False}

    def act(self, obs):
        return self.state


# --------------------------------------------------------------- the fly
@numba.njit(nogil=True, parallel=True, cache=True)
def _propagate(indptr, indices, weights, fired, n):
    """Sum the outgoing weights of every neuron that spiked. Mirrors
    flybrain/brain.py exactly: the matrix is (post, pre) so its CSC columns are
    presynaptic neurons."""
    threads = numba.get_num_threads()
    partial = np.zeros((threads, n), np.float32)
    chunk = (len(fired) + threads - 1) // threads
    for t in numba.prange(threads):
        acc = partial[t]
        for k in range(t * chunk, min(len(fired), (t + 1) * chunk)):
            j = fired[k]
            for e in range(indptr[j], indptr[j + 1]):
                acc[indices[e]] += weights[e]
    current = np.zeros(n, np.float32)
    for i in numba.prange(n):
        s = np.float32(0.0)
        for t in range(threads):
            s += partial[t, i]
        current[i] = s
    return current


class FlyPilot:
    """166,700 neurons, 25,582,938 synapses, zero trained weights."""
    name = "fly"

    # published constants (flybrain/brain.py)
    TAU = 0.100
    GAIN = 3.0
    TONIC_20MS = 0.14
    NOISE_HZ = 1.2
    NOISE_AMP = 0.22
    EYE_GAIN = 0.62

    # encoder constants (flybrain/eyes.py)
    LOOM_GAIN = 10.0
    SHOT_GAIN = 10.0
    CHASE_BASE = 0.6
    CHASE_GAIN = 0.2
    THREAT_MAX = 0.8
    CAP = 0.8

    def __init__(self, dt=DT, seed=64, window=9, data=DATA, quiet=False,
                 readout=None, encoder='retino', label=None, barrel_gate='trained',
                 escape_mode='trained'):
        z = np.load(os.path.join(data, "brain.npz"), allow_pickle=True)
        W = sparse.load_npz(os.path.join(data, "weights.npz")).tocsc()
        self.indptr, self.indices, self.weights = W.indptr, W.indices, W.data
        self.n = W.shape[0]
        ct = z["cell_type"].astype(str)
        side = z["side"].astype(str)

        def cells(types, s=None):
            m = np.isin(ct, types)
            if s:
                m &= side == s
            return np.flatnonzero(m).astype(np.int64)

        # --- the fly's own detectors, split by eye
        self.det = {
            "loom_L": cells(["LPLC2"], "L"), "loom_R": cells(["LPLC2"], "R"),
            "threat_L": cells(["LC4"], "L"), "threat_R": cells(["LC4"], "R"),
            "shot_L": cells(["LPLC1"], "L"), "shot_R": cells(["LPLC1"], "R"),
            "chase_L": cells(["LC10a"], "L"), "chase_R": cells(["LC10a"], "R"),
        }
        # --- an approximate retinotopic axis for each detector population
        #
        # Soma positions inside one LC type lie on a 1-D sheet: PC1 carries
        # 70-89% of the variance. That axis is the population's place in the
        # lobula, which is laid out retinotopically. It is an APPROXIMATION —
        # a soma is not a receptive-field centre, and the sheet is oblique to
        # the body axes — but it is the direction information the data has, and
        # collapsing each population to one left/right number throws it away.
        #
        # Sign is anchored so that increasing value = more posterior, i.e.
        # further back in the visual field.
        pos = z["positions"]
        self.bear = {}
        for key, idx in self.det.items():
            p = pos[idx]
            good = ~np.isnan(p).any(1)
            u = np.zeros(len(idx), np.float32)
            if good.sum() > 4:
                c = p[good] - p[good].mean(0)
                _, _, vt = np.linalg.svd(c, full_matrices=False)
                pr = c @ vt[0]
                if np.corrcoef(pr, p[good, 2])[0, 1] < 0:
                    pr = -pr
                r = np.ptp(pr) or 1.0
                u[good] = ((pr - pr.min()) / r).astype(np.float32)
            sgn = -1.0 if key.endswith("_L") else 1.0
            # 13 deg straight ahead ... 167 deg straight behind
            self.bear[key] = (sgn * (0.22 + u * 2.70)).astype(np.float32)
        self.see = {}

        # --- the readout groups that ship inside brain.npz
        self.grp = {k.removeprefix("group_"): z[k].astype(np.int64)
                    for k in z.files if k.startswith("group_")}
        # --- every descending neuron, for a trained readout
        sc = z["superclass"].astype(str)
        self.dn_idx = np.flatnonzero(sc == "descending_neuron").astype(np.int64)
        self.dn_trace = np.zeros(len(self.dn_idx), np.float32)
        self.DN_DECAY = np.float32(math.exp(-dt / 0.15))   # 150 ms leaky trace

        self.dt = dt
        self.decay = np.float32(math.exp(-dt / self.TAU))
        # brain.py rescales tonic so a silent network behaves the same at any dt
        self.tonic = np.float32(self.TONIC_20MS * (1 - math.exp(-dt / self.TAU))
                                / (1 - math.exp(-0.020 / self.TAU)))
        self.rng = np.random.default_rng(seed)
        self.v = np.zeros(self.n, np.float32)
        self.fired = np.zeros(0, np.int64)
        self.window = window
        self.hist: list[np.ndarray] = []
        self.steps = 0
        self.fleeing = False
        self.steer_ema = 0.0
        self.esc_dir = None
        self.blocked = 0.0
        self.encoder = encoder
        self.barrel_gate = barrel_gate
        self.escape_mode = escape_mode
        self.encode_last = {}
        self.see = {}
        self.last = {"move": (0.0, 0.0), "turn": 0.0, "fire": False, "swap": False}
        # optional trained linear readout over ALL 1,314 descending neurons
        self.ro = None
        if label:
            self.name = label
        if readout:
            r = np.load(readout)
            self.ro = {k: r[k] for k in r.files}
            if not label:
                self.name = "fly-trained"
            if not quiet:
                print(f"[FlyPilot] trained readout loaded: rank {int(r['rank'])}, "
                      f"L2 {float(r['lam']):g}")
        self.trace = {k: 0.0 for k in
                      ("forward", "steer_L", "steer_R", "backward", "escape", "punch", "kick")}
        if not quiet:
            print(f"[FlyPilot] {self.n:,} neurons  {len(self.weights):,} synapses  "
                  f"dt={dt*1000:.1f} ms  decay={self.decay:.4f}  tonic={self.tonic:.4f}")

    # ------------------------------------------------------------ encoder
    SIGMA = 0.50          # rad, how broad one detector's receptive field is

    def _bump(self, key, bearing, strength, out):
        """Add a Gaussian of drive centred on where the thing actually is."""
        d = self.bear[key] - bearing
        d = (d + math.pi) % (2 * math.pi) - math.pi
        np.add(out, strength * np.exp(-0.5 * (d / self.SIGMA) ** 2), out=out)

    def encode_bins(self, obs):
        """The first encoder: one number per eye.

        Every zombie on the left raises the same single value on the whole left
        population, and anything more than 90 degrees off the nose is dropped
        entirely. Kept so the early versions can still be run and compared.
        """
        drive = {k: 0.0 for k in self.det}
        for z in obs["zombies"]:
            if z["dist"] > 520:
                continue
            side = "L" if z["rel"] < 0 else "R"
            front = max(0.0, math.cos(z["rel"]))
            expand = max(0.0, z["closing"]) / max(z["dist"], 30.0)
            loom = min(self.CAP, self.LOOM_GAIN * expand * 0.12) * (0.35 + 0.65 * front)
            drive["loom_" + side] = max(drive["loom_" + side], loom)
            near = max(0.0, 1.0 - z["dist"] / 300.0)
            threat = min(self.THREAT_MAX, near * near * (0.4 + 0.6 * front))
            if z["kind"] == "devil":
                threat = min(self.THREAT_MAX, threat * 1.5)
            drive["threat_" + side] = max(drive["threat_" + side], threat)
            if z["kind"] == "devil":
                v = min(self.CAP, self.SHOT_GAIN * expand * 0.10)
                drive["shot_" + side] = max(drive["shot_" + side], v)
        for b in obs["barrels"]:
            if b["fuse"] >= 0 and b["dist"] < 220:
                side = "L" if b["rel"] < 0 else "R"
                drive["shot_" + side] = min(self.CAP, drive["shot_" + side] + 0.5)
        if obs["zombies"]:
            z = min(obs["zombies"], key=lambda q: q["dist"])
            if abs(z["rel"]) < 2.2:
                side = "L" if z["rel"] < 0 else "R"
                size = max(0.0, 1.0 - z["dist"] / 520.0)
                drive["chase_" + side] = min(
                    self.CAP, self.CHASE_BASE + self.CHASE_GAIN * size * 2.0)
        self.see = dict(drive)
        return drive

    def encode(self, obs):
        if self.encoder == "bins":
            return self.encode_bins(obs)
        return self.encode_retino(obs)

    def encode_retino(self, obs):
        """Game state -> a voltage pattern across each detector population.

        Not one number per eye: every zombie lands on the detectors whose place
        in the lobula matches its bearing, so the brain gets direction, not just
        a side — including things behind, which the old two-bin encoder could
        not represent at all.
        """
        drive = {k: np.zeros(len(v), np.float32) for k, v in self.det.items()}
        for z in obs["zombies"]:
            if z["dist"] > 560:
                continue
            b = z["rel"]
            side = "L" if b < 0 else "R"
            expand = max(0.0, z["closing"]) / max(z["dist"], 30.0)
            loom = min(self.CAP, self.LOOM_GAIN * expand * 0.16)
            if loom > 0.01:
                self._bump("loom_" + side, b, loom, drive["loom_" + side])
            near = max(0.0, 1.0 - z["dist"] / 320.0)
            threat = near * near * (1.5 if z["kind"] == "devil" else 1.0)
            if threat > 0.01:
                self._bump("threat_" + side, b, min(self.THREAT_MAX, threat),
                           drive["threat_" + side])
        # LPLC1 answers "small object". A barrel is not a fly stimulus at all —
        # putting it here is a human decision, made so the brain can see the
        # thing that keeps blowing it up.
        for bar in obs["barrels"]:
            if bar["dist"] > 420:
                continue
            side = "L" if bar["rel"] < 0 else "R"
            v = (0.75 if bar["fuse"] >= 0 else 0.34) * max(0.15, 1.0 - bar["dist"] / 420.0)
            self._bump("shot_" + side, bar["rel"], v, drive["shot_" + side])
        # LC10a tracks one target, the way it tracks one female
        if obs["zombies"]:
            z = min(obs["zombies"], key=lambda q: q["dist"])
            side = "L" if z["rel"] < 0 else "R"
            size = max(0.0, 1.0 - z["dist"] / 560.0)
            self._bump("chase_" + side, z["rel"],
                       min(self.CAP, self.CHASE_BASE + self.CHASE_GAIN * size * 2.0),
                       drive["chase_" + side])
        for k in drive:
            np.clip(drive[k], 0.0, self.CAP, out=drive[k])
        self.see = {k: float(v.max()) if len(v) else 0.0 for k, v in drive.items()}
        return drive

    # ------------------------------------------------------------ one step
    def brain_step(self, drive):
        current = _propagate(self.indptr, self.indices, self.weights,
                             self.fired, self.n) if len(self.fired) else \
            np.zeros(self.n, np.float32)
        self.v *= self.decay
        self.v += current * np.float32(self.GAIN) + self.tonic
        self.v += (self.rng.random(self.n) < self.NOISE_HZ * self.dt) * np.float32(self.NOISE_AMP)
        for k, amount in drive.items():
            a = np.asarray(amount, np.float32)
            if a.size == 1:
                if float(a) > 0.0:
                    self.v[self.det[k]] += a
            elif a.any():
                self.v[self.det[k]] += a
        fired = np.flatnonzero(self.v >= 1.0)
        self.v[fired] = 0.0
        self.fired = fired.astype(np.int64)
        self.steps += 1
        spk = np.zeros(self.n, np.uint8)
        spk[fired] = 1
        self.dn_trace *= self.DN_DECAY
        self.dn_trace += spk[self.dn_idx]
        self.hist.append(spk)
        if len(self.hist) > self.window:
            self.hist.pop(0)

    def rate(self, key):
        """Spikes per neuron per second for a readout group, over the window."""
        idx = self.grp[key]
        if not self.hist or len(idx) == 0:
            return 0.0
        tot = sum(int(h[idx].sum()) for h in self.hist)
        return tot / len(idx) / (len(self.hist) * self.dt)

    # ------------------------------------------------------------ decoder
    #
    # Calibrated against 3,600 logged frames (diag.py). Only two of the six
    # hand-picked channels carry the encoder's signal:
    #
    #   DNa02  steering   corr(chase_L, steer_L) = +0.59, cross term -0.31
    #                     sign matches the target's side 64.8% of frames
    #   DNp01  escape     2.98 Hz with nothing near, 14.21 Hz with 3 zombies
    #                     inside 220 px
    #
    # The other three are not usable and are NOT wired to a key:
    #   DNg100 forward    silent 97.2% of frames, |corr| < 0.05
    #   DNg11  punch      |corr| < 0.05 against every encoder channel
    #   pIP10  kick       |corr| < 0.07
    #
    # Anything below marked HAND is a human rule, not the fly.
    ESCAPE_ON = 7.0           # Hz; measured 2.98 idle / 8.06 with one closing
    ESCAPE_OFF = 3.5
    TURN_K = 2.3              # Hz of DNa02 difference -> rad/s
    TURN_MAX = 6.5
    TURN_EMA = 0.45           # smoothing on the difference, a readout filter

    def _ro(self, key):
        x = self.dn_trace - self.ro[key + "_mu"]
        return float(x @ self.ro[key + "_P"] @ self.ro[key + "_w"] + self.ro[key + "_b"])

    def _has(self, key):
        return self.ro is not None and (key + "_w") in self.ro

    def act(self, obs):
        self.encode_last = self.encode(obs)
        self.brain_step(self.encode_last)
        return self.decode(obs)

    def clone(self, seed):
        """Another fly with the same wiring and its own noise.

        The weight arrays, the detector index tables and the readout are shared
        by reference — they are never written to — so N copies cost N times the
        compute but only one copy of the 205 MB matrix.
        """
        f = FlyPilot.__new__(FlyPilot)
        f.__dict__.update(self.__dict__)
        f.rng = np.random.default_rng(seed)
        f.v = np.zeros(self.n, np.float32)
        f.fired = np.zeros(0, np.int64)
        f.hist = []
        f.dn_trace = np.zeros(len(self.dn_idx), np.float32)
        f.steer_ema = 0.0
        f.fleeing = False
        f.esc_dir = None
        f.blocked = 0.0
        f.trace = dict(self.trace)
        f.see = {}
        f.encode_last = {}
        return f

    def decode(self, obs):

        sl, sr = self.rate("steer_L"), self.rate("steer_R")
        esc = 0.5 * (self.rate("escape_L") + self.rate("escape_R"))
        self.trace.update(
            forward=0.5 * (self.rate("forward_L") + self.rate("forward_R")),
            steer_L=sl, steer_R=sr,
            backward=0.5 * (self.rate("backward_L") + self.rate("backward_R")),
            escape=esc,
            punch=0.5 * (self.rate("punch_L") + self.rate("punch_R")),
            kick=0.5 * (self.rate("kick_L") + self.rate("kick_R")))

        if self.ro is not None:
            # TRAINED — a linear readout of all 1,314 descending neurons.
            # The connectome itself is untouched; only this layer was fitted.
            if self._has("aim_sin"):
                bearing = math.atan2(self._ro("aim_sin"), self._ro("aim_cos"))
            else:
                bearing = math.atan2(self._ro("sin"), self._ro("cos"))
            self.steer_ema += 0.5 * (bearing - self.steer_ema)
            turn = max(-self.TURN_MAX, min(self.TURN_MAX, self.steer_ema * 4.0))
        else:
            # BRAIN — steering. Left eye drives the left DNa02, so left winning
            # means turn left. This is the LC10a -> DNa02 courtship-pursuit
            # circuit. One neuron per side spiking inside a 300 ms window is a
            # coarse signal, so the difference is smoothed first.
            self.steer_ema += self.TURN_EMA * ((sr - sl) - self.steer_ema)
            turn = max(-self.TURN_MAX, min(self.TURN_MAX, self.steer_ema * self.TURN_K))

        # BRAIN — escape. Schmitt trigger so one quiet frame does not drop it.
        if self.ro is not None:
            f = self._ro("flee")
            if f >= 0.60:
                self.fleeing = True
            elif f <= 0.35:
                self.fleeing = False
        elif esc >= self.ESCAPE_ON:
            self.fleeing = True
        elif esc <= self.ESCAPE_OFF:
            self.fleeing = False

        # DODGE — where to go when it needs to leave.
        # Untrained, it can only walk backwards along its body axis. With the
        # trained escape readout it gets a direction, which is what a real
        # escape is: aimed away from the looming stimulus, not simply reverse.
        if self.escape_mode == "trained" and self._has("esc_sin"):
            esc_dir = math.atan2(self._ro("esc_sin"), self._ro("esc_cos"))
            self.esc_dir = esc_dir
        else:
            esc_dir = None
            self.esc_dir = None
        if self.fleeing:
            fwd = -1.0 if esc_dir is None else math.cos(esc_dir)
        else:
            fwd = 0.35 if (obs["zombies"] and
                           min(z["dist"] for z in obs["zombies"]) > 320) else 0.0
        from game import ARENA_W, ARENA_H, WALL
        p = obs["self"]
        strafe = 0.0
        if self.fleeing and esc_dir is not None:
            # the trained heading already includes the wall push
            strafe = math.sin(esc_dir)
        elif self.fleeing:
            # HAND fallback — do not reverse into a wall
            rx = max(0.0, 1 - (p.x - WALL) / 140.0) - max(0.0, 1 - (ARENA_W - WALL - p.x) / 140.0)
            ry = max(0.0, 1 - (p.y - WALL) / 140.0) - max(0.0, 1 - (ARENA_H - WALL - p.y) / 140.0)
            if rx or ry:
                ca, sa = math.cos(p.aim), math.sin(p.aim)
                strafe = max(-1.0, min(1.0, (-rx * sa + ry * ca) * 1.6))

        # TRIGGER — ammo is unlimited, so it is held down, except when
        # something says an explosive barrel is in the line of fire. One blast
        # costs 26 HP out of 100, so this decides games.
        if self.barrel_gate == "trained" and self._has("blocked"):
            # READOUT — held-out AUC 0.608, barely above chance. A barrel is a
            # still object; LPLC1 answers "small thing moving in", so the fly
            # has no circuit this question maps onto.
            self.blocked = self._ro("blocked")
            fire = self.blocked < 0.45
        elif self.barrel_gate in ("hand", "tight"):
            # HAND — geometry, not the fly.
            #
            # "hand" is the first version: a fixed cone. It stops the fly
            # blowing itself up, but +-0.30 rad at 420 px means holding fire
            # whenever a barrel is within 124 px of the line, and with seven
            # barrels on the map that is most of the time. It cost v6 about
            # fifteen kills a run.
            #
            # "tight" asks the real question instead: will a bullet actually
            # reach the barrel? A barrel is 22 px across and the gun scatters
            # by its own spread, so the width to clear is the barrel's radius
            # plus the cone the weapon itself throws at that range.
            p_ = obs["self"]
            spread = WEAPONS[p_.weapon].spread
            self.blocked = 0.0
            for b in obs["barrels"]:
                if b["dist"] > 520 or abs(b["rel"]) > 1.2:
                    continue
                if self.barrel_gate == "hand":
                    if abs(b["rel"]) < 0.30 and b["dist"] < 420:
                        self.blocked = 1.0
                        break
                else:
                    perp = b["dist"] * abs(math.sin(b["rel"]))
                    if perp < 11.0 + b["dist"] * spread + 6.0:
                        self.blocked = 1.0
                        break
            fire = self.blocked < 0.5
        else:
            self.blocked = 0.0
            fire = True
        # HAND — shotgun in a crowd, uzi otherwise
        crowd = sum(1 for z in obs["zombies"] if z["dist"] < 240)
        want = 1 if crowd >= 3 else 2
        swap = (want != p.weapon) and (self.steps % 20 == 0)

        self.last = {"move": (fwd, strafe), "turn": turn, "fire": fire, "swap": swap}
        return self.last
