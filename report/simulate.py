"""Run the real fly.ai leaky integrate-and-fire loop and record spikes.

Reimplements flybrain.FlyBrain.step() (same constants, same order of operations)
so every number in the report comes from the actual connectome.
W is stored with rows = POSTsynaptic, cols = PREsynaptic, so the synaptic input
of one step is a plain sparse mat-vec: current = W @ spikes.
"""
import numpy as np, scipy.sparse as sp, time, json

DATA = r"C:\Users\Pawaruj_P\fly-data"
Z = np.load(DATA + r"\brain.npz", allow_pickle=True)
W = sp.load_npz(DATA + r"\weights.npz").tocsr()
CT, SC, SIDE, VIS = Z["cell_type"], Z["superclass"], Z["side"], Z["visual"]
N = len(CT)

# constants from flybrain/brain.py
DT, TAU, GAIN, TONIC = 0.020, 0.100, 3.0, 0.14
NOISE_HZ, NOISE_AMP, EYE_GAIN = 1.2, 0.22, 0.62
DECAY = np.float32(np.exp(-DT / TAU))

def cells(types, side=None):
    m = np.isin(CT, types)
    if side: m &= SIDE == side
    return np.flatnonzero(m)

def run(steps, stim=None, stim_from=25, stim_to=75, seed=64, eye=0.45, watch=None):
    rng = np.random.default_rng(seed)
    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    counts = {k: np.zeros(steps) for k in watch}
    total = np.zeros(steps)
    for t in range(steps):
        cur = (W @ s).astype(np.float32)
        v *= DECAY
        v += cur * GAIN + TONIC
        v += (rng.random(N) < NOISE_HZ * DT) * np.float32(NOISE_AMP)
        v[VIS] += eye * EYE_GAIN
        if stim is not None and stim_from <= t < stim_to:
            v[stim[0]] += stim[1]
        fired = np.flatnonzero(v >= 1.0)
        v[fired] = 0.0
        s[:] = 0.0; s[fired] = 1.0
        total[t] = len(fired)
        for k, idx in watch.items():
            counts[k][t] = s[idx].sum()
    return counts, total

WATCH = {"LC4": cells(["LC4"], "L"), "LPLC2": cells(["LPLC2"], "L"), "LC10a": cells(["LC10a"], "L"),
         "DNp01 L (escape)": cells(["DNp01"], "L"), "DNp01 R (escape)": cells(["DNp01"], "R"),
         "DNa02 L (steer)": cells(["DNa02"], "L"), "DNa02 R (steer)": cells(["DNa02"], "R"),
         "MDN (backward)": cells(["MDN"]), "DNg100 (forward)": cells(["DNg100"]),
         "DNg11 (punch)": cells(["DNg11"]), "pIP10 (song)": cells(["pIP10"]),
         "descending (1314)": np.flatnonzero(SC == "descending_neuron"),
         "VNC motor (708)": np.flatnonzero(SC == "vnc_motor"),
         "optic lobe": np.flatnonzero(np.char.startswith(SC.astype(str), "ol_")),
         "central brain": np.flatnonzero(np.char.startswith(SC.astype(str), "cb_"))}

STEPS = 150
runs = {"rest": None,
        "loom L (LC4+LPLC2)": (np.concatenate([cells(["LC4"], "L"), cells(["LPLC2"], "L")]), 0.8),
        "chase L (LC10a)": (cells(["LC10a"], "L"), 0.8)}
out = {}
for label, stim in runs.items():
    t0 = time.time()
    counts, total = run(STEPS, stim=stim, watch=WATCH)
    dur = time.time() - t0
    out[label] = {"rates": {k: (v / len(WATCH[k]) / DT).tolist() for k, v in counts.items()},
                  "total": total.tolist()}
    print(f"{label:22s} {1000*dur/STEPS:5.1f} ms/step   pop {total.mean()/N/DT:5.2f} Hz")

json.dump({"runs": out, "sizes": {k: int(len(v)) for k, v in WATCH.items()},
           "steps": STEPS, "dt": DT, "stim_window": [25, 75]}, open("sim.json", "w"))

print(f"\n{'population':22s} {'n':>5s} {'rest':>7s} {'loom':>7s} {'chase':>7s}   (Hz in stim window)")
for k in WATCH:
    row = [np.mean(out[r]["rates"][k][25:75]) for r in runs]
    print(f"{k:22s} {len(WATCH[k]):5d} {row[0]:7.1f} {row[1]:7.1f} {row[2]:7.1f}")
