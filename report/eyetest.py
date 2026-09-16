"""Does route A (the real photoreceptors) actually reach the detectors?

The demos all take route B: inject straight into LC4 / LPLC2 / LPLC1 / LC10a.
Route A exists in the API (brain.step(eye_drive)) but the project says the
lamina pathway fades in a spiking model. This measures how much fades.
"""
import numpy as np, scipy.sparse as sp, json

DATA = r"C:\Users\Pawaruj_P\fly-data"
Z = np.load(DATA + r"\brain.npz", allow_pickle=True)
W = sp.load_npz(DATA + r"\weights.npz").tocsr()
CT, SC, SIDE = Z["cell_type"], Z["superclass"], Z["side"]
VIS, AZ = Z["visual"], Z["azimuth"]
N = len(CT)

DT, TAU, GAIN, TONIC = 0.020, 0.100, 3.0, 0.14
NOISE_HZ, NOISE_AMP, EYE_GAIN = 1.2, 0.22, 0.62
DECAY = np.float32(np.exp(-DT / TAU))


def cells(types, side=None):
    m = np.isin(CT, types)
    if side:
        m &= SIDE == side
    return np.flatnonzero(m)


WATCH = {"LC4 L": cells(["LC4"], "L"), "LC4 R": cells(["LC4"], "R"),
         "LPLC2 L": cells(["LPLC2"], "L"), "LPLC2 R": cells(["LPLC2"], "R"),
         "LC10a L": cells(["LC10a"], "L"), "LC10a R": cells(["LC10a"], "R"),
         "DNp01 L": cells(["DNp01"], "L"), "DNp01 R": cells(["DNp01"], "R"),
         "DNa02 L": cells(["DNa02"], "L"), "DNa02 R": cells(["DNa02"], "R"),
         "optic lobe": np.flatnonzero(np.char.startswith(SC.astype(str), "ol_")),
         "descending": np.flatnonzero(SC == "descending_neuron")}


def run(steps, eye_fn, seed=64):
    rng = np.random.default_rng(seed)
    v = np.zeros(N, np.float32)
    s = np.zeros(N, np.float32)
    acc = {k: 0.0 for k in WATCH}
    warm = 25
    for t in range(steps):
        cur = (W @ s).astype(np.float32)
        v *= DECAY
        v += cur * GAIN + TONIC
        v += (rng.random(N) < NOISE_HZ * DT) * np.float32(NOISE_AMP)
        v[VIS] += eye_fn(t) * EYE_GAIN
        fired = np.flatnonzero(v >= 1.0)
        v[fired] = 0.0
        s[:] = 0.0
        s[fired] = 1.0
        if t >= warm:
            for k, idx in WATCH.items():
                acc[k] += s[idx].sum()
    win = (steps - warm) * DT
    return {k: acc[k] / len(WATCH[k]) / win for k in WATCH}


STEPS = 150
flat = np.full(len(VIS), 0.45, np.float32)


def bump(center, width, height, base=0.45):
    """A bright object sitting at one azimuth, like something filling that part
    of the visual field."""
    d = np.exp(-0.5 * ((AZ - center) / width) ** 2)
    return (base + height * d).astype(np.float32)


def loom(center, t, base=0.45):
    """Something rushing in: the object grows and brightens over the run."""
    grow = np.clip((t - 25) / 50.0, 0, 1)
    return bump(center, 0.08 + 0.42 * grow, 0.55 * grow, base)


CONDS = {
    "A0  uniform light (what inject.py uses)": lambda t: flat,
    "A1  static object far LEFT (az -0.7)": lambda t: bump(-0.7, 0.12, 0.55),
    "A2  static object far RIGHT (az +0.7)": lambda t: bump(+0.7, 0.12, 0.55),
    "A3  object LOOMING from the LEFT": lambda t: loom(-0.7, t),
    "A4  whole field flashing bright (1.0)": lambda t: np.full(len(VIS), 1.0, np.float32),
}

print(f"photoreceptors: {len(VIS)}   azimuth range {AZ.min():+.2f} .. {AZ.max():+.2f}")
print(f"receptors on the LEFT half (az<0): {(AZ < 0).sum()}   RIGHT half: {(AZ > 0).sum()}\n")

rows = {}
for label, fn in CONDS.items():
    rows[label] = run(STEPS, fn)
    print("done:", label)

print(f"\n{'condition':42s}" + "".join(f"{k:>11s}" for k in WATCH))
for label, r in rows.items():
    print(f"{label:42s}" + "".join(f"{r[k]:11.2f}" for k in WATCH))

base = rows["A0  uniform light (what inject.py uses)"]
print(f"\n{'change vs uniform light':42s}" + "".join(f"{k:>11s}" for k in WATCH))
for label, r in rows.items():
    if label.startswith("A0"):
        continue
    print(f"{label:42s}" + "".join(f"{r[k]-base[k]:+11.2f}" for k in WATCH))

json.dump({"rows": rows, "sizes": {k: int(len(v)) for k, v in WATCH.items()}},
          open("eyetest.json", "w"))
