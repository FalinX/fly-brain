"""Clean full-bleed cover render: one brain, no labels, no legend."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

Z = np.load(r"C:\Users\Pawaruj_P\fly-data\brain.npz", allow_pickle=True)
POS, SC, SIDE = Z["positions"], Z["superclass"].astype(str), Z["side"].astype(str)
OK = ~np.isnan(POS).any(1)
P = np.column_stack([POS[:, 0], POS[:, 2], -POS[:, 1]]) / 1000.0
BRAIN = OK & (P[:, 1] < 55.0)

BG = "#0b0e14"
COL = {"olL": "#38bdf8", "olR": "#4f46e5", "cb": "#fb923c", "vp": "#ef4444", "dn": "#fde047"}
ol = np.char.startswith(SC, "ol_")
LAYERS = [(BRAIN & ol & (SIDE == "L"), COL["olL"], 0.30, 0.9),
          (BRAIN & ol & (SIDE == "R"), COL["olR"], 0.30, 0.9),
          (BRAIN & np.char.startswith(SC, "cb_"), COL["cb"], 0.42, 1.0),
          (BRAIN & np.char.startswith(SC, "visual_projection"), COL["vp"], 0.60, 1.3),
          (BRAIN & (SC == "descending_neuron"), COL["dn"], 0.95, 3.2)]

fig = plt.figure(figsize=(16, 9), facecolor=BG)
ax = fig.add_axes([-0.06, -0.10, 1.12, 1.20], projection="3d", facecolor=BG)
for m, c, a, s in LAYERS:
    ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=s, c=c, alpha=a, linewidths=0,
               rasterized=True, depthshade=False)
p = P[BRAIN]
lo, hi = p.min(0), p.max(0)
ctr, rng = (hi + lo) / 2, (hi - lo) * 1.02
for setter, cc, rr in ((ax.set_xlim, ctr[0], rng[0]), (ax.set_ylim, ctr[1], rng[1]),
                       (ax.set_zlim, ctr[2], rng[2])):
    setter(cc - rr / 2, cc + rr / 2)
ax.set_box_aspect(tuple(rng / rng.max()))
ax.set_axis_off()
ax.view_init(elev=16, azim=-68)
try:
    ax.set_proj_type("persp", focal_length=0.32)
except Exception:
    pass
fig.savefig("fig/fig_cover.png", dpi=170, facecolor=BG)
print("cover")
