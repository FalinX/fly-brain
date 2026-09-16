"""3D renders of the MaleCNS connectome from the real neuron positions."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

Z = np.load(r"C:\Users\Pawaruj_P\fly-data\brain.npz", allow_pickle=True)
POS, SC, SIDE, CT = Z["positions"], Z["superclass"].astype(str), Z["side"].astype(str), Z["cell_type"].astype(str)
OK = ~np.isnan(POS).any(1)
# axes for display: X = left-right, Y = anterior-posterior (brain -> VNC), Z = dorsal-ventral
P = np.column_stack([POS[:, 0], POS[:, 2], -POS[:, 1]]) / 1000.0

BG = "#0b0e14"
plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG,
                     "font.family": "DejaVu Sans", "text.color": "#e6edf3"})

def division(sc, side):
    d = np.full(len(sc), "other", "<U16")
    ol = np.char.startswith(sc, "ol_")
    d[ol & (side == "L")] = "optic L"; d[ol & (side == "R")] = "optic R"
    d[np.char.startswith(sc, "cb_")] = "central brain"
    d[np.char.startswith(sc, "vnc_")] = "VNC"
    d[np.char.startswith(sc, "visual_projection")] = "visual proj."
    d[sc == "visual_centrifugal"] = "visual proj."
    d[sc == "descending_neuron"] = "descending"
    d[sc == "ascending_neuron"] = "ascending"
    return d

DIV = division(SC, SIDE)
COL = {"optic L": "#38bdf8", "optic R": "#4f46e5", "central brain": "#fb923c",
       "VNC": "#14b8a6", "visual proj.": "#ef4444", "descending": "#fde047",
       "ascending": "#c084fc", "other": "#3a4150"}
SIZE = {"descending": 2.5, "visual proj.": 0.8, "ascending": 1.0}
ORDER = ["other", "optic L", "optic R", "central brain", "VNC", "ascending", "visual proj.", "descending"]

def frame(ax, m, pad=0.03):
    """Limits = true data range, box aspect = same ratios, so anatomy keeps its
    proportions but the object fills the panel instead of a padded cube."""
    p = P[m]
    lo, hi_ = p.min(0), p.max(0)
    ctr, rng = (hi_ + lo) / 2, np.maximum(hi_ - lo, 1e-6) * (1 + pad)
    for setter, c, r in ((ax.set_xlim, ctr[0], rng[0]), (ax.set_ylim, ctr[1], rng[1]),
                         (ax.set_zlim, ctr[2], rng[2])):
        setter(c - r / 2, c + r / 2)
    ax.set_box_aspect(tuple(rng / rng.max())); ax.set_axis_off()

def panel(ax, elev, azim, title, sel, zoom=1.0):
    for k in ORDER:
        m = OK & (DIV == k) & sel
        if not m.any(): continue
        a = 0.18 if k in ("optic L", "optic R") else (0.3 if k == "other" else 0.45)
        ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=SIZE.get(k, 0.5), c=COL[k],
                   alpha=a, linewidths=0, rasterized=True, depthshade=False)
    frame(ax, OK & sel)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, color="#9aa6b2", fontsize=11, pad=-2)
    try: ax.set_proj_type("persp", focal_length=0.4)
    except Exception: pass

BRAIN = P[:, 1] < 55.0
ALL = np.ones(len(P), bool)

fig = plt.figure(figsize=(15, 6.4))
gs = fig.add_gridspec(1, 3, wspace=0.02, left=0.01, right=0.99, top=0.955, bottom=0.085)
for i, (e, a, t, sel) in enumerate([(4, -90, "BRAIN — front (facing you)", BRAIN),
                                    (80, 0, "WHOLE CNS — brain (left) + ventral nerve cord (right)", ALL),
                                    (89, -90, "BRAIN — top (dorsal)", BRAIN)]):
    ax = fig.add_subplot(gs[i], projection="3d", facecolor=BG)
    panel(ax, e, a, t, sel)
h = [plt.Line2D([], [], marker="o", ls="", ms=7, c=COL[k], label=f"{k}  {int((DIV==k).sum()):,}")
     for k in ["optic L", "optic R", "central brain", "VNC", "visual proj.", "descending", "ascending"]]
fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, fontsize=9.5,
           labelcolor="#e6edf3", bbox_to_anchor=(0.5, 0.005))
fig.savefig("fig/fig_3d_cns.png", dpi=190)
plt.close(fig)
print("fig_3d_cns.png")

# ---------- FIGURE B: pathway ---------------------------------------------
LAYERS = [(np.isin(CT, ["LC4", "LPLC2"]), "#ef4444", 16, "LC4 + LPLC2 — looming detectors (311)"),
          (CT == "LC10a", "#fb923c", 16, "LC10a — chase / target tracking (275)"),
          (CT == "DNp01", "#fde047", 110, "DNp01 — giant fiber, escape (2)"),
          (CT == "DNa02", "#38bdf8", 110, "DNa02 — steering (2)"),
          (SC == "vnc_motor", "#14b8a6", 10, "VNC motor neurons (708)")]

def hi(ax, elev, azim, title, sel):
    m = OK & sel
    ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=0.55, c="#2c3849", alpha=0.6,
               linewidths=0, rasterized=True, depthshade=False)
    for mask, c, s, _ in LAYERS:
        mm = OK & mask & sel
        if not mm.any(): continue
        ew = 0.8 if s > 40 else 0
        ax.scatter(P[mm, 0], P[mm, 1], P[mm, 2], s=s, c=c, alpha=0.95,
                   edgecolors="#0b0e14", linewidths=ew, rasterized=True, depthshade=False)
    frame(ax, m); ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, color="#9aa6b2", fontsize=11, pad=-2)
    try: ax.set_proj_type("persp", focal_length=0.4)
    except Exception: pass

fig = plt.figure(figsize=(15, 6.4))
gs = fig.add_gridspec(1, 3, wspace=0.02, left=0.01, right=0.99, top=0.955, bottom=0.095)
for i, (e, a, t, sel) in enumerate([(4, -90, "front — detectors sit in both optic lobes", BRAIN),
                                    (25, -60, "three-quarter", BRAIN),
                                    (80, 0, "brain → VNC — the whole motor chain", ALL)]):
    ax = fig.add_subplot(gs[i], projection="3d", facecolor=BG)
    hi(ax, e, a, t, sel)
h = [plt.Line2D([], [], marker="o", ls="", ms=8, c=c, label=lab) for _, c, _, lab in LAYERS]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
           labelcolor="#e6edf3", bbox_to_anchor=(0.5, 0.005))
fig.savefig("fig/fig_3d_pathway.png", dpi=190)
plt.close(fig)
print("fig_3d_pathway.png")
