"""Architecture diagram, decoder map, connectivity statistics."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

INK, MUTED, GRID = "#16181d", "#6f6a60", "#e0dbd0"
C = {"in": "#2563eb", "res": "#0f172a", "out": "#d97706", "dec": "#0d9488",
     "red": "#dc2626", "vio": "#7c3aed", "bg": "#f6f8fb"}
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "figure.facecolor": "#FBFAF7", "savefig.facecolor": "#FBFAF7"})


def box(ax, x, y, w, h, fc, ec, lw=1.4, r=0.02, alpha=1.0, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=%s" % r,
                                fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z))


def arrow(ax, x1, y1, x2, y2, c=MUTED, lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=c, lw=lw,
                                 linestyle=ls, mutation_scale=16, zorder=3,
                                 shrinkA=0, shrinkB=0))


def txt(ax, x, y, s, size=10, c=INK, w="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=c, weight=w, ha=ha, va=va, zorder=4, **kw)


# =========== FIGURE: full architecture ====================================
fig, ax = plt.subplots(figsize=(13.2, 7.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 58)
ax.axis("off")

txt(ax, 1, 55.5, "How one step of fly.ai runs", 17, INK, "bold", ha="left")
txt(ax, 1, 52.3, "The middle block is never trained. Only the two thin ends are written by a human.",
    11, MUTED, ha="left")

box(ax, 1, 20, 19, 26, "#eef4ff", C["in"], 1.6)
txt(ax, 10.5, 43.5, "1 - ENCODER", 11.5, C["in"], "bold")
txt(ax, 10.5, 40.8, "written by hand", 9, MUTED)
txt(ax, 10.5, 36.8, "game / world state\ninjected as voltage\ninto sensory neurons", 10)
box(ax, 2.6, 21.8, 15.8, 11.5, "white", "#c7d7f5", 1.1)
txt(ax, 10.5, 31.4, "two routes", 9.5, C["in"], "bold")
txt(ax, 10.5, 27.0, "A   6,006 photoreceptors\n      1-D panorama, azimuth -1..+1\n\n"
                    "B   feature detectors, direct:\n      LC4  LPLC2  LPLC1  LC10a", 8.8)

box(ax, 24, 8, 34, 38, "#0f172a", "#0f172a", 0)
txt(ax, 41, 43.2, "2 - THE CONNECTOME, frozen", 12, "white", "bold")
txt(ax, 41, 40.5, "no trained weights, no learning, no policy", 9.2, "#9fb0c7")
sub = [("optic lobes", "95,501", "#38bdf8"), ("central brain", "37,229", "#fb923c"),
       ("VNC", "20,429", "#14b8a6"), ("visual proj.", "9,764", "#ef4444")]
for i, (n, v, c) in enumerate(sub):
    x = 26 + i * 7.9
    box(ax, x, 27, 7.0, 10.5, "#182236", c, 1.2)
    txt(ax, x + 3.5, 33.6, n.replace(" ", "\n"), 8.4, "white")
    txt(ax, x + 3.5, 29.2, v, 9.6, c, "bold")
box(ax, 26, 10.5, 30, 14, "#111c2f", "#2a3a55", 1.2)
txt(ax, 41, 22.2, "every step, for all 166,700 at once", 9.2, "#9fb0c7")
txt(ax, 41, 17.4, r"$v \leftarrow e^{-dt/\tau} v \;+\; gain\cdot W s \;+\; tonic \;+\; noise$"
                  "\n" r"$v \geq 1 \;\Rightarrow\; \mathrm{spike},\;\; v \leftarrow 0$", 12.5, "white")
txt(ax, 41, 12.4, "W = 25,582,938 synapses,  38.4% inhibitory,  rows sum to 1",
    8.8, "#9fb0c7")

box(ax, 62, 20, 14, 26, "#fff7ed", C["out"], 1.6)
txt(ax, 69, 43.5, "3 - DESCENDING", 11.5, C["out"], "bold")
txt(ax, 69, 40.8, "the only way out of the brain", 8.4, MUTED)
txt(ax, 69, 35.5, "1,314\nneurons", 15, C["out"], "bold")
txt(ax, 69, 30.2, "0.79% of the\nwhole network", 9, MUTED)
txt(ax, 69, 25.0, "count their spikes\nin a short window", 9)

box(ax, 80, 20, 19, 26, "#ecfdf5", C["dec"], 1.6)
txt(ax, 89.5, 43.5, "4 - DECODER", 11.5, C["dec"], "bold")
txt(ax, 89.5, 40.8, "written by hand OR trained", 9, MUTED)
box(ax, 81.4, 31.5, 16.2, 7.0, "white", "#b9e6da", 1.1)
txt(ax, 89.5, 36.7, "hand-coded", 9.5, C["dec"], "bold")
txt(ax, 89.5, 33.6, "12 named neurons to 6 keys\nthreshold on spike rate", 8.6)
box(ax, 81.4, 22.2, 16.2, 8.0, "white", "#b9e6da", 1.1)
txt(ax, 89.5, 28.4, "trained readout", 9.5, C["dec"], "bold")
txt(ax, 89.5, 24.9, "flybrain.Readout\nPCA then ridge / logistic\ncross-validated rank + L2", 8.6)

for x1, x2 in [(20, 24), (58, 62), (76, 80)]:
    arrow(ax, x1, 33, x2, 33, MUTED, 2.2)
arrow(ax, 99, 33, 99, 48.6, MUTED, 1.6, ls="--")
arrow(ax, 99, 48.6, 10.5, 48.6, MUTED, 1.6, ls="--")
txt(ax, 55, 47.3, "action changes the world, which changes the next step's input", 9.2, MUTED)

txt(ax, 1, 5.5, "TRAINED", 9.5, C["red"], "bold", ha="left")
txt(ax, 11, 5.5, "only the decoder's linear layer - a few thousand parameters at most",
    9.5, MUTED, ha="left")
txt(ax, 1, 2.6, "FROZEN", 9.5, INK, "bold", ha="left")
txt(ax, 11, 2.6, "the 25.6 M synapse weights, taken from electron microscopy of a real male fly",
    9.5, MUTED, ha="left")
fig.tight_layout()
fig.savefig("fig/fig_pipeline.png", dpi=190)
plt.close(fig)
print("pipeline")

# =========== FIGURE: decoder map ==========================================
ROWS = [("forward", "DNg100", 1, 1, "walk forward", "#2563eb"),
        ("steer", "DNa02", 1, 1, "turn left / right", "#0d9488"),
        ("backward", "MDN", 2, 2, "walk backward - the published moonwalker neuron", "#7c3aed"),
        ("escape", "DNp01", 1, 1, "giant fiber - the jump-away reflex", "#dc2626"),
        ("punch", "DNg11", 3, 3, "foreleg strike", "#d97706"),
        ("kick", "pIP10", 1, 1, "courtship-song command neuron", "#db2777")]
fig, ax = plt.subplots(figsize=(13.5, 4.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 46)
ax.axis("off")
txt(ax, 1, 43, "The hand-coded decoder in full", 16, INK, "bold", ha="left")
txt(ax, 1, 39.6, "6 game actions read from 12 individual neurons - 0.9% of the descending "
    "population, 0.007% of the brain.", 10.5, MUTED, ha="left")
for x, h in [(3, "ACTION"), (20, "CELL TYPE"), (38, "LEFT"), (46, "RIGHT"),
             (56, "WHAT IT DOES IN THE REAL FLY")]:
    txt(ax, x, 34.3, h, 8.6, MUTED, "bold", ha="left")
ax.plot([1, 99], [32.8, 32.8], color=GRID, lw=1.2)
for i, (act, ct, nl, nr, note, c) in enumerate(ROWS):
    y = 28.6 - i * 4.7
    ax.add_patch(Rectangle((1, y - 1.7), 98, 4.0, fc=C["bg"] if i % 2 else "white", ec="none"))
    ax.add_patch(Rectangle((1, y - 1.7), 0.7, 4.0, fc=c, ec="none"))
    txt(ax, 3, y, act, 11, INK, "bold", ha="left")
    txt(ax, 20, y, ct, 11, c, "bold", ha="left")
    txt(ax, 38, y, str(nl), 10.5, INK, ha="left")
    txt(ax, 46, y, str(nr), 10.5, INK, ha="left")
    txt(ax, 56, y, note, 9.8, MUTED, ha="left")
fig.tight_layout()
fig.savefig("fig/fig_decoder.png", dpi=190)
plt.close(fig)
print("decoder")

# =========== FIGURE: connectivity statistics ==============================
import scipy.sparse as sp
W = sp.load_npz(r"C:\Users\Pawaruj_P\fly-data\weights.npz")
d = W.data
indeg = np.diff(W.tocsr().indptr)
outdeg = np.diff(W.tocsc().indptr)
fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
for a in axs:
    a.grid(color=GRID, lw=0.7, axis="y")
    a.set_axisbelow(True)
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
    a.spines["left"].set_color(GRID)
    a.spines["bottom"].set_color(GRID)
    a.tick_params(colors=MUTED)

bins = np.logspace(0, np.log10(max(indeg.max(), outdeg.max())), 46)
axs[0].hist(indeg[indeg > 0], bins=bins, color="#2563eb", alpha=0.72, label="inputs per neuron")
axs[0].hist(outdeg[outdeg > 0], bins=bins, color="#d97706", alpha=0.62, label="outputs per neuron")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_title("how many partners a neuron has", fontsize=11, loc="left", color=INK)
axs[0].set_xlabel("partners")
axs[0].legend(frameon=False, fontsize=9, loc="upper left")
axs[0].text(0.97, 0.95, "median %d in / %d out\nbiggest hub %s in / %s out"
            % (np.median(indeg), np.median(outdeg), f"{indeg.max():,}", f"{outdeg.max():,}"),
            transform=axs[0].transAxes, ha="right", va="top", fontsize=8.6, color=MUTED)

axs[1].hist(np.abs(d), bins=np.logspace(-6, 0, 50), color="#0d9488", alpha=0.8)
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_title("synapse strength after normalisation", fontsize=11, loc="left", color=INK)
axs[1].set_xlabel("|w|   (the inputs of one neuron sum to 1)")
axs[1].text(0.03, 0.95, "median |w| = %.4f\nso hundreds of inputs\nmust agree to fire one neuron"
            % np.median(np.abs(d)),
            transform=axs[1].transAxes, va="top", fontsize=8.6, color=MUTED)

exc, inh = int((d > 0).sum()), int((d < 0).sum())
axs[2].barh([1, 0], [exc, inh], color=["#2563eb", "#dc2626"], height=0.55)
axs[2].set_yticks([1, 0])
axs[2].set_yticklabels(["excitatory\n(+w)", "inhibitory (-w)\nGABA / glutamate\n/ histamine"],
                       fontsize=9.5)
axs[2].set_xlim(0, exc * 1.42)
for yy, v in [(1, exc), (0, inh)]:
    axs[2].text(v * 1.04, yy, "%s\n%.1f%%" % (f"{v:,}", 100 * v / len(d)),
                va="center", fontsize=9.5, color=INK)
axs[2].set_title("sign of the 25,582,938 synapses", fontsize=11, loc="left", color=INK)
axs[2].grid(axis="y", visible=False)
axs[2].set_xticks([])
fig.suptitle("What the wiring actually looks like", fontsize=13.5, weight="bold", x=0.007, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig/fig_stats.png", dpi=190)
plt.close(fig)
print("stats")
