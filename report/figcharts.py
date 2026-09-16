"""Charts: composition, LIF dynamics, response to loom/chase, connectivity stats."""
import numpy as np, json, matplotlib, collections
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED, GRID = "#16181d", "#6f6a60", "#e0dbd0"
ACC = {"blue": "#2563eb", "red": "#dc2626", "amber": "#d97706", "teal": "#0d9488",
       "violet": "#7c3aed", "slate": "#64748b"}
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "figure.facecolor": "#FBFAF7",
                     "savefig.facecolor": "#FBFAF7", "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True})

Z = np.load(r"C:\Users\Pawaruj_P\fly-data\brain.npz", allow_pickle=True)
SC = Z["superclass"].astype(str)

# ---------- 1. composition -------------------------------------------------
GROUPS = [("Optic lobes — vision", ["ol_intrinsic", "ol_sensory"], ACC["blue"]),
          ("Visual projection / centrifugal", ["visual_projection", "visual_centrifugal",
                                               "visual_projection_tbc"], ACC["red"]),
          ("Central brain", ["cb_intrinsic", "cb_sensory", "cb_motor", "cb_endocrine",
                             "cb_efferent", "cb_sensory_tbc"], ACC["amber"]),
          ("Ventral nerve cord (VNC)", ["vnc_intrinsic", "vnc_sensory", "vnc_motor",
                                        "vnc_efferent", "vnc_endocrine", "vnc_tbc",
                                        "vnc_sensory_tbc"], ACC["teal"]),
          ("Descending (brain → body)", ["descending_neuron", "descending_neuron_tbc",
                                         "sensory_descending", "efferent_descending"], "#eab308"),
          ("Ascending (body → brain)", ["ascending_neuron", "sensory_ascending",
                                        "sensory_ascending_tbc", "efferent_ascending"], ACC["violet"]),
          ("Gut nervous system (ENS)", ["ENS"], ACC["slate"])]
cnt = collections.Counter(SC)
names = [g[0] for g in GROUPS]
vals = [sum(cnt[s] for s in g[1]) for g in GROUPS]
cols = [g[2] for g in GROUPS]

fig, ax = plt.subplots(figsize=(12.5, 3.5))
y = np.arange(len(names))[::-1]
ax.barh(y, vals, color=cols, height=0.62)
for yi, v in zip(y, vals):
    ax.text(v + 1400, yi, f"{v:,}   {100*v/len(SC):.1f}%", va="center", fontsize=10, color=INK)
ax.set_yticks(y, names, fontsize=9.5)
ax.set_xlim(0, 112000); ax.set_xlabel("neurons")
ax.set_title("What the 166,700 neurons are  —  more than half of the brain is eye",
             fontsize=12.5, weight="bold", loc="left", pad=9)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig("fig/fig_composition.png", dpi=190); plt.close(fig)
print("composition")

# ---------- 2. LIF dynamics ------------------------------------------------
DT, TAU, TONIC, DECAY = 0.020, 0.100, 0.14, np.exp(-0.020 / 0.100)
steps = 60
t = np.arange(steps) * DT * 1000
inp = np.zeros(steps); inp[12:38] = 0.30      # excitatory drive arriving each step
inp[38:46] = -0.25                            # an inhibitory (GABA/glutamate) partner
v, trace, spikes = 0.0, [], []
for k in range(steps):
    v = v * DECAY + inp[k] + TONIC * 0.25
    trace.append(min(v, 1.0))          # draw the spike peak, then the reset
    if v >= 1.0:
        spikes.append(k); v = 0.0
trace = np.array(trace)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.5, 4.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.12})
a1.step(t, inp, where="mid", color=ACC["slate"], lw=1.6)
a1.fill_between(t, 0, inp, step="mid", where=inp > 0, color=ACC["blue"], alpha=0.30)
a1.fill_between(t, 0, inp, step="mid", where=inp < 0, color=ACC["red"], alpha=0.30)
a1.axhline(0, color=GRID, lw=1)
a1.set_ylabel("input\ngain·W·spikes", fontsize=9.5)
a1.text(t[20], 0.34, "excitatory (+w)", color=ACC["blue"], fontsize=9.5)
a1.text(t[38], -0.42, "inhibitory (−w: GABA / glutamate / histamine)", color=ACC["red"], fontsize=9.5)
a1.set_ylim(-0.55, 0.55)

a2.axhline(1.0, color=ACC["red"], lw=1.4, ls="--")
a2.text(t[-1], 1.02, "threshold = 1.0", color=ACC["red"], ha="right", fontsize=9.5)
a2.plot(t, trace, color=INK, lw=2.0)
for s in spikes:
    a2.axvline(t[s], color=ACC["amber"], lw=1.2, alpha=0.8)
    a2.plot([t[s]], [1.0], marker="v", color=ACC["amber"], ms=8)
a2.set_ylabel("membrane voltage  v", fontsize=10)
a2.set_xlabel("time (ms)   —   one step = dt = 20 ms")
a2.set_ylim(-0.32, 1.16)
a2.text(t[2], 0.95, r"$v \leftarrow e^{-dt/\tau}\,v \;+\; gain\cdot W s \;+\; tonic \;+\; noise$"
        "\n" r"$v \geq 1 \Rightarrow \mathrm{spike},\; v \leftarrow 0$",
        fontsize=11.5, va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f6f8fb", ec=GRID))
a1.set_title("One neuron, one step  —  leaky integrate-and-fire  (τ = 100 ms, dt = 20 ms, threshold 1.0)",
             fontsize=13, weight="bold", loc="left", pad=10)
fig.savefig("fig/fig_lif.png", dpi=190); plt.close(fig)
print("lif")

# ---------- 3. loom / chase response --------------------------------------
S = json.load(open("sim.json"))
R, DTs = S["runs"], S["dt"]
tms = np.arange(S["steps"]) * DTs
lo, hi = S["stim_window"]
PANELS = [("LC4", "INPUT · LC4 — looming detector, LEFT", ACC["red"]),
          ("DNp01 L (escape)", "OUTPUT · DNp01 LEFT — giant fiber, escape", ACC["blue"]),
          ("LC10a", "INPUT · LC10a — chase detector, LEFT", ACC["amber"]),
          ("DNa02 L (steer)", "OUTPUT · DNa02 LEFT — steering", ACC["teal"]),
          ("descending (1314)", "all 1,314 descending neurons — population average", ACC["slate"]),
          ("DNp01 R (escape)", "CONTROL · DNp01 RIGHT — wrong side, stays silent", ACC["slate"])]
RUNS = [("rest", "#9aa3af", "no stimulus"),
        ("loom L (LC4+LPLC2)", ACC["red"], "loom on the LEFT eye"),
        ("chase L (LC10a)", ACC["amber"], "chase target on the LEFT")]

fig, axes = plt.subplots(2, 3, figsize=(14.2, 5.0), sharex=True)
for ax, (key, title, _) in zip(axes.ravel(), PANELS):
    ax.axvspan(lo * DTs, hi * DTs, color="#eef2f8", zorder=0)
    for rk, c, lab in RUNS:
        ax.plot(tms, R[rk]["rates"][key], color=c, lw=1.7, label=lab, alpha=0.95)
    ax.set_title(f"{title}   (n={S['sizes'][key]})", fontsize=10, loc="left", color=INK)
    ax.set_ylabel("Hz", fontsize=9)
    ax.set_ylim(-1.5, 53) if "descending" not in key else ax.set_ylim(-0.2, 6)
for ax in axes[-1]:
    ax.set_xlabel("time (s)")
axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Stimulate the fly's own detectors on ONE side — the right descending neuron answers, "
             "on the SAME side", fontsize=13, weight="bold", x=0.012, ha="left")
fig.text(0.012, 0.935, "shaded band = stimulus on (0.5 s – 1.5 s).  Simulation run here with the "
         "published constants: tonic 0.14, gain 3.0, eye drive 0.45.", fontsize=9.5, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.925])
fig.savefig("fig/fig_response.png", dpi=190); plt.close(fig)
print("response")
