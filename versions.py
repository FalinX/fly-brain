"""Every fly we built, in the order we built it, so they can be swapped live.

Each version pairs an encoder with a readout. That pairing matters: a readout
fitted on the two-bin encoder is reading a different pattern of descending
activity than the retinotopic one produces, so they cannot be mixed.

The connectome is identical in all of them — 166,700 neurons, 25,582,938
synapses, never trained. What changes is only how the game is written onto the
fly's own detectors, and how its spikes are read back.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))

VERSIONS = [
    {
        "id": "v1",
        "label": "v1 · hand-coded",
        "encoder": "bins",
        "readout": None,
        "headline": "2 neurons per side",
        "barrel_gate": "none",
        "note": "Steering read from the two DNa02 cells, fleeing from the two "
                "DNp01 giant fibers. Everything else — walking, the trigger, "
                "weapon choice — is a human rule. Escapes by reversing along "
                "its body axis. One number per eye, blind behind ±90°.",
        "score": "75.5 s · 33.2 kills · 18.7% hit",
    },
    {
        "id": "v2",
        "label": "v2 · trained readout",
        "encoder": "bins",
        "readout": "readout.npz",
        "headline": "ridge over all 1,314 DNs",
        "barrel_gate": "none",
        "note": "Same encoder, but aim and flee come from a ridge regression "
                "over the spike traces of every descending neuron instead of "
                "two cells. Fitted off-policy, which is why it plays worse "
                "than it measures.",
        "score": "64.2 s · 39.2 kills · 22.5% hit",
    },
    {
        "id": "v3",
        "label": "v3 · + DAgger",
        "encoder": "bins",
        "readout": "readout_dagger.npz",
        "headline": "retrained on its own states",
        "barrel_gate": "none",
        "note": "Same readout, refitted after letting it drive and collecting "
                "what it actually sees. Offline it barely moved; in the game it "
                "drew level with the hand-written script.",
        "score": "82.1 s · 54.3 kills · 27.5% hit",
    },
    {
        "id": "v4",
        "label": "v4 · retinotopic + dodge",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "sees direction, dodges, checks its line of fire",
        "note": "Each zombie now lands on the detectors whose place in the "
                "lobula matches its bearing, including behind. Two new trained "
                "outputs: which way to run, and whether an explosive barrel is "
                "in the line of fire.",
        "score": "59.5 s · 34.2 kills · 21.2% hit",
        "barrel_gate": "trained",
        "escape_mode": "trained",
    },
    {
        "id": "v5",
        "label": "v5 · dodge + hand barrel rule",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "same brain, barrel check done by geometry",
        "note": "Identical to v4 except the trigger. The trained barrel "
                "detector reached AUC 0.608 — near chance — and v4 took about "
                "104 HP of blast damage a run, which is what killed it. Here "
                "the check is three lines of geometry, clearly labelled as a "
                "human rule, not the fly.",
        "score": "73.7 s · 41.2 kills · 27.6% hit",
        "barrel_gate": "hand",
        "escape_mode": "trained",
    },
    {
        "id": "v6",
        "label": "v6 · retinotopic, hand escape",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "new eyes, old legs",
        "note": "Aim and flee still come from the readout over 1,314 DNs on the "
                "retinotopic encoder, but escaping goes back to the v3 rule — "
                "reverse away from the thing it is facing, slide along walls. "
                "The trained escape heading was 69.7° off on held-out data, and "
                "this isolates whether that is what cost v4 and v5 their runs.",
        "score": "90.0 s · 39.3 kills · 38.1% hit · survived all 12",
        "barrel_gate": "hand",
        "escape_mode": "hand",
    },
]


def available():
    out = []
    for v in VERSIONS:
        if v["readout"] and not os.path.exists(os.path.join(HERE, v["readout"])):
            continue
        out.append(v)
    return out


def get(vid):
    for v in available():
        if v["id"] == vid:
            return v
    av = available()
    return av[-1] if av else VERSIONS[0]
