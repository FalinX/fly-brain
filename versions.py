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
    {
        "id": "v7",
        "label": "v7 · tighter barrel rule",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "only holds fire when a bullet would really hit a barrel",
        "note": "v6 held fire whenever a barrel sat within ±0.30 rad and "
                "420 px — up to 124 px off the line — and with seven barrels "
                "on the map it was silent most of the time. v7 asks whether a "
                "bullet would actually reach one: the barrel's 11 px radius "
                "plus the cone the current weapon throws at that range. Same "
                "brain, same readout, tighter question.",
        "score": "89.9 s · 43.8 kills · 31.7% hit · 11 of 12 full runs",
        "barrel_gate": "tight",
        "escape_mode": "hand",
    },
    {
        "id": "v8",
        "label": "v8 · DAgger round 2",
        "encoder": "retino",
        "readout": "readout_v8.npz",
        "headline": "refitted on the states v7 actually visits",
        "note": "The readout v6 and v7 run was fitted while the pilot was "
                "still steering with the trained escape heading, which they "
                "both threw away. This round let v7 drive and refitted on what "
                "it produces, half on-policy, half script and wander for "
                "coverage. Offline it is flat — aim side 84.1% → 84.7% — which "
                "is what a saturating DAgger curve looks like.",
        "score": "86.7 s · 41.0 kills · 29.1% hit — no better than v7",
        "barrel_gate": "tight",
        "escape_mode": "hand",
    },
    {
        "id": "v9",
        "label": "v9 · two flies voting",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "same wiring twice, different noise, averaged",
        "flies": 2,
        "note": "Two copies of v7 with their own noise streams. Whatever the "
                "connectome is really computing is shared by both; the noise "
                "is not, so averaging their readouts cancels it. fly.ai "
                "measured chase going 63% to 72% with eight copies voting. Two "
                "is what a 33.3 ms frame affords once v7 costs about 10 ms.",
        "score": "90.0 s · 43.0 kills · 31.8% hit — same as v7 at 2.3x the compute",
        "barrel_gate": "tight",
        "escape_mode": "hand",
    },
    {
        "id": "v10",
        "label": "v10 · escape retrained on crowds",
        "encoder": "retino",
        "readout": "readout_v10.npz",
        "headline": "uses the trained escape direction only when surrounded",
        "note": "The survival runs showed v7 dying to zombie contact from wave "
                "4 on, with an escape rule that reverses away from whatever it "
                "faces — perfect against one attacker, useless inside a ring. "
                "The trained heading that was supposed to fix that measured "
                "69.7° off, because a run started at wave 1 is only 12% "
                "crowded frames and the fit optimised the easy majority. "
                "Refitted with half the data collected from wave 5 and 7 "
                "starts and crowded frames oversampled to 50%: 63.3° → 54.2° "
                "when surrounded, unchanged on an open board. Used only when "
                "3+ zombies are inside 250 px; aim, flee and the barrel check "
                "are byte-identical to v7.",
        "score": "136.0 s survival — 20 s worse than v7, kept as a negative result",
        "barrel_gate": "tight",
        "escape_mode": "hybrid",
    },
    {
        "id": "v11",
        "label": "v11 · aim fitted flat over bearing",
        "encoder": "retino",
        "readout": "readout_v11.npz",
        "headline": "the resampling that worked for escape, applied to aim",
        "note": "v10's lesson was that readouts are fitted on data dominated by "
                "frames where the answer does not matter. Applying the same fix "
                "to aim — resample until every bearing band is equally "
                "represented — wrecks the only band that decides kills: error "
                "when the target is already near the centre goes 22.7° to "
                "59.6°. Kept as the control that shows the v10 lesson is not a "
                "general rule.",
        "score": "offline only — 59.6° in the band that matters, not run",
        "barrel_gate": "tight",
        "escape_mode": "hand",
    },
    {
        "id": "v12",
        "label": "v12 · aim weighted toward the centre",
        "encoder": "retino",
        "readout": "readout_v12.npz",
        "headline": "precision where a shot lands or misses",
        "note": "The opposite resampling. A shot only connects when the target "
                "is already near the centre, so the fit is weighted toward "
                "those frames instead of away from them. Error in the 0–0.15 "
                "rad band goes 22.7° to 5.5°, four times sharper, at the cost "
                "of bearings past 0.9 rad where the turn is clipped anyway. "
                "Side-correct is unchanged at 79.0%, so it still turns the "
                "right way. Everything except aim is byte-identical to v7.",
        "score": "measuring",
        "barrel_gate": "tight",
        "escape_mode": "hand",
    },
    {
        "id": "v13",
        "label": "v13 · kiter legs, fly eyes",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "survival-first: move by repulsion, aim with the connectome",
        "note": "The objective changed to staying alive. A hand-written kiter "
                "with no brain at all reaches 216 s against v7's 156.5, and "
                "that whole gap is movement — the fly outruns everything on "
                "the map at 185 px/s and was not using it. v13 keeps the "
                "connectome for aiming and hands movement to a repulsion rule, "
                "labelled HAND. DNp01 still decides the standoff distance. "
                "A kiter that never shoots dies at 114.9 s stuck on wave 1, so "
                "the aim is not decoration: killing is what stops the board "
                "filling up.",
        "score": "197.4 s survival - wave 7.9 - 110 kills - level with the hand-written ceiling",
        "barrel_gate": "tight",
        "escape_mode": "hand",
        "move_mode": "kite",
    },
    {
        "id": "v15a",
        "label": "v15a · standoff 273 px",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "v13 with the kite radius set to 273 px instead of 433",
        "note": "Lowering tonic to 0.10 bought about 15 s of survival, and the "
                "mechanism looked like a side effect on movement: a quieter "
                "DNp01 shrinks the standoff radius from a median of 433 px to "
                "347. This sets the radius directly at tonic 0.14, keeping the "
                "84% aim the lower tonic throws away. reach = 180 + 14 x "
                "DNp01 rate.",
        "score": "measuring",
        "barrel_gate": "tight",
        "escape_mode": "hand",
        "move_mode": "kite",
        "reach": (180.0, 14.0),
    },
    {
        "id": "v15b",
        "label": "v15b · standoff 353 px",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "v13 with the kite radius set to 353 px instead of 433",
        "note": "Lowering tonic to 0.10 bought about 15 s of survival, and the "
                "mechanism looked like a side effect on movement: a quieter "
                "DNp01 shrinks the standoff radius from a median of 433 px to "
                "347. This sets the radius directly at tonic 0.14, keeping the "
                "84% aim the lower tonic throws away. reach = 220 + 20 x "
                "DNp01 rate.",
        "score": "measuring",
        "barrel_gate": "tight",
        "escape_mode": "hand",
        "move_mode": "kite",
        "reach": (220.0, 20.0),
    },
    {
        "id": "v15c",
        "label": "v15c · standoff 520 px",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "v13 with the kite radius set to 520 px instead of 433",
        "note": "Lowering tonic to 0.10 bought about 15 s of survival, and the "
                "mechanism looked like a side effect on movement: a quieter "
                "DNp01 shrinks the standoff radius from a median of 433 px to "
                "347. This sets the radius directly at tonic 0.14, keeping the "
                "84% aim the lower tonic throws away. reach = 320 + 30 x "
                "DNp01 rate.",
        "score": "measuring",
        "barrel_gate": "tight",
        "escape_mode": "hand",
        "move_mode": "kite",
        "reach": (320.0, 30.0),
    },
    {
        "id": "v15d",
        "label": "v15d · standoff 207 px",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "tighter still, because the tight end was never explored",
        "note": "273 px beat 433 on every measure at once — survival, kills, "
                "accuracy and wave — so the sweep never found the tight edge. "
                "Standing closer means bullets travel less, which means less "
                "lead error, which means the board clears faster. This asks "
                "where that stops paying. reach = 140 + 10 x DNp01 rate.",
        "score": "measuring",
        "barrel_gate": "tight",
        "escape_mode": "hand",
        "move_mode": "kite",
        "reach": (140.0, 10.0),
    },
    {
        "id": "v15e",
        "label": "v15e · standoff 132 px",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "the radius the kiter sweep says is right",
        "note": "Sweeping the kiter's own standoff radius showed the ceiling "
                "was never 216 s - that was one untested value, 420 px. At 130 "
                "px the same rule survives 1,197 s and some runs hit the 1,800 "
                "s safety cap. Closer is better because killing clears the "
                "board while distance only delays, and bullets that travel "
                "less need no lead. This asks whether the fly can use the same "
                "radius given that its aim is worse and it kills slower. "
                "reach = 65 + 10 x DNp01 rate.",
        "score": "measuring",
        "barrel_gate": "tight",
        "escape_mode": "hand",
        "move_mode": "kite",
        "reach": (65.0, 10.0),
    },
    {
        "id": "v15f",
        "label": "v15f · standoff 162 px",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "the radius the kiter sweep says is right",
        "note": "Sweeping the kiter's own standoff radius showed the ceiling "
                "was never 216 s - that was one untested value, 420 px. At 130 "
                "px the same rule survives 1,197 s and some runs hit the 1,800 "
                "s safety cap. Closer is better because killing clears the "
                "board while distance only delays, and bullets that travel "
                "less need no lead. This asks whether the fly can use the same "
                "radius given that its aim is worse and it kills slower. "
                "reach = 95 + 10 x DNp01 rate.",
        "score": "measuring",
        "barrel_gate": "tight",
        "escape_mode": "hand",
        "move_mode": "kite",
        "reach": (95.0, 10.0),
    },
    {
        "id": "v15-blind",
        "label": "v15-blind · control, brain sees nothing",
        "encoder": "retino",
        "readout": "readout_dodge.npz",
        "headline": "identical to v15d except the encoder writes nothing",
        "note": "The question this answers is not whether the connectome is "
                "running - it always is - but whether it is deciding anything. "
                "Same wiring, same tonic, same noise, same eye baseline, same "
                "12 ms a frame, same readout reading the same 1,314 descending "
                "neurons. The only difference is that the encoder injects "
                "nothing about the game, so the readout is reading the brain "
                "talking to itself. Whatever v15d scores above this is what "
                "the connectome is worth.",
        "score": "measuring",
        "barrel_gate": "tight",
        "escape_mode": "hand",
        "move_mode": "kite",
        "reach": (180.0, 14.0),
        "blind": True,
    },
    {
        "id": "v14",
        "label": "v14 · smell, at tonic 0.10",
        "encoder": "retino",
        "readout": "readout_v14.npz",
        "headline": "olfaction opened up, and it had nothing to add",
        "tonic20": 0.10,
        "olfaction": True,
        "note": "The connectome has 2,635 olfactory receptors across 53 "
                "glomeruli and they are well wired — 28,447 synapses onto the "
                "antennal-lobe projection neurons, 28% of a PN's input budget. "
                "At the standard tonic of 0.14 the lobe sits pinned near its "
                "30 Hz ceiling and the descending population cannot tell "
                "left-smell from right-smell at all (AUC 0.494). Lowering "
                "tonic frees it (0.567 at 0.10, 0.671 at 0.07) but quietens "
                "everything else. This is the only point where both might "
                "work. Trained with and without the channel from separate "
                "collections, it changed nothing.",
        "score": "offline: aim 71.7% vs 84.1% at tonic 0.14; smell adds nothing",
        "barrel_gate": "tight",
        "escape_mode": "hand",
        "move_mode": "kite",
    },
    {
        "id": "v14n",
        "label": "v14n · tonic 0.10, no smell",
        "encoder": "retino",
        "readout": "readout_v14n.npz",
        "headline": "the control that isolates the tonic change",
        "tonic20": 0.10,
        "olfaction": False,
        "note": "Same operating point as v14 with the olfactory channel off, "
                "so the two together separate what lowering tonic did from "
                "what smell did. Smell did nothing; the tonic change cost 12 "
                "points of aim.",
        "score": "offline: aim 72.6%, rear AUC 0.741 — slightly better than with smell",
        "barrel_gate": "tight",
        "escape_mode": "hand",
        "move_mode": "kite",
    },
]

# What `server.py` and `play.py` start on when no version is named. Not the
# newest — the best measured against the current objective, staying alive.
# v8, v9, v10, v11 and v12 record what did not work; they are not for playing.
DEFAULT = "v13"


def available():
    out = []
    for v in VERSIONS:
        if v["readout"] and not os.path.exists(os.path.join(HERE, v["readout"])):
            continue
        out.append(v)
    return out


def get(vid):
    av = available()
    for v in av:
        if v["id"] == vid:
            return v
    for v in av:
        if v["id"] == DEFAULT:
            return v
    return av[-1] if av else VERSIONS[0]
