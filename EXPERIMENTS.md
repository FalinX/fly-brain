# Lab notebook

One entry per version, in the order they were built, with what changed, what
was expected, what was measured, and what it actually meant. Raw per-seed runs
for every row are in `results/*.json`.

**Protocol, identical for every version:** 12 arenas, 90 s each, solo, seeds
`7, 11, 23, 41, 57, 63, 88, 101, 119, 144, 177, 203`, unlimited ammo, one brain
step per 33.3 ms game frame. Offline metrics are held out by *seed*, never by
frame — frames 33 ms apart are near-copies and splitting on them inflates
everything.

The connectome is byte-identical in every version: MaleCNS v1.0, 166,700
neurons, 25,582,938 synapses, zero trained weights. What changes is the
**encoder** (game → voltage on the fly's own detectors) and the **decoder**
(descending spikes → keys).

---

## Summary

| | encoder | readout | escape | barrel check | survived (s) | kills | hit rate | full runs |
|---|---|---|---|---|---|---|---|---|
| v1 | 1 value/eye | 4 named cells | reverse | none | 75.5 ± 18.6 | 33.2 ± 9.8 | 18.7% | 5/12 |
| v2 | 1 value/eye | ridge, 1,314 DNs | reverse | none | 64.2 ± 27.9 | 39.2 ± 20.2 | 22.5% | 4/12 |
| v3 | 1 value/eye | + DAgger round 1 | reverse | none | 82.1 ± 12.2 | 54.3 ± 9.6 | 27.5% | 8/12 |
| v4 | retinotopic | + escape + barrel | trained | trained | 59.5 ± 20.7 | 34.2 ± 14.5 | 21.2% | 1/12 |
| v5 | retinotopic | + escape + barrel | trained | fixed cone | 73.7 ± 19.4 | 41.2 ± 13.3 | 27.6% | 4/12 |
| v6 | retinotopic | aim + flee | reverse | fixed cone | 90.0 ± 0.0 | 39.3 ± 5.3 | 38.1% | 12/12 |
| **v7** ⭐ | retinotopic | aim + flee | reverse | true geometry | 89.9 ± 0.2 | 43.8 ± 2.8 | 31.7% | 11/12 |
| v8 | retinotopic | + DAgger round 2 | reverse | true geometry | 86.7 ± 11.1 | 41.0 ± 8.6 | 29.1% | 11/12 |
| v9 | retinotopic | 2 copies voting | reverse | true geometry | 90.0 ± 0.0 | 43.0 ± 4.6 | 31.8% | 12/12 |
| _script_ | — | — | — | — | 77.0 ± 20.7 | 55.8 ± 16.6 | 58.1% | 7/12 |

---

## The survival experiment — the question the project is actually asking

*If zombies overran the world, how long would a fly with a gun last?*

Every benchmark above stops at 90 s, and from v6 onward the fly hits that
ceiling in almost every arena — so all of those survival figures are censored
and this question had never been measured. Nothing in the game heals the
player, so death is certain; only the timing was open.

`survive.py` removes the limit. 24 arenas (the standard twelve plus a second
block added because the headline claim sat at t = 2.25 on twelve alone), waves
escalating indefinitely: wave *N* spawns 5 + 2.6*N* zombies, devils appear from
wave 3 and reach 45% of spawns, zombie health rises 3 per wave.

| | survived | median | wave | kills | hit rate | died to own barrels |
|---|---|---|---|---|---|---|
| **v7 connectome** | **156.5 ± 40.9 s** | 147.8 | 6.9 | 88.5 | 41.1% | 0/12 |
| script + v7's barrel check | 104.4 ± 29.5 s | 105.2 | 5.5 | 60.4 | 66.2% | 0/12 |
| script, as originally written | 101.4 ± 42.4 s | 107.1 | 6.2 | 78.1 | 61.1% | 4/12 |
| v1, raw connectome, 4 cells | 83.4 ± 39.9 s | 80.4 | 4.1 | 37.5 | 18.6% | 10/12 |

Paired, v7 against each baseline over identical arenas:

| | difference | t | longer in |
|---|---|---|---|
| vs script + barrel check (n=24) | **+52.1 s** | **+4.81** | 19/24 |
| vs script as written (n=12) | +47.3 s | +2.84 | 11/12 |
| vs v1 (n=12) | +65.3 s | +3.17 | 10/12 |

**Answer: about two and a half minutes, wave 7, 88 zombies — half again as
long as a hand-written bot, with two thirds of its accuracy.** Best single run
253 s and wave 10.

*Fairness note:* the script as originally written has no barrel rule and kills
itself in 4 of 12 runs, which flatters the fly. `script+barrel` gives the
baseline the identical line-of-fire check and is the comparison quoted above.
It still loses.

**How it dies.** Per-wave health, logged in `results/survival_v7.json`:

| entering wave | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| seed 7 | 100 | 97 | 97 | 97 | 81 | 41 | — | — |
| seed 11 | 100 | 100 | 95 | 81 | 71 | 67 | 52 | 35 |
| seed 23 | 100 | 100 | 97 | 80 | 64 | 59 | 36 | 10 |

Untouched for three waves, then a steady bleed, and the damage is zombie
contact (74–103 HP a run) rather than barrels (0 in 12 of 12). Wave 6 puts
about twenty zombies on the map from every edge, and the escape rule — reverse
away from whatever it is facing — is perfect against one attacker and useless
against a ring of them.

That is the open target: the escape direction, which is also the one trained
output that measured badly (69.7° error, chance is 90°).

---

## v0 — what the descending population can even say

Before wiring anything to a key, 3,600 frames were logged with the fly watching
while a script drove (`diag.py`). Of the six descending channels that ship as
named groups inside `brain.npz`, only two carry the encoder's signal:

| channel | cells | measurement | usable |
|---|---|---|---|
| DNa02 steering | 1/side | corr(chase_L, steer_L) **+0.59**, cross term −0.31; sign matches the target's side on **64.8%** of informative frames | yes |
| DNp01 escape | 1/side | 2.98 Hz idle → 8.06 Hz with one zombie inside 220 px → 14.21 Hz with three | yes |
| DNg100 forward | 1/side | silent on 97.2% of frames, \|corr\| < 0.05 | no |
| DNg11 punch | 3/side | \|corr\| < 0.05 against every encoder channel | no |
| pIP10 kick | 1/side | \|corr\| < 0.07 | no |
| MDN backward | 2/side | \|corr\| < 0.05 | no |

64.8% side-correct with no training at all is the same figure fly.ai reported
for chase in SSH Fighter, reproduced in a different game.

**A separate check, `report/eyetest.py`:** the photoreceptor route is a dead
end. Drive enters fine (0.64 → 12.96 → 21.50 Hz on the 6,006 receptors as
drive rises) but nothing survives the crossing — an object anywhere in the
visual field moves LC4, LPLC2 and LC10a by less than ±0.1 Hz and DNp01 stays at
exactly 0.00 Hz. There are **zero** direct synapses from a photoreceptor to any
of the four detector types; the signal has to cross 26,610 optic-lobe neurons
and is absorbed. Every version therefore injects straight into the detectors,
and 95,501 optic-lobe neurons sit near-silent throughout.

---

## v1 — hand-coded, two neurons per side

**Config** · bins encoder · no readout · reverse escape · no barrel check

**What it is.** Turn rate from the DNa02 difference, smoothed; a Schmitt
trigger on DNp01 for fleeing; walking, the trigger and weapon choice are human
rules. The encoder gives one number per eye and drops anything more than 90°
off the nose.

**Result** · 75.5 ± 18.6 s · 33.2 ± 9.8 kills · 18.7% hit

**Read.** It plays. Survival is already within noise of the script baseline
(paired difference −1.5 s, t = −0.17) purely because DNp01 is a genuinely good
graded threat detector. It cannot aim, because DNa02 is right only 65% of the
time and one neuron per side inside a 300 ms window is a very coarse signal.

---

## v2 — ridge readout over all 1,314 descending neurons

**Config** · bins encoder · `readout.npz` · reverse escape · no barrel check

**Hypothesis.** Two cells is throwing away 1,312 others. A linear readout over
the leaky spike traces of the whole descending population should aim better.

**Offline** · side-correct 66.6% → **78.5%** · flee AUC 0.945 → 0.978

**Result** · 64.2 ± 27.9 s · 39.2 ± 20.2 kills · 22.5% hit

**Read.** Offline gain was real and large; in the game it gained 6 kills and
**lost 11 s of survival**. Latency is not the cause — 9.3 ms of a 33.3 ms
budget. The cause is distribution shift: the readout was fitted on states the
script produced, and once it steers it visits states it never saw.

*Aside worth keeping:* the first attempt at this measurement was wrong. Data
was collected with the script driving, which keeps the target centred, so 75%
of frames had a bearing inside ±0.25 rad and "predict zero" scored a 4.1°
median error while learning nothing. The collector was changed to spend half
its time wandering, and a tie in the two DNa02 cells was scored as half a point
rather than a loss. Only then were the two readouts comparable.

---

## v3 — one DAgger round

**Config** · bins encoder · `readout_dagger.npz` · reverse escape · no barrel check

**Hypothesis.** If distribution shift is the problem, letting the readout drive
and refitting on its own states should fix it.

**Offline** · side-correct 80.2% → **80.0%**, i.e. nothing · escape err 55.1° → 49.3° · flee AUC 0.975 → 0.978

**Result** · 82.1 ± 12.2 s · **54.3 ± 9.6** kills · 27.5% hit

**Read.** The single largest jump in the project, and offline metrics predicted
none of it. Kills 39.2 → 54.3. Paired against the script over the same twelve
arenas: survival **+5.1 s (t = +0.80)**, kills **−1.4 (t = −0.29)** — the
connectome draws with the hand-written baseline.

**Lesson.** Offline metrics computed on off-policy data are nearly worthless
here. The only offline number that moved in the right direction was the one the
game actually cares about, mean angular error.

---

## v4 — retinotopic encoder, trained escape, trained barrel gate

**Config** · retino encoder · `readout_dodge.npz` · trained escape · trained barrel gate

**Hypothesis.** Three at once, which in hindsight was one too many.

1. Soma positions inside each LC type lie on a 1-D sheet — PC1 carries 70–89%
   of the variance (LC4 88%, LPLC2 89%, LC10a 70%) over a 6–15 µm spread. That
   is the population's place in the retinotopically organised lobula. Collapsing
   it to one number per eye throws away all direction. v4 instead lands each
   zombie as a Gaussian bump (σ = 0.50 rad) on the detectors whose place matches
   its bearing, covering 13°–167° each side, **including behind**.
2. A trained escape heading, because a real fly escape is aimed away from the
   looming stimulus, not simply reverse.
3. A trained "is a barrel in my line of fire", because it kept shooting them.

*Caveat stated plainly:* a soma is not a receptive-field centre and the sheet
is oblique to the body axes, so this is an approximation of retinotopy, not a
measured map. The sign is anchored so that increasing value = more posterior.

**Offline** · aim side 80.0% → **84.1%** · flee AUC 0.952 · escape heading 69.7° mean error (chance 90°) · **barrel AUC 0.608**

**Result** · 59.5 ± 20.7 s · 34.2 ± 14.5 kills · 21.2% hit — the worst version built

**Read.** The clearest single number in the project: **10 of 12 runs took
exactly 104 HP of blast damage** — four barrels × 26 HP — **with 0 contact
damage**. The zombies were not killing it. It was blowing itself up, 18–38 own
bullets into barrels per run.

The barrel detector failing is not a bug to fix. LPLC1 answers *small object
moving in*; a barrel sits still. There is no circuit in a fly for "explosive
barrel", because no fly has ever needed one. AUC 0.608 is what asking an
animal a question it has no machinery for looks like.

---

## v5 — same, with the barrel check done by geometry

**Config** · retino · `readout_dodge.npz` · trained escape · fixed-cone barrel gate

**Change.** Replace the trained gate with three lines of geometry, labelled
`HAND` in the HUD: hold fire if a barrel is within ±0.30 rad and 420 px.

**Result** · 73.7 ± 19.4 s · 41.2 ± 13.3 kills · 27.6% hit

**Read.** +14 s and +7 kills over v4, confirming the barrel gate was the
primary cause of death. Still below v3, and barrel hits only halved (14–29 per
run) rather than stopping — so something else was still wrong.

---

## v6 — new eyes, old legs

**Config** · retino · `readout_dodge.npz` · **reverse escape** · fixed-cone gate

**Hypothesis.** Isolate the remaining suspect. The trained escape heading is
69.7° off; v3's rule (reverse away from what it is facing, slide along walls)
is crude but reliable, because the fly turns to face the threat, so "backward"
*is* "away". Put the old rule back and keep everything else.

**Result** · **90.0 ± 0.0 s — survived every single run** · 39.3 ± 5.3 kills · **38.1%** hit

**Read.** Hypothesis confirmed. The retinotopic encoder was never the problem;
the trained escape heading was. Hit rate 27.5% → 38.1%, barrel damage 0 in 11
of 12 runs, and it is the first version to finish all twelve arenas — against
8/12 for v3 and 7/12 for the script.

It kills less (39.3 vs v3's 54.3), because the barrel gate keeps it silent.

---

## v7 ⭐ — ask the barrel question correctly

**Config** · retino · `readout_dodge.npz` · reverse escape · **true-geometry gate**

**Hypothesis.** The cone was the wrong question. At 420 px, ±0.30 rad allows a
barrel 124 px off the line of fire — for a barrel 22 px wide, with seven of
them on a 960×640 map. Ask instead whether a bullet would arrive:

```python
perp      = dist * abs(sin(rel))                      # offset from the line
clearance = 11 + dist * WEAPONS[weapon].spread + 6    # barrel radius + this gun's cone
blocked   = perp < clearance
```

Weapon spread matters: pistol 0.010 rad, uzi 0.055, shotgun 0.200.

**Result** · 89.9 ± 0.2 s · **43.8 ± 2.8** kills · 31.7% hit · barrel damage 0 in 9/12

**Read.** +4.5 kills at unchanged survival. The number worth noticing is the
spread: **± 2.8 kills against the script's ± 16.6**. The connectome is now far
more consistent than the hand-written baseline, just lower-scoring. Hit rate
drops 38.1% → 31.7% because it now takes the harder shots it used to skip —
kills is the metric, not the ratio.

---

## v8 — DAgger round 2

**Config** · retino · `readout_v8.npz` · reverse escape · true-geometry gate

**Hypothesis.** The readout v6 and v7 run was fitted while the pilot still
steered with the trained escape heading, which both dropped — so v7 visits
states its own readout never saw. That is the same distribution shift that
round 1 fixed. `learn2.py`: v7 drives, label what it produces, keep half the
data script-and-wander for coverage, refit on 18,181 frames.

**Offline** · aim side 84.1% → 84.7% · aim err 48.5° → 48.5° · escape 69.7° → 69.6° · flee AUC 0.952 → 0.933 · barrel AUC 0.608 → 0.609

**Result** · 86.7 ± 11.1 s · 41.0 ± 8.6 kills · 29.1% hit

Paired against v7: survival **−3.3 s (t = −0.98)**, kills **−2.8 (t = −1.08)**.
Excluding one blow-up seed the two are identical, **43.3 vs 43.7 kills**.

**Read.** Round 1 was worth fifteen kills a run. Round 2 was worth nothing.
The curve saturated. Kept in the repo as a negative result; `versions.py`
defaults to v7, the best measured, not the newest.

One incidental observation: with v7 driving, the collector recorded **0 deaths**
across six 60-second runs, where every earlier round saw four or five.

---

## v9 — two flies voting

**Config** · retino · `readout_dodge.npz` · reverse escape · true-geometry gate · **2 copies**

**Hypothesis.** Same wiring, same encoder, same readout; only the noise stream
differs. Whatever the connectome is really computing is shared by both copies,
the noise is not, so averaging should cancel it. fly.ai measured exactly this
on SSH Fighter: moves toward the opponent went 63% with one fly to 72% with
eight voting. Two copies is what a 33.3 ms frame affords now that v7 costs
about 10 ms.

Implementation note: the encoder is deterministic given the observation, so it
runs once and both copies are handed the same drive; only `brain_step` and
`decode` are duplicated. `FlyPilot.clone()` shares the weight arrays by
reference — N copies cost N times the compute but one copy of the 205 MB
matrix.

**Result** · 90.0 ± 0.0 s (12/12) · 43.0 ± 4.6 kills · 31.8% hit · **23.0 ms/frame**

Paired against v7: survival **+0.08 s (t = +1.00)**, kills **−0.75 (t = −0.48)**,
hit rate **+0.09 pts (t = +0.04)**. Nothing, at 2.3× the compute.

**Read, and this is the interesting part.** The copies are *not* agreeing and
quietly doing nothing — measured over a 60 s run they genuinely disagree:

| | |
|---|---|
| turn sign differs | **26.7%** of frames |
| mean \|turn_A − turn_B\| | 1.30 rad/s (turn is clipped at ±6.5) |
| flee decision differs | 11.5% of frames |

So there is real noise-driven variance in the output, averaging really does
cancel it, and **cancelling it changes nothing**. What limits this pilot is not
variance, it is bias — the encoder and the readout. You cannot average your way
out of a readout that points the right way 84% of the time.

Why fly.ai saw a gain and this does not: their 63% → 72% was on the raw
hand-coded chase readout, one neuron per side, where noise dominates the
decision. Here the readout already pools all 1,314 descending neurons, which is
itself a large amount of averaging. A second brain adds little on top of that.

Kept as a negative result. Worth revisiting only if a future version goes back
to reading few cells — v1 is where voting should actually pay.

---

## Running notes

* **Frame cost scales with competence.** 9–12 ms when it dies at wave 3, 16–21
  ms when it reaches wave 5, against a 33.3 ms budget — a pilot that survives
  has more zombies on screen and therefore more neurons spiking.
* **numba cannot cache `_propagate`** (it closes over large global arrays), so
  the first brain step of every process spends seconds compiling. Any ms figure
  read in the first few seconds of a run is that, not the steady state.
* **Encoder and readout are a pair.** A readout fitted on the two-bin encoder
  reads a different pattern of descending activity than the retinotopic one
  produces. `versions.py` keeps the pairings; do not mix by hand.
