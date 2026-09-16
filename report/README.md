# The report

A 15-page A4 landscape explainer, in Thai, of what the MaleCNS connectome is
and how the simulation works: **`fly-ai-explained-TH.pdf`**.

Every number in it was read out of `brain.npz` and `weights.npz` directly, and
the two experiments in it were run on the same machine that produced the game
benchmarks. Nothing is copied from a README.

## What is in it

| page | |
|---|---|
| cover | the whole brain rendered from real soma positions |
| 01 | one-page summary: it is a wiring diagram, not a trained model |
| 02 | where the data comes from, EM → a sparse matrix |
| 03 | **plate I** — all 166,700 neurons, three views |
| 04 | what the parts are: 57% of the brain is eye |
| 05 | one neuron, one step — leaky integrate-and-fire |
| 06 | what the wiring looks like: degree, weight, sign |
| 07 | the full architecture of one step |
| 08 | encoder and decoder — the only two things a human writes |
| 09 | the twelve neurons the hand-coded decoder reads |
| 10 | **plate II** — the escape and chase pathway in 3D |
| 11 | **the experiment**: stimulate one eye, watch the right cell answer |
| 12 | what it has been used for, wins and losses |
| 13 | limitations the project states about itself |
| 14 | how to run it yourself |

## The two experiments

**Page 11 — the pathway works.** Reimplements the leaky integrate-and-fire loop
with the published constants (`dt 0.020`, `tau 0.100`, `gain 3.0`, `tonic 0.14`,
eye drive 0.45), loads the real matrix, and stimulates LC4 + LPLC2 on the
**left** only:

| | DNp01 L | DNp01 R | DNa02 L | DNa02 R |
|---|---|---|---|---|
| rest | 0 Hz | 0 Hz | 0 Hz | 0 Hz |
| loom, left eye | **47 Hz** | 0 Hz | 1 Hz | 0 Hz |
| chase, left eye | 0 Hz | 0 Hz | **5 Hz** | 0 Hz |

Right pathway, right side, no crosstalk — and nobody told the network that LC4
should connect to DNp01. `simulate.py`, writes `sim.json`.

**`eyetest.py` — the photoreceptor route does not work.** Drives all 6,006
photoreceptors and measures how far the signal gets. It enters fine
(0.64 → 12.96 → 21.50 Hz as drive rises) and then dies crossing the optic lobe:
putting an object anywhere in the visual field moves LC4, LPLC2 and LC10a by
less than ±0.1 Hz, and DNp01 stays at exactly 0.00 Hz in every condition.
There are **zero** direct synapses from a photoreceptor to LC4, LPLC2, LPLC1 or
LC10a; the signal has to cross 26,610 optic-lobe neurons and is absorbed.

This is why the game injects straight into the detectors, and why 95,501 optic
lobe neurons sit near-silent in the live 3D view.

## Rebuilding it

```bash
python simulate.py     # the page-11 experiment -> sim.json
python eyetest.py      # the photoreceptor experiment, prints its table
python fig3d.py        # plates I and II
python figcover.py     # the cover render
python figcharts.py    # composition, LIF, response
python figdiagram.py   # architecture, decoder map, wiring stats
python topdf.py        # report.html -> PDF via headless Chromium
python checkfit.py     # warns if any page overflows
```

Needs `matplotlib`, `scipy`, `playwright` (`playwright install chromium`) and
`pypdfium2` if you want to rasterise pages to check them. The PDF picks up
Fraunces, Noto Serif Thai, IBM Plex Mono and IBM Plex Sans from Google Fonts at
render time, so it needs a network connection once.
