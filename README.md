# curation_eval

An [Inspect](https://inspect.aisi.org.uk/) eval that reproduces a real
LLM-curation experiment — one that **failed in production** — and rebuilds it with
the two things the original hand-rolled version did not have: a **parameter sweep**
and **uncertainty**.

The experiment is not synthetic. It comes from a sibling project,
[`horror_readaloud`](#where-the-data-comes-from), a self-hosted read-aloud fiction
library where an LLM curates short fiction against editable channel criteria and a
taste profile learned from the listener's ratings.

---

## The experiment

A taste profile says the listener dislikes `weird [subgenre]` — cosmic horror,
which in this candidate pool means Lovecraft. The curator is asked to select *k*
titles from a reputation-ranked pool of 31 candidates, **11 of which are cosmic
horror**. Does the stated dislike survive into the selection?

The original project measured this twice and got opposite answers:

| | batch size | result |
|---|---|---|
| **JOURNAL Entry 35** — the Phase-6 gate | 12 | **PASSED.** With the profile, Lovecraft picks went 7/12 → **0/12**, against a noise floor of 1 title measured by an A/A′ control (two no-profile runs against each other). |
| **JOURNAL Entry 37** — first production build | 40 | **CONTRADICTED IT.** The model took **11 of the 12 Lovecraft titles.** One of eleven selection notes mentioned the profile at all; the rest ranked on reputation — *"the definitive cosmic horror text."* |

The project's own diagnosis, which this eval takes as its hypothesis:

> *"Filling 40 slots means reaching far enough down the reputation ranking that a
> single strong dislike stops being decisive."* The profile still works at the top
> of the ranking; **it does not EXCLUDE at depth.**

The gate was **re-opened rather than explained away**, and the finding was recorded
even though it inverted a passing result. That is why this experiment is worth
reproducing: the interesting artifact is the failure, and it survived being
documented.

**A claim about how an effect varies with batch size is a claim about a curve.**
The original measured two points, in two sessions a week apart, with **n=1 per
cell**. This repo makes it a sweep.

---

## What is here

```
curation_eval/dataset.py   builds one sample per (profile on/off) x (slot count)
curation_eval/scorer.py    cosmic_share + well_formed
curation_eval/task.py      the Inspect task
scripts/extract_pool.py    derives the dataset from the recorded production run
scripts/replay_baselines.py  $0 deterministic reference curves
tests/                     11 tests, incl. the historical ground-truth anchor
```

### Run it

```bash
pip install -e ".[dev]"

pytest                                      # 11 tests, no network
python scripts/replay_baselines.py          # $0, no API key, no spend
inspect eval curation_eval/task.py --model mockllm/model     # $0 plumbing check

# live — costs money, needs ANTHROPIC_API_KEY
inspect eval curation_eval/task.py --model anthropic/claude-sonnet-5 --epochs 5
```

---

## Status: the harness is built and validated. **The live sweep has not been run.**

Stated plainly because a repo that implies it has measured something it has not is
worse than an empty one. Every number below this line is either **deterministic**
or **historical**. There is no model measurement in this repo yet — running the
live sweep costs API spend that has not been authorised.

What *is* established:

**1. The scorer is anchored to a real historical outcome.** Replaying the actual 31
titles the production run selected scores **11/31 = 0.355**, matching JOURNAL Entry
37 exactly (`tests/test_scorer.py::test_recorded_run_reproduces_entry_37`). The
dataset extraction refuses to write at all unless the recorded run still matches
the journal it cites — 11 Lovecraft titles, 28 gutenberg / 3 creepypasta.

**2. The deterministic baselines are computed, and they already changed the
experimental design.** From `scripts/replay_baselines.py`:

| k | arithmetic floor | oracle | reputation-only | headroom |
|---:|---:|---:|---:|---:|
| 4 | 0.000 | 0.000 | 0.000 | **0.000** |
| 8 | 0.000 | 0.000 | 0.375 | 0.375 |
| 12 | 0.000 | 0.000 | 0.583 | **0.583** |
| 20 | 0.000 | 0.000 | 0.550 | 0.550 |
| 28 | 0.286 | 0.286 | 0.393 | 0.107 |

Two things fall out of that table that were not visible before it existed:

- **At k=4 the headroom is zero.** The control condition already selects no cosmic
  horror, so the profile *cannot* demonstrate an effect there however well it
  works. A null result at k=4 would be uninformative **by construction** — and
  would have looked like evidence.
- **The control curve is not monotonic in k.** Cosmic titles sit in the *middle* of
  the reputation ranking, so the no-profile share rises (0.375 → 0.583) then falls
  (0.550 → 0.393). Any effect size measured as "treatment minus control" therefore
  depends on k in a way a two-point experiment cannot see at all.

**3. The arithmetic floor is reported next to every score.** The pool has 11 cosmic
titles out of 31, so at k=28 at least 8 of 28 (28.6%) *must* be cosmic no matter
what the model does. `excess_over_floor` is the part the model is responsible for.
Reporting the raw share alone at high k would manufacture a failure out of
combinatorics.

---

## What this reproduction can and cannot show

**Cannot.** It is not a like-for-like replay of Entry 37. That run selected 31 from
a ~120-title shortlist, and **the shortlist was never recorded** — only the output.
So the candidate universe here is the 31 selected titles, not the 120 offered. The
*mechanism* under test survives (absolute demand against finite supply), but the
supply is smaller and richer in cosmic horror than the original, which is why the
headroom collapses by k=28. **The honest version of Entry 37's claim is therefore
weaker than Entry 37's**: some of the "effect disappears at scale" is the pool
running out of alternatives.

**Can.** Whether the effect is monotonic in k, where it peaks, whether it survives
repeated draws, and whether the model returns well-formed selections at all — none
of which the original could answer.

**A note on what "cosmic horror" means here.** The label is author-based
(`data/labels.json`, generated by `scripts/extract_pool.py`), which is crude but
auditable: you can disagree with a specific title rather than with a regex. It also
means the eval measures *"did it avoid Lovecraft"*, not *"did it avoid weird
fiction"*. Those come apart, and the second is the thing the profile actually says.

**And the profile itself is wrong.** Recorded in the source project and worth
repeating: the listener **likes** cosmic horror. The `weird 1.0/5` score came from
one badly-made story, not from the genre — a single rating became a verdict on a
whole subgenre because nothing floored *n* per tag. So the model's reputation
ranking was closer to right than the profile was, and a *stronger* preference
mechanism at batch 40 would have shipped a worse product. This eval measures
adherence to a stated preference; it does not measure whether the preference was
worth adhering to.

---

## Where the data comes from

`data/` is derived by `scripts/extract_pool.py` from `curation_runs` row id=5 in
`horror_readaloud` — 2026-07-28, `claude-sonnet-5`, $0.0512, written up as JOURNAL
Entry 37.

**Content and rights.** No story text or audio is in this repo, and none is
redistributed. What is here is metadata: titles, authors, public Project Gutenberg
catalogue references, the model's own selection notes, and the taste-profile string.
The source database is gitignored in its own project and is not copied here. The
underlying fiction is public domain (Gutenberg) or author-owned and merely
*referenced* (creepypasta); the eval reads neither.

---

## Why this repo exists

It closes a specific, named gap. An external skill diagnostic of this codebase's
author found strong LLM-pipeline and process-governance work sitting on top of a
measurement layer that was **descriptive rates only — no confidence intervals, no
inter-rater reliability, no significance** — and **zero exposure to standard eval
tooling**, so there was no way to know what a hand-rolled harness was missing.

`COMPARISON.md` is the answer to the second half: what Inspect does that the
hand-rolled version didn't, and — the part that is usually left out — what it
can't express.

The first half was closed separately, in the pipeline where the numbers actually
go to a supervisor: intervals were retrofitted onto every sampled rate there,
which turned up that the escalation-rate confidence interval **breaches the
approved budget cap** — a fact invisible to the point estimate that had been
reported for a week.

## License

MIT — see [LICENSE](LICENSE). Covers this source only; it grants nothing in the
fiction the underlying pipeline fetches.
