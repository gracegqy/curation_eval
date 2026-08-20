# Inspect vs. the hand-rolled version

One `infinite_audiobook`-style scoring task reproduced in an established
framework, then an honest account of **what the framework does that the hand-rolled
version didn't, and what it can't express.**

The hand-rolled version is the A/A′/B comparison in `infinite_audiobook` JOURNAL
Entries 34/35/37 — a sandboxed script, run twice, a week apart, producing a table
of three numbers in a journal entry. The Inspect version is this repo.

Written after building both, so the comparison is from use rather than from docs.

---

## What Inspect gave me that I did not build myself

**1. Repetition is a flag, not a rewrite.** `--epochs 5` reruns every sample five
times and the metrics aggregate automatically. This is the single biggest gap in
the original: **n=1 per cell**. Entry 35's headline — "1 title of noise, 8 of
effect" — is three numbers with no spread, and the A/A′ control was doing the work
of an error bar with a sample size of one. In the hand-rolled version, adding
repeats meant editing the script, re-running by hand, and writing my own
aggregation. Here it is a command-line argument.

**2. `stderr()` and `bootstrap_stderr()` ship as metrics.** Uncertainty is a
one-line declaration next to the scorer, not a statistics project. The reason
years of my own measurement work stayed at descriptive rates with no confidence
intervals is partly that computing them was always *a separate task you had to
decide to do*. When it is a default in the metric list, the failure mode of
forgetting stops existing.

**3. The parameter sweep is the dataset, not a loop.** Slot count is sample
metadata, so "run this at five batch sizes" is a dataset comprehension. Entry 35
gated at batch 12; production ran at 40; **nobody ran 20 or 28, because each point
was a manual session.** The whole defect in Entry 37 is a claim about how an effect
varies with a parameter, discovered only when production picked a different value.
Sweeping was always the right design; the framework is what made it cheap enough to
be the obvious one.

**4. Structured logs, viewable per sample.** `inspect view` shows the exact prompt,
completion, score, and explanation for every sample. The hand-rolled equivalent was
a printed table plus whatever the journal entry remembered to record. Entry 37's
diagnosis needed the model's *selection notes* — the evidence that it was ranking
on reputation and never weighing the dislike — and recovering those meant reading a
database by hand.

**5. Multiple scorers, cheaply.** `well_formed` runs alongside `cosmic_share` for
one extra decorator. It earned itself immediately: the first mockllm run scored
`cosmic_share = 0.000` — *apparently perfect adherence* — while `well_formed`
correctly reported 0.000 because the output was garbage. **A hand-rolled scorer
that only counts Lovecraft titles cannot distinguish "the profile worked" from "the
model returned nothing parseable."** The original had exactly that shape.

**6. A model interface that makes $0 replay a first-class citizen.**
`custom_outputs` takes a callable, so the deterministic baselines run through the
*same* task, solver, and scorer as a live model. In the hand-rolled version the
baseline would have been a different code path — which is how baselines quietly
stop matching the thing they baseline.

---

## What Inspect did not do, and what it cannot express

**1. It has no opinion about whether the experiment is meaningful.** The framework
computed `cosmic_share` at k=4 and reported it as a number like any other. It took
a deterministic baseline run — my own code, not Inspect's — to discover that
**headroom at k=4 is zero** and the cell is uninformative by construction. Inspect
will happily give you a tight confidence interval on a quantity that cannot move.
*Uncertainty tooling does not protect you from measuring the wrong thing.*

**2. The arithmetic floor is mine.** Nothing in a scoring framework knows that a
pool of 31 with 11 cosmic titles forces ≥8 cosmic picks at k=28. `accuracy` and
`mean` are indifferent to the fact that part of the score is combinatorics. That
correction — `excess_over_floor` — is domain reasoning, and it is the difference
between a real result and a manufactured failure.

**3. Clustering is not in the metric set.** `stderr()` treats samples as
independent. In a separate extraction pipeline of mine, the same naive assumption
made an interval **too narrow by a design effect of 11** — observations were
nested inside documents. Inspect gives you `bootstrap_stderr`,
not a *cluster* bootstrap; if your samples share structure, the default is wrong
and nothing warns you. Here every sample is a separate prompt, so independence
holds — but that is a fact about my dataset, not about the framework.

**4. It cannot tell you the preference you are scoring against is wrong.** The
profile says `weird 1.0/5`; the listener actually *likes* cosmic horror, and the
score came from one badly-made story. A perfect adherence result would have
certified a defect. **The eval measures conformance to a stated target; whether
the target is right is upstream of every framework.** This is the single most
important limitation on the page, and it is not a criticism of Inspect — it is what
evals are *for* and what they are silently *not* for.

**5. It does not do experimental design.** Control conditions, the A/A′ noise
floor, which parameter to sweep, what the null policy should be — all mine. The
original hand-rolled version was *good at exactly this part*: the A/A′ control was
a genuinely well-designed noise floor, and it is what caught Entry 34's non-result.
The framework replaced the plumbing, not the thinking.

**6. Friction, honestly.** Inspect loads task files by path, so a package-relative
import fails until the project is installed (`pip install -e .`) — a real ten
minutes. The default data directory is `~/Library/Application Support`, which a
locked-down environment will refuse. Neither is hard; both are the kind of thing
that makes "just use the framework" more expensive than one afternoon.

---

## The one-sentence version

**Inspect removed the reasons not to run the experiment properly** — repetition,
sweeps, uncertainty, per-sample logs, a second scorer all became defaults instead
of decisions — **and it removed none of the reasons to get the experiment wrong.**
The hand-rolled version's fatal flaw was not that it lacked a framework; it was
that it measured one batch size and shipped. A framework makes measuring five
batch sizes cheap. It does not make you ask how many you should have measured.

## What I would do differently in the source project

1. **Make batch size a swept parameter in the gate itself.** The gate passed at the
   value the gate used and failed at the value production used. That is a gate
   testing its own configuration, not the system's.
2. **Add a well-formedness scorer to every gate.** Cheap, and it is the difference
   between a pass and an unnoticed parse failure.
3. **Report the achievable floor next to any rate over a constrained pool.**
4. **Never gate on n=1.** The A/A′ control was a real attempt at a noise floor and
   it deserved repeats — one run of each cell cannot estimate the variance it was
   built to estimate.
