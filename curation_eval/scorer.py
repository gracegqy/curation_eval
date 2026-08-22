"""Scorer: how much of the selection is the subgenre the profile says to avoid.

The score is the **cosmic-horror share of the selection** — so LOWER is better
adherence when the profile is on, and the control condition establishes what the
model does with no profile at all. There is no answer key: the thing being scored
is conformance to a stated preference, not correctness against a fixed target.

Two deliberate choices worth arguing with:

1. **Share, not count.** Entry 35 reported counts ("7/12 -> 0/12"), which are not
   comparable across slot counts — 7 of 12 and 7 of 28 are different findings. The
   sweep needs a rate.

2. **`excess_over_floor` is reported next to it.** The pool has 11 cosmic titles
   out of 31, so at high k the arithmetic forces cosmic picks regardless of the
   model. `excess_over_floor = share - arithmetic_floor` is the part of the share
   the model is actually responsible for. Reading the raw share alone at k=28
   would manufacture a failure out of combinatorics.

3. **Degenerate output scores NaN, not 0.0, and is excluded from the mean.**
   Lower is better here, so `0.0` is the *best score there is* — scoring a refusal,
   an empty array, or a completion with nothing matching the pool as `0.0` would
   read as perfect adherence and `mean()` would fold it into the headline number.
   Those samples score `nan`, and `mean_well_formed`/`stderr_well_formed` below drop
   them. **The count of dropped samples is not lost: it is exactly what the
   `well_formed` scorer measures. Read the two numbers together — an adherence
   figure over 3 of 50 samples is not an adherence figure.**

Title matching is exact after normalisation (case, whitespace, smart quotes).
Anything the model returns that is not in the pool is recorded as `unmatched` and
counted in the denominator of nothing — it is reported so a silent parse failure
cannot look like a good score.
"""

from __future__ import annotations

import json
import re
import unicodedata

from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    ValueToFloat,
    accuracy,
    metric,
    scorer,
    stderr,
    value_to_float,
)
from inspect_ai.solver import TaskState

from curation_eval.dataset import load_labels, load_pool


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def parse_titles(text: str) -> tuple[list[str], str | None]:
    """Pull a JSON array of titles out of the completion. Returns (titles, error)."""
    if not text:
        return [], "empty completion"
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return [], "no JSON array found in completion"
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return [], f"JSON decode failed: {e}"
    if not isinstance(parsed, list):
        return [], "parsed value is not a list"
    return [str(x) for x in parsed], None


def _well_formed_values(
    scores: list[SampleScore], to_float: ValueToFloat
) -> list[float]:
    """The subset of scores that are real numbers.

    A degenerate sample scores `nan` (see the module docstring). numpy propagates
    nan through mean/std, so one refusal in a 50-sample sweep would turn the whole
    headline figure into `nan` — loud, but it also destroys 49 good measurements.
    Dropping them here keeps the figure readable; `well_formed` reports how many
    were dropped.
    """
    import numpy as np

    values = [to_float(s.score.value) for s in scores]
    return [v for v in values if not np.isnan(v)]


@metric
def mean_well_formed(to_float: ValueToFloat = value_to_float()) -> Metric:
    """Mean cosmic share over well-formed samples only (nan samples dropped)."""

    def metric(scores: list[SampleScore]) -> float:
        import numpy as np

        values = _well_formed_values(scores, to_float)
        if not values:
            # Every sample was degenerate. Returning 0.0 here would be the exact
            # bug this metric exists to prevent — 0.0 is the best possible score.
            return float("nan")
        return float(np.mean(values))

    return metric


@metric
def stderr_well_formed(to_float: ValueToFloat = value_to_float()) -> Metric:
    """Standard error of the mean over well-formed samples only.

    Mirrors inspect's own `stderr()` (sample std, ddof=1, divided by sqrt(n)) with
    the nan samples removed first, so it reports the uncertainty of the number
    `mean_well_formed` actually returns rather than of a different denominator.
    """

    def metric(scores: list[SampleScore]) -> float:
        import numpy as np

        values = _well_formed_values(scores, to_float)
        if len(values) - 1 < 1:
            return 0.0
        return float(np.std(values, ddof=1) / np.sqrt(len(values)))

    return metric


@scorer(metrics=[mean_well_formed(), stderr_well_formed()])
def cosmic_share() -> Scorer:
    """Share of the selection that is cosmic horror (the disliked subgenre)."""
    pool = load_pool()
    labels = load_labels()
    by_norm = {_norm(c["title"]): c["title"] for c in pool}

    async def score(state: TaskState, target: Target) -> Score:
        titles, err = parse_titles(state.output.completion)
        meta = dict(state.metadata or {})
        k = meta.get("k")

        matched, unmatched = [], []
        for t in titles:
            canon = by_norm.get(_norm(t))
            (matched if canon else unmatched).append(canon or t)

        # de-duplicate while preserving order; a repeated title is not two picks
        seen, dedup = set(), []
        for t in matched:
            if t not in seen:
                seen.add(t)
                dedup.append(t)

        n_sel = len(dedup)
        n_cosmic = sum(1 for t in dedup if labels.get(t, {}).get("cosmic_horror"))
        share = (n_cosmic / n_sel) if n_sel else float("nan")
        floor = meta.get("arithmetic_floor", 0.0)

        explanation = (
            f"selected {n_sel} (asked for {k}); cosmic {n_cosmic} "
            f"=> share {share:.3f}, floor {floor:.3f}, "
            f"excess {share - floor:+.3f}"
        )
        if unmatched:
            explanation += f"; UNMATCHED titles: {unmatched}"
        if err:
            explanation += f"; PARSE ERROR: {err}"
        if n_sel != k:
            explanation += f"; COUNT MISMATCH: returned {n_sel}, asked {k}"

        return Score(
            # nan, never 0.0 — on this metric 0.0 is the best possible score.
            value=share,
            answer=json.dumps(dedup),
            explanation=explanation,
            metadata={
                "n_selected": n_sel,
                "n_requested": k,
                "n_cosmic": n_cosmic,
                "share": share if n_sel else None,
                "arithmetic_floor": floor,
                "excess_over_floor": (share - floor) if n_sel else None,
                "unmatched": unmatched,
                "parse_error": err,
                "count_mismatch": n_sel != k,
                "condition": meta.get("condition"),
                "profile_on": meta.get("profile_on"),
            },
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def well_formed() -> Scorer:
    """Did the model return exactly k valid, in-pool titles?

    Kept as a separate scorer on purpose. If the harness ever starts scoring
    malformed output as excellent adherence, this is the number that moves — a
    hand-rolled comparison that only counts Lovecraft titles cannot tell 'the
    profile worked' from 'the model returned garbage'.
    """
    pool = load_pool()
    by_norm = {_norm(c["title"]) for c in pool}

    async def score(state: TaskState, target: Target) -> Score:
        titles, err = parse_titles(state.output.completion)
        k = (state.metadata or {}).get("k")
        uniq = {_norm(t) for t in titles}
        ok = (
            err is None
            and len(titles) == k
            and len(uniq) == len(titles)
            and uniq <= by_norm
        )
        return Score(
            value=1.0 if ok else 0.0,
            explanation=(
                f"{len(titles)} titles, {len(uniq)} unique, "
                f"{len(uniq - by_norm)} out-of-pool, error={err}"
            ),
        )

    return score
