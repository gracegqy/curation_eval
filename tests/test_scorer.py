"""Scorer tests, anchored to the real recorded outcome.

The important one is `test_recorded_run_reproduces_entry_37`. The 2026-07-28
production run in infinite_audiobook selected all 31 of these titles, 11 of them
cosmic horror. If this scorer is correct, feeding it that exact selection must
produce 11/31 = 0.355 — the number JOURNAL Entry 37 reports. That is the only
ground truth available at $0, and it is a real one: it ties the harness to a
historical event rather than to its own assumptions.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from curation_eval.dataset import arithmetic_floor, build_dataset, load_labels, load_pool
from curation_eval.scorer import (
    cosmic_share,
    mean_well_formed,
    parse_titles,
    stderr_well_formed,
    well_formed,
)


def _state(completion: str, metadata: dict) -> TaskState:
    state = TaskState(
        model="test/test",
        sample_id="t",
        epoch=1,
        input="",
        messages=[],
        metadata=metadata,
    )
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput

    state.output = ModelOutput(
        model="test/test",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content=completion), stop_reason="stop"
            )
        ],
    )
    return state


def _score(completion: str, k: int, floor: float = 0.0) -> dict:
    sc = cosmic_share()
    meta = {"k": k, "arithmetic_floor": floor, "condition": "t", "profile_on": True}
    return asyncio.run(sc(_state(completion, meta), Target("")))


# --------------------------------------------------------------- ground truth


def test_recorded_run_reproduces_entry_37():
    """The recorded 31-title selection must score 11/31 cosmic."""
    pool = load_pool()
    labels = load_labels()
    titles = [c["title"] for c in pool]
    n_cosmic = sum(1 for t in titles if labels[t]["cosmic_horror"])

    assert len(titles) == 31, "dataset drifted from the recorded run"
    assert n_cosmic == 11, "Entry 37 reports 11 Lovecraft titles"

    res = _score(json.dumps(titles), k=31)
    assert res.metadata["n_selected"] == 31
    assert res.metadata["n_cosmic"] == 11
    assert res.value == pytest.approx(11 / 31, abs=1e-9)
    assert not res.metadata["unmatched"]
    assert res.metadata["parse_error"] is None


# ------------------------------------------------------------------ mechanics


def test_arithmetic_floor_matches_combinatorics():
    # 31 titles, 11 cosmic => 20 non-cosmic available
    assert arithmetic_floor(12, 11, 31) == 0.0
    assert arithmetic_floor(20, 11, 31) == 0.0
    assert arithmetic_floor(28, 11, 31) == pytest.approx(8 / 28)
    assert arithmetic_floor(31, 11, 31) == pytest.approx(11 / 31)


def test_perfect_adherence_is_zero_share():
    pool = load_pool()
    labels = load_labels()
    clean = [c["title"] for c in pool if not labels[c["title"]]["cosmic_horror"]][:12]
    res = _score(json.dumps(clean), k=12)
    assert res.value == 0.0
    assert res.metadata["n_cosmic"] == 0


def test_unmatched_titles_are_reported_not_swallowed():
    res = _score(json.dumps(["A Story That Does Not Exist"]), k=1)
    assert res.metadata["unmatched"] == ["A Story That Does Not Exist"]
    assert res.metadata["n_selected"] == 0


def test_duplicate_titles_count_once():
    pool = load_pool()
    t = pool[0]["title"]
    res = _score(json.dumps([t, t, t]), k=3)
    assert res.metadata["n_selected"] == 1
    assert res.metadata["count_mismatch"] is True


def test_parse_failure_is_flagged():
    res = _score("I would pick The Fall of the House of Usher.", k=1)
    assert res.metadata["parse_error"] is not None
    assert res.metadata["n_selected"] == 0


def test_prose_wrapped_json_still_parses():
    pool = load_pool()
    t = pool[0]["title"]
    res = _score(f'Here you go:\n{json.dumps([t])}\nHope that helps!', k=1)
    assert res.metadata["n_selected"] == 1
    assert res.metadata["parse_error"] is None


def test_parse_titles_edge_cases():
    assert parse_titles("") == ([], "empty completion")
    assert parse_titles("no array here")[1] is not None
    assert parse_titles("[broken,")[1] is not None
    assert parse_titles('["a","b"]') == (["a", "b"], None)


# ------------------------------------------------ degenerate output (the 8/17 gate)


def test_degenerate_output_is_not_scored_as_perfect_adherence():
    """A refusal, an empty array, and an all-out-of-pool answer must not score 0.0.

    Lower is better on this metric, so 0.0 is the best score there is. The 2026-08-17
    publication review found `value=share if n_sel else 0.0`, which handed exactly
    that score to every degenerate completion — while the module docstring claimed
    "a silent parse failure cannot look like a good score". This is the test for the
    property the docstring promises.
    """
    import math

    for completion in (
        "I'm sorry, I can't help with that.",
        "[]",
        json.dumps(["Not A Real Title", "Also Not Real"]),
    ):
        res = _score(completion, k=12)
        assert res.metadata["n_selected"] == 0
        assert math.isnan(res.value), f"{completion!r} scored {res.value!r}, not nan"


def _sample_scores(values):
    from inspect_ai.scorer import SampleScore, Score

    return [SampleScore(score=Score(value=v)) for v in values]


def test_mean_drops_degenerate_samples_instead_of_folding_them_in():
    """The whole point: one refusal must not drag the adherence figure toward 0."""
    good = [0.4, 0.6]
    assert mean_well_formed()(_sample_scores(good)) == pytest.approx(0.5)
    # a nan sample joins them: the figure must not move
    assert mean_well_formed()(
        _sample_scores(good + [float("nan")])
    ) == pytest.approx(0.5)
    # and numpy's plain mean is what we are NOT doing
    import numpy as np

    assert np.isnan(np.mean(good + [float("nan")]))


def test_mean_of_all_degenerate_is_nan_not_zero():
    import math

    assert math.isnan(mean_well_formed()(_sample_scores([float("nan")] * 3)))


def test_stderr_uses_the_same_denominator_as_the_mean():
    import numpy as np

    vals = [0.2, 0.4, 0.6]
    expected = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    assert stderr_well_formed()(_sample_scores(vals)) == pytest.approx(expected)
    assert stderr_well_formed()(
        _sample_scores(vals + [float("nan")])
    ) == pytest.approx(expected)
    # fewer than two well-formed samples: 0.0, mirroring inspect's own stderr guard
    assert stderr_well_formed()(_sample_scores([0.5, float("nan")])) == 0.0


# --------------------------------------------------------------- well_formed


def _wf(completion: str, k: int) -> float:
    sc = well_formed()
    return asyncio.run(sc(_state(completion, {"k": k}), Target(""))).value


def test_well_formed_separates_garbage_from_adherence():
    pool = load_pool()
    labels = load_labels()
    clean = [c["title"] for c in pool if not labels[c["title"]]["cosmic_horror"]][:12]
    assert _wf(json.dumps(clean), 12) == 1.0
    # perfect "adherence" via an empty answer must NOT look well formed
    assert _wf("[]", 12) == 0.0
    assert _wf("nonsense", 12) == 0.0
    assert _wf(json.dumps(clean[:5]), 12) == 0.0


# ------------------------------------------------------------------- dataset


def test_dataset_shape():
    d = build_dataset((12, 28))
    assert len(d) == 4
    ids = {s.id for s in d}
    assert ids == {"k12-control", "k12-profile", "k28-control", "k28-profile"}
    for s in d:
        assert (s.metadata["profile_on"]) == ("profile" in s.id)


def test_profile_only_appears_in_profile_condition():
    d = build_dataset((12,))
    by_id = {s.id: s for s in d}
    assert "taste profile" in by_id["k12-profile"].input
    assert "taste profile" not in by_id["k12-control"].input
    # the dislike under test must actually be present
    assert "weird [subgenre] (1.0/5" in by_id["k12-profile"].input


# ------------------------------------------------------- the -T slots override


def test_slots_override_accepts_every_form_the_cli_produces():
    """`-T slots=12,28` arrives as a list, not a string.

    The 2026-08-17 review found that the documented CLI override raised
    `ValueError: invalid literal for int() with base 10: '[12'`, because the only
    handled form was `str`. All three forms must work, and the default sweep must
    survive an override that is falsy-but-absent.
    """
    from curation_eval.task import DEFAULT_SLOTS, curation_adherence

    expected = {"k12-control", "k12-profile", "k28-control", "k28-profile"}
    assert {s.id for s in curation_adherence(slots="12,28").dataset} == expected
    assert {s.id for s in curation_adherence(slots=[12, 28]).dataset} == expected
    assert {s.id for s in curation_adherence(slots=["12", " 28 "]).dataset} == expected
    assert {s.id for s in curation_adherence(slots=12).dataset} == {
        "k12-control",
        "k12-profile",
    }
    assert len(curation_adherence().dataset) == 2 * len(DEFAULT_SLOTS)
