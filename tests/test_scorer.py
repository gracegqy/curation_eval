"""Scorer tests, anchored to the real recorded outcome.

The important one is `test_recorded_run_reproduces_entry_37`. The 2026-07-28
production run in horror_readaloud selected all 31 of these titles, 11 of them
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
from curation_eval.scorer import cosmic_share, parse_titles, well_formed


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
