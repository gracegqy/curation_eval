"""The Inspect task.

Run it:

    # $0 — deterministic replay, no API calls, no key needed. This is the default
    # and it is what CI runs.
    inspect eval curation_eval/task.py --model mockllm/model

    # live, costs money, needs ANTHROPIC_API_KEY
    inspect eval curation_eval/task.py --model anthropic/claude-sonnet-5 --epochs 5

`--epochs N` repeats every sample N times. That is the single most important flag
here: the original hand-rolled comparison ran n=1 per cell, so its headline
numbers ("1 title of noise, 8 of effect") had no interval and could not have had
one. Epochs plus `stderr()` gives every cell a spread for free.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.solver import TaskState, generate, solver, system_message

from curation_eval.dataset import DEFAULT_SLOTS, build_dataset
from curation_eval.scorer import cosmic_share, well_formed


@solver
def per_sample_system():
    """Each sample carries its own system prompt because k varies per sample."""
    async def solve(state: TaskState, generate_fn):
        sys = (state.metadata or {}).get("system")
        if sys:
            state = await system_message(sys)(state, generate_fn)
        return state

    return solve


@task
def curation_adherence(slots: str | int | list[str | int] | None = None):
    """Does a stated taste dislike survive into the curator's selection, and does
    that depend on how many slots it is filling?

    Args:
        slots: slot counts to sweep, e.g. `-T slots=12,28`. Defaults to the sweep.

    Note on the parameter type: inspect parses a comma-separated `-T` value into a
    **list**, not a string, so the documented `-T slots=12,28` arrives here as
    `[12, 28]`. An earlier version only handled `str` and did `str(slots).split(",")`,
    which turned that list into `"[12, 28]"` and raised `ValueError: invalid literal
    for int() with base 10: '[12'` — the CLI override its own docstring documented
    could not be used. Both forms are accepted now, plus a bare `-T slots=12`.
    """
    chosen = DEFAULT_SLOTS
    if slots:
        raw = slots if isinstance(slots, (list, tuple)) else str(slots).split(",")
        chosen = tuple(int(str(s).strip()) for s in raw if str(s).strip())

    return Task(
        dataset=build_dataset(chosen),
        solver=[per_sample_system(), generate()],
        scorer=[cosmic_share(), well_formed()],
    )
