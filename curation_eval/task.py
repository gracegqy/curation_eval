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
def curation_adherence(slots: str | None = None):
    """Does a stated taste dislike survive into the curator's selection, and does
    that depend on how many slots it is filling?

    Args:
        slots: comma-separated slot counts, e.g. "12,40". Defaults to the sweep.
    """
    chosen = DEFAULT_SLOTS
    if slots:
        chosen = tuple(int(s) for s in str(slots).split(",") if s.strip())

    return Task(
        dataset=build_dataset(chosen),
        solver=[per_sample_system(), generate()],
        scorer=[cosmic_share(), well_formed()],
    )
