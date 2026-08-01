"""Build the Inspect dataset: one sample per (profile condition x slot count).

The experiment under test comes from horror_readaloud JOURNAL Entries 34/35/37.
A taste profile says the listener dislikes `weird [subgenre]` — cosmic horror.
The curator is asked to select k titles from a reputation-ranked candidate pool
in which 11 of 31 titles are cosmic horror. The question is whether the stated
dislike survives into the selection.

Entry 35 gated this at k=12 and passed: with the profile, Lovecraft picks went
7/12 -> 0/12, against a measured noise floor of 1 title (an A/A' control run with
no profile on either side). Entry 37 then ran production at k=40 and the model
took 11 of 12 Lovecraft titles. Same prompt, same profile, opposite answer.

The proposed mechanism, in Entry 37's words: *"Filling 40 slots means reaching far
enough down the reputation ranking that a single strong dislike stops being
decisive."* The profile still works at the top of the ranking; it does not
EXCLUDE at depth.

That claim is a statement about how the effect varies with k — so it should be
measured as a curve, not at one point. The original was measured at two points,
in two separate hand-run sessions, a week apart, with n=1 per cell. Here it is a
parameter sweep.

CEILING NOTE — read this before interpreting any result. The pool is fixed at 31
candidates of which 11 are cosmic horror, so selecting k titles cannot push the
cosmic share below max(0, (k - 20) / k). At k=28 at least 8 of 28 (29%) MUST be
cosmic horror no matter how well the profile works. The arithmetic floor is
computed per sample and reported alongside the observed share; an eval that
reported the raw share without it would show a "failure" at high k that is pure
combinatorics. This ceiling is a property of the recorded pool, not of the model,
and it is the main reason this reproduction is not a like-for-like replay of
Entry 37 (see README §What this reproduction can and cannot show).
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample

DATA = Path(__file__).resolve().parents[1] / "data"

# Slot counts to sweep. 12 is Entry 35's gate; the rest walk out toward the
# pool size so the decay Entry 37 describes becomes visible if it is there.
DEFAULT_SLOTS = (4, 8, 12, 20, 28)

SYSTEM = """You are curating a read-aloud horror fiction channel.

You will be given a list of candidate stories, each with the evidence that got it
onto the shortlist. Select exactly {k} of them for the listening queue.

Reply with ONLY a JSON array of the exact titles you select, and nothing else.
Example: ["Title One", "Title Two"]"""

PROFILE_BLOCK = """
The listener's taste profile, learned from their ratings:

{profile}
"""


def load_pool() -> list[dict]:
    return json.loads((DATA / "pool.json").read_text())


def load_labels() -> dict[str, dict]:
    return json.loads((DATA / "labels.json").read_text())


def load_profile() -> str:
    return (DATA / "profile.txt").read_text().strip()


def render_candidates(pool: list[dict]) -> str:
    lines = []
    for i, c in enumerate(pool, 1):
        author = c.get("author") or "unknown"
        lines.append(f"{i}. {c['title']} — {author} [{c['source_class']}]")
        for e in (c.get("evidence") or [])[:2]:
            lines.append(f"     evidence: {e}")
    return "\n".join(lines)


def arithmetic_floor(k: int, n_cosmic: int, n_total: int) -> float:
    """Lowest cosmic share achievable at k slots given the pool's composition.

    With n_total - n_cosmic non-cosmic titles available, a selection of k must
    include at least k - (n_total - n_cosmic) cosmic titles.
    """
    forced = max(0, k - (n_total - n_cosmic))
    return forced / k if k else 0.0


def build_dataset(slots: tuple[int, ...] = DEFAULT_SLOTS) -> MemoryDataset:
    pool = load_pool()
    labels = load_labels()
    profile = load_profile()
    n_total = len(pool)
    n_cosmic = sum(1 for t in labels.values() if t["cosmic_horror"])
    candidates = render_candidates(pool)

    samples: list[Sample] = []
    for k in slots:
        if k > n_total:
            continue
        for profile_on in (False, True):
            body = f"Candidate stories:\n\n{candidates}\n"
            if profile_on:
                body += PROFILE_BLOCK.format(profile=profile)
            samples.append(
                Sample(
                    input=body,
                    target="",  # scored against the profile, not a fixed answer key
                    id=f"k{k}-{'profile' if profile_on else 'control'}",
                    metadata={
                        "k": k,
                        "profile_on": profile_on,
                        "condition": f"k={k:02d} {'profile' if profile_on else 'control'}",
                        "n_total": n_total,
                        "n_cosmic": n_cosmic,
                        "arithmetic_floor": arithmetic_floor(k, n_cosmic, n_total),
                        "system": SYSTEM.format(k=k),
                    },
                )
            )
    return MemoryDataset(samples)
