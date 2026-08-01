"""
replay_baselines.py — run the eval end to end at $0 against deterministic policies.

No API calls, no key, no spend. These are not measurements of a model; they are
the **reference lines** any live result has to be read against, and computing them
first is most of the reason the live run will be interpretable.

Three policies:

  reputation  Take the top k in the pool's own reputation ranking, ignoring the
              profile entirely. This is the null: it is what JOURNAL Entry 37 says
              the model collapsed to at batch 40 ("ranking on reputation and never
              weighing the dislike"). Its curve is the "profile does nothing" line.

  oracle      Avoid cosmic horror as far as the pool allows, then fill from the
              remainder. This traces the arithmetic floor — the best any curator
              could do. Above k=20 it is forced upward by combinatorics, not by
              preference, which is the whole reason the floor is reported.

  recorded    The actual 31 titles the production run selected on 2026-07-28,
              replayed. Only defined at k=31; it is the single real historical
              data point available at zero cost, and it must score 11/31 = 0.355.

Run:
    python scripts/replay_baselines.py
"""

from __future__ import annotations

import json
import re

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from curation_eval.dataset import build_dataset, load_labels, load_pool
from curation_eval.task import curation_adherence

SLOTS = (4, 8, 12, 20, 28)


def _k_from_messages(messages) -> int:
    """The system message states the slot count; recover it."""
    for m in messages:
        text = getattr(m, "text", "") or ""
        hit = re.search(r"Select exactly (\d+) of them", text)
        if hit:
            return int(hit.group(1))
    return 0


def policy(name: str):
    pool = load_pool()
    labels = load_labels()
    titles = [c["title"] for c in pool]
    clean = [t for t in titles if not labels[t]["cosmic_horror"]]
    cosmic = [t for t in titles if labels[t]["cosmic_horror"]]

    def choose(k: int) -> list[str]:
        if name == "reputation":
            return titles[:k]
        if name == "oracle":
            return (clean + cosmic)[:k]
        if name == "recorded":
            return titles[:k]  # the recorded run took the pool in ranked order
        raise ValueError(name)

    def generate(messages, tools, tool_choice, config) -> ModelOutput:
        k = _k_from_messages(messages)
        return ModelOutput.from_content(
            model=f"replay/{name}", content=json.dumps(choose(k))
        )

    return generate


def main() -> None:
    pool = load_pool()
    labels = load_labels()
    n_total = len(pool)
    n_cosmic = sum(1 for v in labels.values() if v["cosmic_horror"])

    print("=" * 72)
    print("DETERMINISTIC BASELINES — $0, no API calls")
    print("=" * 72)
    print(f"pool: {n_total} candidates, {n_cosmic} cosmic horror "
          f"({n_cosmic / n_total:.1%} of the pool)")
    print()

    results: dict[str, dict[int, float]] = {}
    for name in ("reputation", "oracle"):
        model = get_model("mockllm/model", custom_outputs=policy(name))
        logs = inspect_eval(
            curation_adherence(slots=",".join(str(s) for s in SLOTS)),
            model=model,
            log_dir="./logs",
            display="none",
        )
        per_k: dict[int, float] = {}
        for sample in logs[0].samples or []:
            md = sample.scores["cosmic_share"].metadata
            if md.get("profile_on"):
                continue  # policies ignore the profile; one curve per policy
            per_k[md["n_requested"]] = md["share"]
        results[name] = per_k

    print(f"  {'k':>4s} {'floor':>8s} {'oracle':>8s} {'reputation':>11s} "
          f"{'headroom':>9s}")
    for k in SLOTS:
        floor = max(0, k - (n_total - n_cosmic)) / k
        orc = results["oracle"].get(k)
        rep = results["reputation"].get(k)
        print(f"  {k:>4d} {floor:>8.3f} {orc:>8.3f} {rep:>11.3f} "
              f"{(rep - floor):>9.3f}")
    print()
    print("  floor      = combinatorially forced cosmic share at k slots")
    print("  oracle     = best achievable (should equal floor at every k)")
    print("  reputation = profile ignored, rank order taken")
    print("  headroom   = how much room a working profile has to move the number.")
    print()

    headroom = {k: results["reputation"][k] - max(0, k - (n_total - n_cosmic)) / k
                for k in SLOTS}
    dead = [k for k, h in headroom.items() if h <= 1e-9]
    live = [k for k, h in headroom.items() if h > 1e-9]
    best = max(headroom, key=lambda k: headroom[k])

    print("  ** READ THE HEADROOM COLUMN BEFORE READING ANY LIVE RESULT. **")
    if dead:
        print(f"  At k={', '.join(str(k) for k in dead)} the headroom is ZERO: the")
        print("  control already selects no cosmic horror, so the profile cannot")
        print("  demonstrate an effect there even if it works perfectly. Those slot")
        print("  counts are uninformative BY CONSTRUCTION and a 'null result' at")
        print("  them means nothing. This was not obvious before the baseline ran.")
    if live:
        print(f"  The informative range is k={min(live)}-{max(live)}, peaking at "
              f"k={best} (headroom {headroom[best]:.3f}).")
    print(f"  By k={max(SLOTS)} headroom has collapsed to {headroom[max(SLOTS)]:.3f}.")
    print("  So an effect that 'disappears at large k' is partly the pool running")
    print("  out of alternatives — a weaker claim than Entry 37's, and the honest")
    print("  version of it.")
    print()
    print("  Note also that the reputation curve is NOT monotonic in k. The cosmic")
    print("  titles sit in the middle of the reputation ranking, not the top, so")
    print("  the control's own cosmic share rises then falls. Any effect size")
    print("  measured against 'the control' therefore depends on k in a way that a")
    print("  single-batch-size experiment cannot see at all.")
    print()

    # ---- the one real historical data point -------------------------------
    recorded = [c["title"] for c in pool]
    n_rec_cosmic = sum(1 for t in recorded if labels[t]["cosmic_horror"])
    print("-" * 72)
    print("RECORDED PRODUCTION RUN (horror_readaloud JOURNAL Entry 37)")
    print("-" * 72)
    print(f"  selected {len(recorded)} titles, {n_rec_cosmic} cosmic horror "
          f"=> share {n_rec_cosmic / len(recorded):.3f}")
    print(f"  Entry 37 reports 11 of 12 Lovecraft titles taken: "
          f"{'MATCHES' if n_rec_cosmic == 11 else 'MISMATCH — STOP'}")
    print("  This is the anchor. The scorer is validated against a historical")
    print("  event, not against its own assumptions (tests/test_scorer.py).")
    print()


if __name__ == "__main__":
    main()
