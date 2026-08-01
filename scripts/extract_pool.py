"""
extract_pool.py — build the evaluation dataset from the recorded horror_readaloud run.

The dataset for this eval is not synthetic and is not hand-written. It is lifted
from an actual production curation run in the sibling project `horror_readaloud`:
`curation_runs` row id=5, 2026-07-28 06:38 UTC, model claude-sonnet-5, $0.0512 —
the run written up in that project's JOURNAL **Entry 37**.

That run is the interesting one precisely because it FAILED. The project's Phase-6
gate (Entry 35) had measured the taste profile driving Lovecraft picks from 7/12 to
0/12 at batch size 12. Entry 37 then ran production at `POOL_BATCH_SIZE = 40` and
the model took **11 of the 12 Lovecraft titles in the shortlist** — the mechanism
reorders the top of the ranking but does not exclude at depth. The gate was
re-opened rather than explained away.

This script pulls three things out of that row:

  data/pool.json     the 31 selected candidates, with author, source class, evidence
                     and the model's own selection_note for each
  data/profile.txt   the exact taste-profile text that reached the model
  data/labels.json   the ground-truth label per candidate: is it cosmic-horror /
                     "weird" fiction — the subgenre the profile says to avoid

VALIDATION. The extraction asserts the two counts Entry 37 reports, so the dataset
cannot silently drift from the journal it cites: 11 Lovecraft titles, and a source
mix of 28 gutenberg / 3 creepypasta. If either assertion fails, the recorded run is
not the run the journal describes and the script refuses to write.

The source database is gitignored in horror_readaloud and is NOT redistributed here
— only this derived metadata (titles, authors, public catalogue references, and
model-generated notes). No story text is copied by this script or stored in this
repo. See README §Content and data.

Run:
    python scripts/extract_pool.py [--db ../horror_readaloud/data/app.db]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_DB = ROOT.parent / "horror_readaloud" / "data" / "app.db"
RUN_ID = 5

# What JOURNAL Entry 37 says this run did. Asserted, not assumed.
EXPECTED_LOVECRAFT = 11
EXPECTED_SOURCE_MIX = {"gutenberg": 28, "creepypasta": 3}

# The disliked subgenre under test. The profile carries `weird [subgenre] 1.0/5`
# as its single strongest dislike; in this shortlist "weird" means cosmic horror,
# and cosmic horror means Lovecraft plus a small number of others.
COSMIC_AUTHORS = {"H. P. (Howard Phillips) Lovecraft"}
# Non-Lovecraft titles that are still squarely cosmic/weird fiction. Kept as an
# explicit list rather than a keyword rule so the labelling is auditable and any
# future reader can disagree with a specific line rather than with a regex.
COSMIC_EXTRA_TITLES: set[str] = set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--run-id", type=int, default=RUN_ID)
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(
            f"source database not found: {args.db}\n"
            "This script only regenerates the dataset; the committed data/ files are "
            "sufficient to run every eval in this repo."
        )

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = list(con.execute("select * from curation_runs where id = ?", (args.run_id,)))
    if not rows:
        raise SystemExit(f"no curation_runs row with id={args.run_id}")
    row = dict(rows[0])

    candidates = json.loads(row["candidates_json"])
    profile = row["taste_profile_text"]

    # ---- validate against the journal before writing anything -----------------
    authors = Counter(c.get("author") or "?" for c in candidates)
    n_lovecraft = sum(n for a, n in authors.items() if a in COSMIC_AUTHORS)
    mix = dict(Counter(c["source_class"] for c in candidates))

    problems = []
    if n_lovecraft != EXPECTED_LOVECRAFT:
        problems.append(f"Lovecraft count {n_lovecraft} != {EXPECTED_LOVECRAFT}")
    if mix != EXPECTED_SOURCE_MIX:
        problems.append(f"source mix {mix} != {EXPECTED_SOURCE_MIX}")
    if "weird [subgenre] (1.0/5" not in (profile or ""):
        problems.append("profile does not carry `weird [subgenre] (1.0/5` as recorded")
    if problems:
        raise SystemExit(
            "REFUSING TO WRITE — recorded run does not match JOURNAL Entry 37:\n  "
            + "\n  ".join(problems)
        )

    # ---- write ---------------------------------------------------------------
    DATA.mkdir(exist_ok=True)

    pool = []
    for c in candidates:
        pool.append({
            "title": c["title"],
            "author": c.get("author"),
            "year": c.get("year"),
            "source_class": c["source_class"],
            "source_ref": c.get("source_ref"),
            "evidence": c.get("evidence") or [],
            "selection_note": c.get("selection_note"),
        })
    (DATA / "pool.json").write_text(json.dumps(pool, indent=2) + "\n")

    (DATA / "profile.txt").write_text(profile.rstrip() + "\n")

    labels = {}
    for c in candidates:
        cosmic = (c.get("author") in COSMIC_AUTHORS) or (c["title"] in COSMIC_EXTRA_TITLES)
        labels[c["title"]] = {"cosmic_horror": cosmic}
    (DATA / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")

    meta = {
        "source_project": "horror_readaloud",
        "source_table": "curation_runs",
        "source_run_id": args.run_id,
        "source_run_created_at": row["created_at"],
        "model": row["model"],
        "cost_usd": row["cost_usd"],
        "journal_entry": 37,
        "n_candidates": len(pool),
        "n_cosmic_horror": sum(1 for v in labels.values() if v["cosmic_horror"]),
        "source_mix": mix,
        "validated_against_journal": True,
    }
    (DATA / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {len(pool)} candidates -> data/pool.json")
    print(f"  cosmic-horror labelled : {meta['n_cosmic_horror']}/{len(pool)}")
    print(f"  source mix             : {mix}")
    print(f"  profile                : {len(profile)} chars -> data/profile.txt")
    print("  validated against JOURNAL Entry 37: OK")


if __name__ == "__main__":
    main()
