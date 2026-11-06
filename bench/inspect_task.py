"""Show what each model actually did on given tasks (from the run logs).

  python -m bench.inspect_task append-line csv-sum            # latest run
  python -m bench.inspect_task --run bench/results/<ts> remove-blank

Prints, per pathway and rep: success, the tool calls with their arguments, the files the
model wrote (content preview), and its final answer. Use it to decide whether a failure is
the model's fault or the checker's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.tasks import TASKS  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--run", help="results folder (default: latest)")
    ap.add_argument("--reps", type=int, default=1, help="how many reps to show per pathway")
    a = ap.parse_args(argv)
    run = Path(a.run) if a.run else sorted(p for p in RESULTS.glob("*") if (p / "logs").exists())[-1]
    by_id = {t.id: t for t in TASKS}
    recs = [json.loads(l) for f in sorted((run / "logs").glob("*.jsonl"))
            for l in f.read_text().splitlines() if l.strip()]
    for tid in a.tasks:
        t = by_id.get(tid)
        if not t:
            print(f"unknown task {tid}"); continue
        print(f"\n{'=' * 70}\n{tid}  [{t.category}/{t.difficulty}]\nRequest: {t.request}")
        shown: dict = {}
        for r in recs:
            if r["task"] != t.request:
                continue
            key = r["router"]
            if shown.get(key, 0) >= a.reps:
                continue
            shown[key] = shown.get(key, 0) + 1
            path = " -> ".join(x["pathway"] for x in r["attempts"])
            print(f"\n--- {r['router']} ({path}) success={r['success']} {r['seconds']:.0f}s")
            for c in r["tool_calls"]:
                args = json.dumps(c["args"])
                print(f"  {c['tool']}({args[:160]}{'...' if len(args) > 160 else ''})")
                if c["tool"] in ("read_file", "run_shell", "list_dir", "file_stats", "find_files"):
                    print(f"     -> {c['result_preview'][:120]!r}")
            errs = [x["error"] for x in r["attempts"] if x.get("error")]
            if errs:
                print(f"  errors: {errs}")
            print(f"  answer: {r['answer'][:300]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
