"""Analyze a benchmark run: per-router cost per success, plus the ORACLE router
(cheapest pathway that succeeded on each task) - the upper bound a learned router chases.

  python -m bench.analyze                       # latest run
  python -m bench.analyze bench/results/<ts>    # specific run

Cost = measured joules when every row has them, otherwise wall-clock seconds
(a rough proxy on CPU-only machines; say so in the paper).
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(run_dir: Path) -> list[dict]:
    with (run_dir / "runs.csv").open() as f:
        return list(csv.DictReader(f))


def analyze(rows: list[dict]) -> tuple[str, list[dict]]:
    energy = [(_f(r["energy_above_idle_j"]) if _f(r["energy_above_idle_j"]) is not None
               else _f(r["energy_local_j"])) for r in rows]
    use_j = all(e is not None for e in energy)
    unit = "J" if use_j else "s"
    for r, e in zip(rows, energy):
        r["_cost"] = (e + (_f(r["energy_cloud_est_j"]) or 0)) if use_j else float(r["seconds"])
        r["_ok"] = r["success"] == "1"

    out = []
    by_router = defaultdict(list)
    for r in rows:
        by_router[r["router"]].append(r)
    for name, rs in by_router.items():
        succ = sum(r["_ok"] for r in rs)
        cost = sum(r["_cost"] for r in rs)
        silent = sum(1 for r in rs if not r["_ok"] and r["pathway"] != "tool" and r["escalations"] == "0")
        out.append({"router": name, "success": f"{succ}/{len(rs)}", "rate": succ / len(rs),
                    f"total_{unit}": cost, f"{unit}_per_success": cost / succ if succ else float("inf"),
                    "silent_failures": silent,
                    "escalations": sum(int(r["escalations"]) for r in rs)})

    # Oracle: per (rep, task), cheapest successful outcome among all routers' runs
    cells = defaultdict(list)
    for r in rows:
        cells[(r["rep"], r["task"])].append(r)
    o_succ = o_cost = 0
    for rs in cells.values():
        ok = [r for r in rs if r["_ok"]]
        if ok:
            o_succ += 1
            o_cost += min(r["_cost"] for r in ok)
        else:
            o_cost += min(r["_cost"] for r in rs)
    out.append({"router": "ORACLE", "success": f"{o_succ}/{len(cells)}", "rate": o_succ / len(cells),
                f"total_{unit}": o_cost, f"{unit}_per_success": o_cost / o_succ if o_succ else float("inf"),
                "silent_failures": "-", "escalations": "-"})
    return unit, out


def per_task(rows: list[dict]) -> str:
    routers = sorted({r["router"] for r in rows})
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    lines = ["task".ljust(15) + "".join(x[:16].ljust(18) for x in routers)]
    for t in tasks:
        cells = []
        for rt in routers:
            rs = [r for r in rows if r["task"] == t and r["router"] == rt]
            ok = sum(r["success"] == "1" for r in rs)
            path = "/".join(sorted({r["pathway"] for r in rs}))
            cells.append(f"{ok}/{len(rs)} {path}".ljust(18))
        lines.append(t.ljust(15) + "".join(cells))
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv:
        run = Path(argv[0])
    else:
        runs = sorted(p for p in RESULTS.glob("*") if (p / "runs.csv").exists())
        if not runs:
            print("No results found."); return 1
        run = runs[-1]
    rows = load(run)
    unit, table = analyze(rows)
    print(f"Run: {run}\nCost unit: {'joules (measured)' if unit == 'J' else 'seconds (proxy: no energy meter)'}\n")
    hdr = ["router", "success", "rate", f"total_{unit}", f"{unit}_per_success", "silent_failures", "escalations"]
    print("  ".join(h.ljust(16) for h in hdr))
    for r in table:
        vals = [r["router"], r["success"], f"{r['rate']:.0%}", f"{r[f'total_{unit}']:.1f}",
                f"{r[f'{unit}_per_success']:.1f}", str(r["silent_failures"]), str(r["escalations"])]
        print("  ".join(v.ljust(16) for v in vals))
    print("\nPer task (successes / runs, pathway used):\n" + per_task(rows))
    with (run / "analysis.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore")
        w.writeheader(); w.writerows(table)
    print(f"\nSaved {run / 'analysis.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
