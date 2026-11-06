"""Run every task under every router, with repetitions, and report the paper's metrics.

  python -m bench.run_bench --routers always-small always-large heuristic --reps 3
  python -m bench.run_bench --resume bench/results/<timestamp> ...   # continue an interrupted run

On native Linux, run with sudo so RAPL energy counters are readable:
  sudo .venv/bin/python -m bench.run_bench ...

Each finished task is appended to runs.csv immediately, so a crash or Ctrl+C loses nothing.
The analysis (bench.analyze) prints at the end.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.tasks import TASKS                     # noqa: E402
from neuroos.agent import NeuroOS                 # noqa: E402
from neuroos.config import Config                 # noqa: E402
from neuroos.energy import EnergyMeter, idle_baseline_watts  # noqa: E402

FIELDS = ["router", "rep", "task", "category", "split", "difficulty", "pathway", "escalations", "success", "self_check",
          "seconds", "energy_local_j", "energy_above_idle_j", "energy_cloud_est_j", "local",
          "tokens", "tool_calls", "error"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", nargs="+", default=["always-small", "always-large", "heuristic"],
                    help="add '+verify' to enable self-verification, e.g. heuristic+verify")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tasks", nargs="*", help="subset of task ids")
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    ap.add_argument("--category", nargs="*", help="only these categories")
    ap.add_argument("--idle-seconds", type=float, default=10.0, help="idle baseline measurement")
    ap.add_argument("--cooldown", type=float, default=2.0, help="seconds between tasks")
    ap.add_argument("--resume", help="results folder of an interrupted run to continue")
    a = ap.parse_args(argv)

    if a.resume:
        out = Path(a.resume)
        if not (out / "runs.csv").exists():
            print(f"No runs.csv in {out}"); return 1
    else:
        out = Path(__file__).parent / "results" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    ws = out / "workspace"
    runs_csv = out / "runs.csv"

    done: set[tuple[str, str, str]] = set()
    if runs_csv.exists():
        with runs_csv.open() as f:
            done = {(r["router"], r["rep"], r["task"]) for r in csv.DictReader(f)}
        print(f"Resuming: {len(done)} task runs already recorded")
    else:
        with runs_csv.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    meter = EnergyMeter()
    print(f"Energy meter: {meter.describe()}")
    idle_w = idle_baseline_watts(meter, a.idle_seconds) if a.idle_seconds > 0 else None
    print(f"Idle baseline: {idle_w if idle_w is None else round(idle_w, 2)} W")

    tasks = [t for t in TASKS if (not a.tasks or t.id in a.tasks)
             and (a.split == "all" or t.split == a.split)
             and (not a.category or t.category in a.category)]
    print(f"Tasks: {len(tasks)} | routers: {a.routers} | reps: {a.reps}")
    shell_reported = False
    try:
        for router in a.routers:
            for rep in range(a.reps):
                for t in tasks:
                    if (router, str(rep), t.id) in done:
                        continue
                    if ws.exists():
                        shutil.rmtree(ws)
                    ws.mkdir()
                    t.setup(ws)
                    cfg = Config.load()
                    cfg.workspace, cfg.auto_approve, cfg.log_dir = str(ws), True, str(out / "logs")
                    cfg.shell_mode = "sandbox"          # benchmarks never get an unsandboxed shell
                    base, _, opt = router.partition("+")
                    cfg.self_verify = opt == "verify"
                    # fixed baselines measure ONE pathway: no escalation, so outcomes are pure
                    if base.startswith("always-") and not cfg.self_verify:
                        cfg.escalate_on_failure = False
                        cfg.validate = False      # record validator findings, but don't act on them
                    agent = NeuroOS(cfg, router=base)
                    if not shell_reported:
                        print(f"Shell for agents: {agent.shell_status}")
                        shell_reported = True

                    t0 = time.perf_counter()
                    try:
                        res = agent.run(t.request, checker=t.check)
                        above_idle = None
                        if res.energy_local_j is not None and idle_w is not None:
                            above_idle = max(0.0, res.energy_local_j - idle_w * res.seconds)
                        errs = "; ".join(x.error for x in res.attempts if x.error)
                        row = {
                            "router": router, "rep": rep, "task": t.id, "category": t.category, "split": t.split,
                            "difficulty": t.difficulty, "pathway": res.pathway, "escalations": len(res.attempts) - 1,
                            "success": int(res.success), "self_check": res.self_check or "",
                            "seconds": round(res.seconds, 3),
                            "energy_local_j": res.energy_local_j, "energy_above_idle_j": above_idle,
                            "energy_cloud_est_j": round(res.energy_cloud_est_j, 3),
                            "local": int(res.pathway in ("tool", "small", "large")),
                            "tokens": sum(x.prompt_tokens + x.completion_tokens for x in res.attempts),
                            "tool_calls": len(res.tool_calls), "error": errs[:300],
                        }
                    except Exception as e:      # any crash = recorded failure, run continues
                        row = {k: "" for k in FIELDS}
                        row.update({"router": router, "rep": rep, "task": t.id, "category": t.category,
                                    "split": t.split, "difficulty": t.difficulty,
                                    "pathway": "error", "escalations": 0, "success": 0,
                                    "seconds": round(time.perf_counter() - t0, 3), "local": 1,
                                    "energy_cloud_est_j": 0,
                                    "error": f"{type(e).__name__}: {e}"[:300]})
                        traceback.print_exc(limit=2)

                    with runs_csv.open("a", newline="") as f:
                        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                    note = f"  ! {row['error'][:80]}" if row["error"] else ""
                    print(f"{router:<20} rep{rep} {t.id:<14} {row['pathway']:<6} ok={row['success']} "
                          f"{float(row['seconds']):.1f}s{note}")
                    time.sleep(a.cooldown)
    except KeyboardInterrupt:
        print(f"\nStopped. Continue later with:\n  python -m bench.run_bench --resume {out} "
              f"--routers {' '.join(a.routers)} --reps {a.reps}")

    print()
    from bench.analyze import main as analyze_main
    analyze_main([str(out)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
