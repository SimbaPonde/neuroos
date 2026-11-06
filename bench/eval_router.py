"""Main experiment: how well does each routing policy do on tasks it has NEVER seen?

Method: counterfactual replay. Run every task on every pathway first (always-small and
always-large in run_bench). Every routing policy is then scored on those recorded outcomes,
so no models are rerun and all policies are compared on identical data.

  python -m bench.eval_router                         # latest run that has both pathways
  python -m bench.eval_router bench/results/<ts> ...  # one or more runs (pooled)
  python -m bench.eval_router --cv 5                  # 5-fold cross-validation over tasks

Policies:
  always-small, always-large  fixed baselines
  heuristic                   rule-based router
  learned-knn@t               learned router trained ONLY on training tasks; t = success threshold
  oracle                      cheapest pathway that succeeded, per task and rep (upper bound)

Reported per policy on the held-out tasks: success rate, cost per success (joules if
measured, otherwise seconds), share of tasks sent to the small model, and the
cost-per-success gap to the oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.tasks import TASKS                                     # noqa: E402
from neuroos.router import HeuristicRouter, Pathway, TextKNNRouter, extract  # noqa: E402
from neuroos.validators import validate  # noqa: E402

RESULTS = Path(__file__).parent / "results"
AVAILABLE = [Pathway.TOOL, Pathway.SMALL, Pathway.LARGE]
BY_ID = {t.id: t for t in TASKS}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


_PRE: dict = {}


def preexisting(task_id: str) -> set[str]:
    """Files present before the task ran (recreated from the task's setup)."""
    if task_id not in _PRE:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            BY_ID[task_id].setup(root)
            _PRE[task_id] = {str(x.relative_to(root)) for x in root.rglob("*") if x.is_file()}
    return _PRE[task_id]


def _first_attempts(run_dir: Path) -> dict:
    """(router, request) -> queue of log info per task run: first-attempt seconds, its share of
    the total, number of attempts, tool calls and final answer."""
    q: dict = defaultdict(list)
    for f in sorted((run_dir / "logs").glob("*.jsonl")) if (run_dir / "logs").exists() else []:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            atts = rec.get("attempts") or []
            total = sum(a.get("seconds", 0) for a in atts) or 1.0
            first = atts[0].get("seconds", 0) if atts else 0.0
            q[(rec.get("router"), rec.get("task"))].append(
                {"secs": first, "share": first / total, "n_attempts": len(atts),
                 "calls": rec.get("tool_calls") or [], "answer": rec.get("answer", "")})
    return q


def load(run_dirs: list[Path]) -> tuple[dict, str]:
    """outcomes[task_id][pathway][rep] = (success, cost) for the pathway each fixed router ASKED for.

    - If a fixed router's attempt failed and the agent escalated (e.g. always-small -> large), it is
      counted as a FAILURE of the requested pathway, costing only that first attempt (from the logs).
    - Runs are applied in the order given: a later run REPLACES earlier outcomes for the same
      task and pathway (use this to rerun a few tasks after fixing them).
    - Validator findings are recomputed from the logs for every single-attempt run.
    """
    rows = []
    for d in run_dirs:
        firsts = _first_attempts(d)
        with (d / "runs.csv").open() as f:
            for r in csv.DictReader(f):
                if r["router"] in ("always-small", "always-large") and r["task"] in BY_ID:
                    r["_run"] = d.name
                    key = (r["router"], BY_ID[r["task"]].request)
                    r["_log"] = firsts[key].pop(0) if r["pathway"] != "error" and firsts.get(key) else None
                    rows.append(r)
    use_j = rows and all((_f(r.get("energy_above_idle_j")) if _f(r.get("energy_above_idle_j")) is not None
                          else _f(r.get("energy_local_j"))) is not None for r in rows)
    latest_run: dict = {}
    for r in rows:
        path = "tool" if r["pathway"] == "tool" else r["router"].split("-", 1)[1]
        latest_run[(r["task"], path)] = r["_run"]
    out: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if use_j:
            e = _f(r["energy_above_idle_j"]) if _f(r["energy_above_idle_j"]) is not None else _f(r["energy_local_j"])
            cost = e + (_f(r.get("energy_cloud_est_j")) or 0)
        else:
            cost = float(r["seconds"])
        intended = r["router"].split("-", 1)[1]
        ok = r["success"] == "1"
        log = r["_log"]
        escalated = False
        if r["pathway"] == "tool":
            path = "tool"
        else:
            path = intended
            escalated = r["pathway"] not in (intended, "error") or (_f(r.get("escalations")) or 0) > 0 \
                or (log is not None and log["n_attempts"] > 1)
            if escalated:
                ok = False                                   # the requested pathway did not finish the task
                if log is not None:
                    cost = cost * log["share"] if use_j else log["secs"]
        if latest_run[(r["task"], path)] != r["_run"]:
            continue                                         # superseded by a later run
        key = f'{r["_run"]}:{r["rep"]}'
        out[r["task"]][path][key] = (ok, cost)
        err = r.get("error") or ""
        DETECTABLE[(r["task"], path, key)] = (not ok) and (
            escalated or any(x in err for x in ("empty answer", "escalate", "Timeout", "budget",
                                                "max steps", "Cannot reach")))
        if log is not None and not escalated and path != "tool":
            FLAGGED[(r["task"], path, key)] = validate(BY_ID[r["task"]].request, log["calls"], log["answer"],
                                                       preexisting(r["task"]))
    return out, ("J" if use_j else "s")


DETECTABLE: dict = {}
FLAGGED: dict = {}


def _cascade(o: dict, task_id: str, use_validators: bool) -> tuple[float, float, dict]:
    d = o[task_id]
    if "small" not in d or "large" not in d:
        res = outcome(o, task_id, Pathway.TOOL) if "tool" in d else outcome(o, task_id, Pathway.LARGE)
        return res[0], res[1], {"tool" if "tool" in d else "large": 1.0}
    s_tot = c_tot = 0.0
    picks: dict = defaultdict(float)
    pairs = aligned({"small": d["small"], "large": d["large"]})
    for pr in pairs:
        ok, c, rep = pr["small"]
        flagged = use_validators and bool(FLAGGED.get((task_id, "small", rep)))
        visible_fail = (not ok) and DETECTABLE.get((task_id, "small", rep))
        if visible_fail or flagged:            # flagged also escalates a correct result (false alarm costs)
            lo = pr["large"]
            s_tot += lo[0]; c_tot += c + lo[1]; picks["large"] += 1 / len(pairs)
        else:
            s_tot += ok; c_tot += c; picks["small"] += 1 / len(pairs)
    return s_tot / len(pairs), c_tot / len(pairs), picks


def cascade(o: dict, task_id: str) -> tuple[float, float, dict]:
    """Small first; switch to large only if small's failure is visible (empty answer, ESCALATE,
    timeout, budget). Silent failures (model says 'done') are not caught."""
    return _cascade(o, task_id, False)


def cascade_validated(o: dict, task_id: str) -> tuple[float, float, dict]:
    """Small first; switch to large if the failure is visible OR the runtime validators flag the
    result (neuroos/validators.py). Uses no ground truth."""
    return _cascade(o, task_id, True)


def validator_report(o: dict, task_ids: list[str]) -> dict:
    """How well validators separate the small model's correct results from its silent failures."""
    tp = fp = fn = tn = 0
    codes: dict = defaultdict(int)
    for tid in task_ids:
        for rep, (ok, _) in o[tid].get("small", {}).items():
            if (tid, "small", rep) not in FLAGGED or DETECTABLE.get((tid, "small", rep)):
                continue
            issues = FLAGGED[(tid, "small", rep)]
            for i in issues:
                codes[i.split(":")[0]] += 1
            if issues and not ok: tp += 1
            elif issues and ok: fp += 1
            elif not issues and not ok: fn += 1
            else: tn += 1
    return {"caught": tp, "false_alarms": fp, "missed": fn, "clean_ok": tn, "codes": dict(codes)}


def outcome(o: dict, task_id: str, p: Pathway) -> tuple[float, float] | None:
    obs = o[task_id].get(p.value) or (o[task_id].get("tool") if p is Pathway.TOOL else None)
    if not obs:
        return None
    vals = list(obs.values())
    return sum(s for s, _ in vals) / len(vals), sum(c for _, c in vals) / len(vals)


def aligned(d: dict) -> list[dict]:
    """Pair up repetitions across pathways by position, not by label: runs with different rep counts
    or from different result folders still line up (shorter lists are cycled). Returns one dict
    {pathway: (success, cost, rep_key)} per paired repetition."""
    lists = {p: [(k, *v) for k, v in sorted(reps.items())] for p, reps in d.items() if reps}
    n = max((len(v) for v in lists.values()), default=0)
    return [{p: (v[i % len(v)][1], v[i % len(v)][2], v[i % len(v)][0]) for p, v in lists.items()} for i in range(n)]


def oracle(o: dict, task_id: str) -> tuple[float, float, dict]:
    """Per paired rep: cheapest pathway that succeeded (or cheapest attempt if none did)."""
    pairs = aligned(o[task_id])
    s_tot = c_tot = 0.0
    picks: dict[str, float] = defaultdict(float)
    for pr in pairs:
        ok = [(c, p) for p, (s, c, _) in pr.items() if s]
        best = min(ok) if ok else min((c, p) for p, (_, c, _) in pr.items())
        s_tot += 1.0 if ok else 0.0
        c_tot += best[0]
        picks[best[1]] += 1 / len(pairs)
    return s_tot / len(pairs), c_tot / len(pairs), picks


def records_for(o: dict, ids: list[str]) -> list[dict]:
    recs = []
    for tid in ids:
        for p, reps in o[tid].items():
            for s, c in reps.values():
                recs.append({"task": BY_ID[tid].request, "pathway": p, "success": s, "cost": c})
    return recs


def score(o: dict, test_ids: list[str], choose) -> dict:
    succ = cost = 0.0
    picks: dict = defaultdict(float)
    n = 0
    for tid in test_ids:
        if tid not in o:
            continue
        p = choose(tid)
        if p is None:
            continue
        if p in ("oracle", "cascade", "cascade+validate"):
            s_, c_, op = {"oracle": oracle, "cascade": cascade, "cascade+validate": cascade_validated}[p](o, tid)
            for k_, v_ in op.items():
                picks[k_] += v_
            succ += s_; cost += c_; n += 1
            continue
        res = outcome(o, tid, p)
        if res is None:                         # chosen pathway never measured -> fall back to large
            res = outcome(o, tid, Pathway.LARGE)
            p = Pathway.LARGE
        if res is None:
            continue
        picks[p if isinstance(p, str) else p.value] += 1
        succ += res[0]
        cost += res[1]
        n += 1
    return {"n": n, "success": succ, "cost": cost, "picks": dict(picks)}


def evaluate(o: dict, train_ids: list[str], test_ids: list[str], thresholds: list[float], k: int,
             skip_below: tuple = (0.3, 0.5)) -> dict:
    heur = HeuristicRouter()
    pols = {
        "always-small": lambda tid: Pathway.TOOL if extract(BY_ID[tid].request).tool_intent else Pathway.SMALL,
        "always-large": lambda tid: Pathway.TOOL if extract(BY_ID[tid].request).tool_intent else Pathway.LARGE,
        "heuristic": lambda tid: heur.choose(BY_ID[tid].request, AVAILABLE),
    }
    recs = records_for(o, [t for t in train_ids if t in o])
    for th in thresholds:
        r = TextKNNRouter(recs, k=k, threshold=th)
        pols[f"learned-knn@{th:g}"] = (lambda router: lambda tid: router.choose(BY_ID[tid].request, AVAILABLE))(r)
    pols["cascade"] = lambda tid: "cascade"
    pols["cascade+validate"] = lambda tid: "cascade+validate"
    # hybrid: skip the small model when past outcomes on similar tasks say it usually fails;
    # otherwise try small first and let the validators decide whether to escalate.
    knn = TextKNNRouter(recs, k=k, threshold=0.5)

    def hybrid(tid, cut):
        if extract(BY_ID[tid].request).tool_intent:
            return Pathway.TOOL
        pred = knn.predict(BY_ID[tid].request).get(Pathway.SMALL)
        return Pathway.LARGE if pred is not None and pred[0] < cut else "cascade+validate"
    for cut in skip_below:
        pols[f"learned+validate@{cut:g}"] = (lambda c: lambda tid: hybrid(tid, c))(cut)
    pols["oracle"] = lambda tid: "oracle"
    return {name: score(o, test_ids, fn) for name, fn in pols.items()}


def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
            continue
        m = out[k]
        picks = dict(m["picks"])
        for p, c in v["picks"].items():
            picks[p] = picks.get(p, 0) + c
        out[k] = {"n": m["n"] + v["n"], "success": m["success"] + v["success"],
                  "cost": m["cost"] + v["cost"], "picks": picks}
    return out


def table(res: dict, unit: str) -> list[dict]:
    o_cps = res["oracle"]["cost"] / res["oracle"]["success"] if res["oracle"]["success"] else float("inf")
    rows = []
    for name, r in res.items():
        cps = r["cost"] / r["success"] if r["success"] else float("inf")
        n_model = sum(c for p, c in r["picks"].items() if p in ("small", "large")) or 1
        rows.append({"policy": name, "tasks": r["n"], "success_rate": r["success"] / r["n"] if r["n"] else 0,
                     f"{unit}_per_success": cps, "share_small": r["picks"].get("small", 0) / n_model,
                     "vs_oracle": cps / o_cps if o_cps and o_cps != float("inf") else float("nan")})
    return rows


def show(rows: list[dict], unit: str, title: str) -> None:
    print(f"\n{title}")
    print(f"{'policy':<18}{'tasks':>6}{'success':>10}{unit + '/success':>14}{'% to small':>12}{'x oracle cost':>15}")
    for r in rows:
        print(f"{r['policy']:<18}{r['tasks']:>6}{r['success_rate']:>10.0%}{r[f'{unit}_per_success']:>14.1f}"
              f"{r['share_small']:>12.0%}{r['vs_oracle']:>15.2f}")


def folds(ids: list[str], k: int) -> list[list[str]]:
    by_cat = defaultdict(list)
    for tid in ids:
        by_cat[BY_ID[tid].category].append(tid)
    out = [[] for _ in range(k)]
    for cat in sorted(by_cat):
        for i, tid in enumerate(sorted(by_cat[cat])):
            out[i % k].append(tid)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="results folders (default: latest with both pathways)")
    ap.add_argument("--cv", type=int, default=0, help="k-fold cross-validation over tasks (e.g. 5)")
    ap.add_argument("--k", type=int, default=5, help="neighbours for learned-knn")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--sensitivity", type=float, nargs="*", default=None,
                    help="scale small-model cost by these factors, e.g. --sensitivity 1 0.5 0.25 0.1")
    a = ap.parse_args(argv)

    if a.runs:
        runs = [Path(x) for x in a.runs]
    else:
        cands = [d for d in sorted(RESULTS.glob("*")) if (d / "runs.csv").exists()]
        runs = [d for d in cands if {"always-small", "always-large"} <=
                {r["router"] for r in csv.DictReader((d / "runs.csv").open())}][-1:]
        if not runs:
            print("No run with both always-small and always-large found."); return 1
    o, unit = load(runs)
    both = [t for t in o if "small" in o[t] or "large" in o[t] or "tool" in o[t]]
    print(f"Runs: {', '.join(str(r) for r in runs)}")
    print(f"Tasks with outcomes: {len(both)} | cost unit: "
          f"{'joules (measured)' if unit == 'J' else 'seconds (proxy: no energy meter)'}")
    vr = validator_report(o, both)
    judged = vr["caught"] + vr["missed"]
    if judged + vr["false_alarms"] + vr["clean_ok"]:
        print(f"\nValidators on the small model's results (no ground truth used):\n"
              f"  silent failures caught: {vr['caught']}/{judged}"
              f"   false alarms on correct results: {vr['false_alarms']}/{vr['false_alarms'] + vr['clean_ok']}\n"
              f"  rules that fired: {vr['codes']}")

    out_rows = []
    train = [t.id for t in TASKS if t.split == "train" and t.id in o]
    test = [t.id for t in TASKS if t.split == "test" and t.id in o]
    if train and test:
        rows = table(evaluate(o, train, test, a.thresholds, a.k), unit)
        show(rows, unit, f"Fixed split: trained on {len(train)} tasks, tested on {len(test)} unseen tasks")
        out_rows += [{"eval": "fixed-split", **r} for r in rows]
    else:
        print("\n(fixed split skipped: need outcomes for both train and test tasks)")

    if a.cv >= 2:
        agg: dict = {}
        fs = folds(both, a.cv)
        for i, test_ids in enumerate(fs):
            train_ids = [t for t in both if t not in test_ids]
            agg = merge(agg, evaluate(o, train_ids, test_ids, a.thresholds, a.k))
        rows = table(agg, unit)
        show(rows, unit, f"{a.cv}-fold cross-validation over {len(both)} tasks (every task tested once, unseen)")
        out_rows += [{"eval": f"cv{a.cv}", **r} for r in rows]

    if a.sensitivity and a.cv >= 2:
        import copy
        print(f"\nCost sensitivity: what if the small model were cheaper relative to the large one?\n"
              f"(small-model costs multiplied by the factor; {a.cv}-fold CV; {unit}/success)")
        names = ["always-small", "always-large", "cascade+validate", "learned+validate@0.3",
                 "learned+validate@0.5", "oracle"]
        labels = ["always-small", "always-large", "cascade+val", "learned+val@.3", "learned+val@.5", "oracle"]
        print(f"{'factor':>8}" + "".join(f"{n:>20}" for n in labels))
        for factor in a.sensitivity:
            o2 = copy.deepcopy(o)
            for tid in o2:
                for rep, (ok, c) in list(o2[tid].get("small", {}).items()):
                    o2[tid]["small"][rep] = (ok, c * factor)
            agg = {}
            fs = folds(both, a.cv)
            for test_ids in fs:
                agg = merge(agg, evaluate(o2, [t for t in both if t not in test_ids], test_ids, a.thresholds, a.k))
            rows = {r["policy"]: r for r in table(agg, unit)}
            print(f"{factor:>8g}" + "".join(f"{rows[n][f'{unit}_per_success']:>14.1f} ({rows[n]['success_rate']:.0%})"
                                             if n in rows else f"{'-':>20}" for n in names))
            out_rows += [{"eval": f"cv{a.cv}-small_x{factor:g}", **r} for r in rows.values()]

    dest = runs[-1] / "router_eval.csv"
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    print(f"\nSaved {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
