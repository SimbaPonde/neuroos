"""Command line: talk to your computer.

  python -m neuroos                    # interactive chat
  python -m neuroos "list files"       # one task
  python -m neuroos --router always-small "..."
  python -m neuroos --doctor           # health check of this computer
  python -m neuroos --doctor "why is my laptop slow?"   # health check, explained
  python -m neuroos --guard            # security check (first run records what is normal)
  python -m neuroos --guard "has anyone tried to hack me?"
  python -m neuroos --guard --accept   # accept the current state as normal after reviewing it
  python -m neuroos --watch notes.md   # AI document: answer "@neuro ..." lines written in the file
  python -m neuroos --watch https://docs.google.com/document/d/<id>/edit   # same, in a Google Doc
  python -m neuroos --check            # check NeuroOS setup: models, energy meter, sandbox
"""

from __future__ import annotations

import argparse
import sys

from .agent import NeuroOS
from .backends import BackendError, OllamaBackend
from .config import Config
from .router import Pathway


def doctor(cfg: Config) -> int:
    agent = NeuroOS(cfg, log=False)
    print(f"workspace     : {agent.gate.root}")
    print(f"energy meter  : {agent.meter.describe()}")
    from .apps import available
    print("apps          : " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in available().items())
          + "   (missing ones: pip install -r requirements-apps.txt)")
    print(f"shell         : {agent.shell_status}"
          + ("   (install bubblewrap for a sandboxed shell: sudo apt install bubblewrap)"
             if agent.shell_status == "off" else ""))
    print(f"self-verify   : {'on' if cfg.self_verify else 'off'}")
    print(f"pathways      : {', '.join(p.value for p in agent.available)}")
    bad = 0
    for p, be in agent.backends.items():
        try:
            r = be.chat([{"role": "user", "content": "Reply with: ok"}], [])
            print(f"  {p.value:<6} {be.name:<28} OK  ({r.latency_s:.1f}s)")
        except BackendError as e:
            bad += 1
            print(f"  {p.value:<6} {be.name:<28} FAIL  {e}")
    if isinstance(agent.backends.get(Pathway.SMALL), OllamaBackend):
        print(f"resident models: {agent.backends[Pathway.SMALL].loaded_models()}")
    return 1 if bad else 0


def show(res) -> None:
    errors = [f"{a.pathway}: {a.error}" for a in res.attempts if a.error]
    if res.answer:
        print(f"\n{res.answer}\n")
    else:
        print("\n(no answer)\n")
    for e in errors:
        hint = ""
        if "not found" in e and "model" in e:
            model = e.split("model '")[-1].split("'")[0] if "model '" in e else "<model>"
            hint = f"  -> fix: ollama pull {model}"
        elif "Cannot reach" in e:
            hint = "  -> fix: start Ollama (sudo systemctl start ollama, or: ollama serve &)"
        print(f"! {e[:200]}{hint}")
    e = "n/a" if res.energy_local_j is None else f"{res.energy_local_j:.1f} J local"
    if res.energy_cloud_est_j:
        e += f" + ~{res.energy_cloud_est_j:.1f} J cloud (est.)"
    path = " -> ".join(a.pathway + (f"({a.self_check})" if a.self_check else "") for a in res.attempts)
    print(f"[{path} | {res.seconds:.1f}s | {e} | tools: {len(res.tool_calls)}]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="neuroos")
    ap.add_argument("task", nargs="*", help="request in plain language")
    ap.add_argument("--router", default="heuristic",
                    help="heuristic | learned | always-small | always-large | always-cloud")
    ap.add_argument("--config", help="path to JSON config")
    ap.add_argument("--check", action="store_true", help="check NeuroOS setup: models, energy meter, sandbox")
    ap.add_argument("--verify", action="store_true", help="self-check results; escalate on FAIL")
    ap.add_argument("--doctor", "--diagnose", dest="diagnose", action="store_true",
                    help="health check of this computer (add a question to get it explained)")
    ap.add_argument("--guard", action="store_true",
                    help="security check: what changed, suspicious programs, failed logins (add a question to explain)")
    ap.add_argument("--accept", action="store_true", help="with --guard: accept the current state as normal")
    ap.add_argument("--watch", metavar="DOC", help="AI document: a local .md/.txt file or a Google Docs link")
    ap.add_argument("--once", action="store_true", help="with --watch: answer pending requests once, then exit")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config)
    if a.verify:
        cfg.self_verify = True
    if a.check:
        return doctor(cfg)
    if a.guard and (not a.task or a.accept):
        from .guard import format_report as gfmt, security_check
        print(gfmt(security_check(accept=a.accept)))
        return 0
    if a.diagnose and not a.task:
        from .doctor import diagnose, format_report
        print(format_report(diagnose()))
        return 0
    agent = NeuroOS(cfg, router=a.router)
    if a.watch:
        from .aidoc import open_doc, watch
        n = watch(open_doc(a.watch), agent, once=a.once)
        print(f"Answered {n} request(s).")
        return 0

    if a.task:
        show(agent.run(" ".join(a.task), doctor=True if a.diagnose else None,
                       guard=True if a.guard else None))
        return 0

    print(f"NeuroOS ready. Workspace: {agent.gate.root}\nEnergy meter: {agent.meter.describe()}\n"
          "Type a request (or 'exit').")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.lower() in ("exit", "quit"):
            return 0
        if line:
            show(agent.run(line))


if __name__ == "__main__":
    sys.exit(main())
