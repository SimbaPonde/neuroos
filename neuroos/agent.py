"""The NeuroOS agent: natural-language request in, result out.

  request -> router picks pathway -> (tool | model tool-calling loop) -> result
             (on failure: escalate to the next, more capable pathway)

Every task is measured (wall time, joules, tokens per pathway) and logged as one
JSONL record. Those logs are both the research dataset and the training data for
LearnedRouter.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .backends import Backend, BackendError, OllamaBackend, OpenAICompatBackend
from .config import Config
from .energy import EnergyMeter
from .permissions import PermissionDenied, PermissionGate
from .router import ORDER, Pathway, Router, extract, make_router
from .tools import build_tools
from .validators import validate

SYSTEM_PROMPT = """You are NeuroOS, an assistant that operates a Linux computer for the user.
You can only act through the provided tools, and only inside the user's workspace folder.
Work step by step: call tools, read their results, continue until the task is done.
When finished, reply with a short plain-language summary of what you did and the result.
If the task is beyond your ability, reply with exactly: ESCALATE"""


@dataclass
class Attempt:
    pathway: str
    backend: str
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    ok: bool = False
    error: str | None = None
    self_check: str | None = None
    verify_tokens: int = 0
    issues: list = field(default_factory=list)   # validator findings


@dataclass
class TaskResult:
    id: str
    task: str
    router: str
    pathway: str                      # pathway that produced the final answer
    answer: str
    success: bool                     # verified if a checker was given, else "finished cleanly"
    verified: bool
    self_check: str | None            # PASS / FAIL from self-verification of the final attempt
    seconds: float
    energy_local_j: float | None      # measured on this machine (None if no meter available)
    energy_cloud_est_j: float         # estimated from cloud tokens
    energy_total_j: float | None
    energy_sources: list[str]
    attempts: list[Attempt] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class NeuroOS:
    def __init__(self, cfg: Config | None = None, router: Router | str = "heuristic",
                 backends: dict[Pathway, Backend] | None = None,
                 ask: Callable[[str], bool] | None = None, log: bool = True):
        self.cfg = cfg or Config.load()
        self.gate = PermissionGate(self.cfg.workspace, self.cfg.auto_approve, ask)
        self.tools = build_tools(self.gate, self.cfg.shell_mode)
        self.shell_status = ("off" if "run_shell" not in self.tools
                             else ("sandbox (bubblewrap)" if self.cfg.shell_mode == "sandbox" else "host"))
        self.schemas = [t.schema() for t in self.tools.values()]
        self.meter = EnergyMeter(self.cfg.energy_sample_hz)
        self.backends = backends if backends is not None else self._default_backends()
        for be in self.backends.values():
            be.max_output_tokens = self.cfg.max_output_tokens
        self.router = router if isinstance(router, Router) else make_router(router, self.cfg.log_dir)
        self.log_path = Path(self.cfg.log_dir) / f"tasks-{time.strftime('%Y%m%d')}.jsonl" if log else None

    # ---------- setup ----------
    def _default_backends(self) -> dict[Pathway, Backend]:
        b: dict[Pathway, Backend] = {
            Pathway.SMALL: OllamaBackend(self.cfg.small_model, self.cfg.ollama_url,
                                          timeout=self.cfg.request_timeout_s),
            Pathway.LARGE: OllamaBackend(self.cfg.large_model, self.cfg.ollama_url,
                                          timeout=self.cfg.request_timeout_s),
        }
        if self.cfg.cloud_api_key:
            b[Pathway.CLOUD] = OpenAICompatBackend(self.cfg.cloud_model, self.cfg.cloud_url,
                                                   self.cfg.cloud_api_key)
        return b

    @property
    def available(self) -> list[Pathway]:
        return [Pathway.TOOL] + [p for p in ORDER if p in self.backends]

    # ---------- main entry ----------
    def run(self, task: str, checker: Callable[[str, Path], bool] | None = None,
            doctor: bool | None = None, guard: bool | None = None) -> TaskResult:
        feats = extract(task)
        if doctor is not None:
            feats.doctor_question = doctor
        if guard is not None:
            feats.guard_question = guard
            if guard:
                feats.doctor_question = False
        start = self.router.choose(task, self.available)
        attempts: list[Attempt] = []
        tool_log: list[dict] = []
        answer, final_path, ok = "", start, False
        cloud_tokens = 0
        self._pre = self._snapshot()

        with self.meter.measure() as m:
            if feats.doctor_question or feats.guard_question:
                answer, att = self._explain(task, tool_log, "security_check" if feats.guard_question
                                            else "diagnose_system")
                attempts.append(att)
                final_path = Pathway(att.pathway) if att.pathway in ("small", "large", "cloud") else Pathway.TOOL
                ok = True
            elif start is Pathway.TOOL and feats.tool_intent:
                answer = self._call_tool(feats.tool_intent, feats.tool_args, tool_log)
                attempts.append(Attempt("tool", feats.tool_intent, steps=1, ok=True))
                ok = True
            else:
                chain = [p for p in ORDER[ORDER.index(start):] if p in self.backends] \
                    if self.cfg.escalate_on_failure else [start]
                for p in chain:
                    att, answer = self._model_loop(p, task, tool_log)
                    attempts.append(att)
                    if not self.backends[p].is_local:
                        cloud_tokens += att.prompt_tokens + att.completion_tokens
                    final_path = p
                    if att.ok:
                        ok = True
                        break

        verified = checker is not None
        if checker is not None:
            try:
                ok = bool(checker(answer, self.gate.root))
            except Exception:
                ok = False

        cloud_j = cloud_tokens / 1000 * self.cfg.cloud_wh_per_1k_tokens * 3600
        local_j = m.reading.joules
        res = TaskResult(
            id=uuid.uuid4().hex[:12], task=task, router=self.router.name, pathway=final_path.value,
            answer=answer, success=ok, verified=verified,
            self_check=attempts[-1].self_check if attempts else None, seconds=m.reading.seconds,
            energy_local_j=local_j, energy_cloud_est_j=cloud_j,
            energy_total_j=None if local_j is None else local_j + cloud_j,
            energy_sources=m.reading.sources, attempts=attempts, tool_calls=tool_log,
            features=asdict(feats))
        self._log(res)
        return res

    # ---------- internals ----------
    def _call_tool(self, name: str, args: dict, tool_log: list[dict]) -> str:
        t0 = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            out = f"Error: unknown tool '{name}'"
        else:
            try:
                out = tool.fn(**args)
            except PermissionDenied as e:
                out = f"Permission denied: {e}"
            except TypeError as e:
                out = f"Error: bad arguments for {name}: {e}"
            except Exception as e:  # tool bugs must not crash the agent
                out = f"Error: {type(e).__name__}: {e}"
        tool_log.append({"tool": name, "args": args, "ms": round((time.perf_counter() - t0) * 1000, 1),
                         "result_preview": out[:200]})
        return out

    def _model_loop(self, p: Pathway, task: str, tool_log: list[dict]) -> tuple[Attempt, str]:
        be = self.backends[p]
        att = Attempt(p.value, be.name)
        msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": task}]
        t0 = time.perf_counter()
        answer = ""
        log_start = len(tool_log)
        try:
            deadline = t0 + self.cfg.attempt_budget_s
            for _ in range(self.cfg.max_steps):
                remaining = deadline - time.perf_counter()
                if remaining <= 1:
                    att.error = f"time budget exceeded ({self.cfg.attempt_budget_s:.0f}s)"
                    break
                r = be.chat(msgs, self.schemas, timeout=remaining)
                att.steps += 1
                att.prompt_tokens += r.prompt_tokens
                att.completion_tokens += r.completion_tokens
                if not r.tool_calls:
                    answer = r.text.strip()
                    att.ok = bool(answer) and answer != "ESCALATE"
                    if not att.ok:
                        att.error = "escalate" if answer == "ESCALATE" else "empty answer"
                    break
                msgs.append({"role": "assistant", "content": r.text or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.name,
                                                          "arguments": json.dumps(tc.arguments)}}
                                            for tc in r.tool_calls]})
                for tc in r.tool_calls:
                    out = self._call_tool(tc.name, tc.arguments, tool_log)
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            else:
                att.error = "max steps reached"
            if att.ok:
                att.issues = validate(task, tool_log[log_start:], answer, self._pre)
                if att.issues and self.cfg.validate:
                    att.ok = False
                    att.error = "validator: " + ", ".join(att.issues)[:200]
            if att.ok and self.cfg.self_verify:
                self._self_verify(be, task, answer, tool_log[log_start:], att)
        except BackendError as e:
            att.ok = False
            att.error = str(e)[:300]
        att.seconds = time.perf_counter() - t0
        return att, answer

    VERIFY_PROMPT = """Check whether a task was completed EXACTLY as requested.
Task: {task}
Actions taken: {actions}
Current state of files that were written:
{files}
Final answer given to the user: {answer}

Look for: placeholder or generic content, wrong file names, missing items, wrong numbers,
claims not supported by the actions. Reply with PASS or FAIL on the first line, then one short reason."""

    def _self_verify(self, be: Backend, task: str, answer: str, actions: list[dict], att: Attempt) -> None:
        """Re-read what was actually written and ask the model to judge it. FAIL -> escalate."""
        written = {a["args"].get("path") or a["args"].get("dst") for a in actions
                   if a["tool"] in ("write_file", "move_path")}
        files = []
        for p in sorted(x for x in written if x):
            try:
                files.append(f"--- {p} ---\n{self.gate.resolve(p).read_text(errors='replace')[:1500]}")
            except Exception as e:
                files.append(f"--- {p} --- (unreadable: {e})")
        acts = "; ".join(f"{a['tool']}({json.dumps(a['args'])[:120]}) -> {a['result_preview'][:120]}"
                         for a in actions) or "none"
        prompt = self.VERIFY_PROMPT.format(task=task, actions=acts, files="\n".join(files) or "(none)",
                                           answer=answer[:800])
        r = be.chat([{"role": "user", "content": prompt}], [])
        att.verify_tokens = r.prompt_tokens + r.completion_tokens
        att.prompt_tokens += r.prompt_tokens
        att.completion_tokens += r.completion_tokens
        verdict = "FAIL" if r.text.strip().upper().startswith("FAIL") else "PASS"
        att.self_check = verdict
        if verdict == "FAIL":
            att.ok = False
            att.error = "self-check FAIL: " + r.text.strip()[:200]

    def _explain(self, task: str, tool_log: list[dict], tool: str) -> tuple[str, "Attempt"]:
        """Run Doctor or Guard (deterministic), then have a model explain it using only those findings.
        The measured report is always shown, so the explanation can be checked against it."""
        report = self._call_tool(tool, {}, tool_log)
        if tool == "security_check":
            from .guard import EXPLAIN_PROMPT
        else:
            from .doctor import EXPLAIN_PROMPT
        for p in (Pathway.SMALL, Pathway.LARGE, Pathway.CLOUD):
            be = self.backends.get(p)
            if be is None:
                continue
            att = Attempt(p.value, be.name)
            t0 = time.perf_counter()
            try:
                r = be.chat([{"role": "user", "content": EXPLAIN_PROMPT.format(question=task, report=report)}], [])
                att.steps, att.ok = 1, bool(r.text.strip())
                att.prompt_tokens, att.completion_tokens = r.prompt_tokens, r.completion_tokens
                att.seconds = time.perf_counter() - t0
                if att.ok:
                    return f"{r.text.strip()}\n\n--- measurements ---\n{report}", att
            except BackendError as e:
                att.error = str(e)[:200]
        return report, Attempt("tool", tool, steps=1, ok=True)

    def _snapshot(self, limit: int = 5000) -> set[str]:
        out: set[str] = set()
        for p in self.gate.root.rglob("*"):
            if p.is_file():
                out.add(str(p.relative_to(self.gate.root)))
                if len(out) >= limit:
                    break
        return out

    def _log(self, res: TaskResult) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(asdict(res), default=str) + "\n")
