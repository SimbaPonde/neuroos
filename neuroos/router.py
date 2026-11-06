"""Pathway routing: decide WHERE a task runs.

Pathways, cheapest first:
  TOOL   - deterministic tool, no model at all
  SMALL  - small local model
  LARGE  - larger local model
  CLOUD  - remote model (energy estimated, not measured)

Routers implemented:
  FixedRouter      - always one pathway (baselines: always-small / always-large / always-cloud)
  HeuristicRouter  - rules on task features
  LearnedRouter    - k-NN over logged outcomes: picks the cheapest pathway whose
                     predicted success >= threshold (in the spirit of carbon-aware
                     routing, but over multi-step agent tasks and measured joules)

Every router sees the same features, so they can be compared fairly.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Pathway(str, Enum):
    TOOL = "tool"
    SMALL = "small"
    LARGE = "large"
    CLOUD = "cloud"


ORDER = [Pathway.TOOL, Pathway.SMALL, Pathway.LARGE, Pathway.CLOUD]

_MULTISTEP = re.compile(r"\b(then|after that|and also|each|every|all|organi[sz]e|compare|summari[sz]e|"
                        r"refactor|plan|analy[sz]e|why|explain)\b", re.I)
_WRITE = re.compile(r"\b(create|write|save|move|rename|delete|edit|fix|update|organi[sz]e)\b", re.I)


@dataclass
class Features:
    words: int
    multistep_cues: int
    writes: bool
    has_code: bool
    tool_intent: str | None
    tool_args: dict = field(default_factory=dict)
    doctor_question: bool = False     # a question about the computer's health -> Doctor + explanation
    guard_question: bool = False      # a question about security / attacks -> Guard + explanation

    def vector(self) -> list[float]:
        return [math.log1p(self.words), float(self.multistep_cues), float(self.writes),
                float(self.has_code), float(self.tool_intent is not None)]


# Requests answerable with zero model calls. Kept deliberately small and exact.
TOOL_INTENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*(guard|security check|security scan|scan for (threats|malware|viruses|attacks)|"
                r"check (my |the )?(computer'?s? |laptop'?s? |system'?s? )?security)\s*[.!?]?\s*$", re.I),
     "security_check"),
    (re.compile(r"^\s*(diagnose|doctor|health check|system check|check (my|the) (computer|laptop|pc|system|machine))"
                r"( (my|the) (computer|laptop|pc|system|machine))?\s*[.!?]?\s*$", re.I), "diagnose_system"),
    (re.compile(r"^\s*(list|show)( me)? (the )?files\s*(here|in workspace)?\s*\??\s*$", re.I), "list_dir"),
    (re.compile(r"^\s*(system info|what('s| is) my (system|os))\s*\??\s*$", re.I), "system_info"),
    (re.compile(r"^\s*(read|show|open|cat)( me)?( the)?( file)?\s+(?P<path>[\w./-]+\.\w+)\s*$", re.I),
     "read_file"),
    (re.compile(r"^\s*(find|list|show)( me)?( all)?( the)?\s+\.?(?P<ext>[a-z0-9]{1,5})\s+files\s*\??\s*$", re.I),
     "find_files"),
]


_DOCTOR_WORDS = re.compile(
    r"\b(slow|slower|sluggish|lag\w*|freez\w*|hang\w*|frozen|overheat\w*|hot|fans?|noisy|battery|charging|"
    r"storage|disk space|space left|memory|ram|cpu|processor|swap|crash\w*|diagnos\w*|health|"
    r"what'?s using|running out|out of space|disk (is )?full|healthy|performance|services?|errors?)\b", re.I)
_DEVICE = re.compile(r"\b(computer|laptop|pc|system|machine|device|ubuntu|linux|my (disk|drive|ssd))\b", re.I)
_FILE_TOKEN = re.compile(r"[\w-]+\.(?:txt|md|csv|json|py|ini|log|jpe?g|png|mp3|pdf|dat)\b", re.I)


_SECURITY_WORDS = re.compile(
    r"\b(hack\w*|attack\w*|malware|virus\w*|trojans?|backdoors?|intrusions?|intruders?|breach\w*|compromised|"
    r"suspicious|security|secure|rootkits?|spyware|keyloggers?|unauthori[sz]ed|firewall|brute.?force|"
    r"open ports?|ports? (are |is )?open|listening|someone (logged|log|logging) in|who logged in|infected)\b", re.I)


def is_guard_question(task: str) -> bool:
    return not _FILE_TOKEN.search(task) and bool(_SECURITY_WORDS.search(task))


def is_doctor_question(task: str) -> bool:
    if _FILE_TOKEN.search(task) or not _DOCTOR_WORDS.search(task):
        return False
    return bool(_DEVICE.search(task)) or task.strip().endswith("?") or len(task.split()) <= 6


def extract(task: str) -> Features:
    intent, args = None, {}
    for pat, name in TOOL_INTENTS:
        m = pat.match(task)
        if m:
            intent = name
            g = m.groupdict()
            if g.get("path"):
                args = {"path": g["path"]}
                if intent == "read_file" and g["path"].lower().endswith((".docx", ".xlsx", ".xlsm", ".pdf")):
                    intent = "read_document"
            elif g.get("ext"):
                args = {"pattern": f"*.{g['ext'].lower()}"}
            break
    return Features(words=len(task.split()), multistep_cues=len(_MULTISTEP.findall(task)),
                    writes=bool(_WRITE.search(task)), has_code="```" in task or "def " in task,
                    tool_intent=intent, tool_args=args,
                    guard_question=intent is None and is_guard_question(task),
                    doctor_question=intent is None and not is_guard_question(task) and is_doctor_question(task))


class Router:
    name = "router"

    def choose(self, task: str, available: list[Pathway]) -> Pathway:
        raise NotImplementedError

    @staticmethod
    def _first_available(pref: list[Pathway], available: list[Pathway]) -> Pathway:
        for p in pref:
            if p in available:
                return p
        return available[0]


class FixedRouter(Router):
    def __init__(self, pathway: Pathway):
        self.pathway, self.name = pathway, f"always-{pathway.value}"

    def choose(self, task, available):
        f = extract(task)
        if f.tool_intent and Pathway.TOOL in available:
            return Pathway.TOOL
        return self._first_available([self.pathway] + ORDER, available)


class HeuristicRouter(Router):
    name = "heuristic"

    def choose(self, task, available):
        f = extract(task)
        if f.tool_intent:
            return self._first_available([Pathway.TOOL, Pathway.SMALL], available)
        score = f.words + 15 * f.multistep_cues + (20 if f.has_code else 0) + (5 if f.writes else 0)
        if score < 25:
            pref = [Pathway.SMALL, Pathway.LARGE, Pathway.CLOUD]
        elif score < 60:
            pref = [Pathway.LARGE, Pathway.CLOUD, Pathway.SMALL]
        else:
            pref = [Pathway.CLOUD, Pathway.LARGE, Pathway.SMALL]
        return self._first_available(pref, available)


class LearnedRouter(Router):
    """Non-parametric router trained from NeuroOS task logs (JSONL).
    For a new task, find k nearest logged tasks; for each pathway estimate success
    rate and mean energy; pick the lowest-energy pathway with success >= threshold."""

    name = "learned-knn"

    def __init__(self, records: list[dict], k: int = 7, threshold: float = 0.8,
                 fallback: Router | None = None):
        self.k, self.threshold = k, threshold
        self.fallback = fallback or HeuristicRouter()
        self.data = [(extract(r["task"]).vector(), Pathway(r["pathway"]), bool(r["success"]),
                      r.get("energy_total_j") or 0.0) for r in records
                     if r.get("pathway") and r.get("pathway") != "tool"]

    @classmethod
    def from_logs(cls, log_dir: str, **kw) -> "LearnedRouter":
        recs = []
        for f in Path(log_dir).glob("*.jsonl"):
            recs += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        return cls(recs, **kw)

    def choose(self, task, available):
        f = extract(task)
        if f.tool_intent and Pathway.TOOL in available:
            return Pathway.TOOL
        if len(self.data) < self.k:
            return self.fallback.choose(task, available)
        v = f.vector()
        nn = sorted(self.data, key=lambda d: math.dist(v, d[0]))[: self.k * 4]
        best, best_e = None, float("inf")
        counts = Counter(p for _, p, _, _ in nn)
        for p in available:
            rows = [d for d in nn if d[1] == p][: self.k]
            if len(rows) < 2:
                continue
            succ = sum(r[2] for r in rows) / len(rows)
            mean_e = sum(r[3] for r in rows) / len(rows)
            if succ >= self.threshold and mean_e < best_e:
                best, best_e = p, mean_e
        if best is None or not counts:
            return self.fallback.choose(task, available)
        return best


_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "into", "is", "it", "its", "that", "this", "with",
          "for", "on", "be", "by", "as", "are", "what", "which", "how", "many", "my", "me", "each", "every",
          "one", "per", "line", "file", "files", "from", "named", "called", "save", "result", "keep",
          "keeping", "then", "only", "just", "all", "there"}


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9_.']+", text.lower()):
        w = w.strip(".'")
        if not w or w in _STOP:
            continue
        out.append(w)
        if "." in w:                         # 'sales.csv' also yields 'csv'
            out.append(w.rsplit(".", 1)[1])
    return out


class TextKNNRouter(Router):
    """Learned router from past outcomes, by request similarity.

    For a new request: find the k most similar past requests (TF-IDF cosine), estimate for each
    pathway the success probability (similarity-weighted, smoothed toward 0.5) and mean cost,
    then choose the CHEAPEST pathway whose predicted success >= threshold; if none qualifies,
    the pathway with the highest predicted success.

    records: [{"task": text, "pathway": "small"|"large"|..., "success": bool, "cost": float}]
    """

    name = "learned-knn"

    def __init__(self, records: list[dict], k: int = 5, threshold: float = 0.7, alpha: float = 0.5,
                 fallback: Router | None = None):
        self.k, self.threshold, self.alpha = k, threshold, alpha
        self.fallback = fallback or HeuristicRouter()
        stats: dict[str, dict[str, list]] = {}
        for r in records:
            if r["pathway"] in ("tool", "error", None, ""):
                continue
            stats.setdefault(r["task"], {}).setdefault(r["pathway"], []).append(
                (1.0 if r["success"] else 0.0, float(r["cost"])))
        self.stats = stats
        texts = list(stats)
        df = Counter(t for x in texts for t in set(tokens(x)))
        self.idf = {t: math.log((1 + len(texts)) / (1 + c)) + 1 for t, c in df.items()}
        self.vecs = {x: self._vec(x) for x in texts}

    @classmethod
    def from_logs(cls, log_dir: str, **kw) -> "TextKNNRouter":
        recs = []
        for f in Path(log_dir).glob("*.jsonl"):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                cost = r.get("energy_total_j") if r.get("energy_total_j") is not None else r.get("seconds", 0)
                recs.append({"task": r["task"], "pathway": r["pathway"], "success": r["success"], "cost": cost})
        return cls(recs, **kw)

    def _vec(self, text: str) -> dict[str, float]:
        tf = Counter(tokens(text))
        v = {t: c * self.idf.get(t, 1.0) for t, c in tf.items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / n for t, x in v.items()}

    def predict(self, task: str) -> dict[Pathway, tuple[float, float]]:
        q = self._vec(task)
        sims = sorted(((sum(q.get(t, 0) * w for t, w in v.items()), x) for x, v in self.vecs.items()),
                      reverse=True)[: self.k]
        sims = [(s, x) for s, x in sims if s > 0]
        out: dict[Pathway, tuple[float, float]] = {}
        for p in (Pathway.SMALL, Pathway.LARGE, Pathway.CLOUD):
            w_sum = s_sum = c_sum = 0.0
            for s, x in sims:
                obs = self.stats[x].get(p.value)
                if not obs:
                    continue
                w_sum += s
                s_sum += s * sum(o[0] for o in obs) / len(obs)
                c_sum += s * sum(o[1] for o in obs) / len(obs)
            if w_sum > 0:
                out[p] = ((s_sum + self.alpha * 0.5) / (w_sum + self.alpha), c_sum / w_sum)
        return out

    def choose(self, task, available):
        f = extract(task)
        if f.tool_intent and Pathway.TOOL in available:
            return Pathway.TOOL
        pred = {p: v for p, v in self.predict(task).items() if p in available}
        if not pred:
            return self.fallback.choose(task, available)
        ok = [(c, p) for p, (s, c) in pred.items() if s >= self.threshold]
        if ok:
            return min(ok, key=lambda x: x[0])[1]
        return max(pred.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]


def make_router(name: str, log_dir: str | None = None) -> Router:
    if name == "heuristic":
        return HeuristicRouter()
    if name == "learned":
        return TextKNNRouter.from_logs(log_dir) if log_dir else HeuristicRouter()
    if name == "learned-features":
        return LearnedRouter.from_logs(log_dir) if log_dir else HeuristicRouter()
    if name.startswith("always-"):
        return FixedRouter(Pathway(name.split("-", 1)[1]))
    raise ValueError(f"Unknown router: {name}")
