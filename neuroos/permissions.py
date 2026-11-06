"""Deterministic safety layer. The LLM proposes; this code decides.

- Every path the agent touches must resolve inside the workspace.
- Tools are tagged SAFE (read-only) or RISKY (writes, deletes, shell).
- RISKY actions require explicit approval unless auto_approve is set (benchmarks only).
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Callable


class Risk(Enum):
    SAFE = "safe"
    RISKY = "risky"


class PermissionDenied(Exception):
    pass


# Shell patterns refused outright, even with approval.
_BLOCKED = [
    r"\brm\s+-[a-z]*r[a-z]*f?\s+/(\s|$)", r"\bmkfs\b", r"\bdd\s+if=", r":\(\)\s*\{",
    r"\bshutdown\b", r"\breboot\b", r"\bsudo\b", r"\bchmod\s+-R\s+777\s+/",
    r"\bcurl\b.*\|\s*(ba)?sh", r"\bwget\b.*\|\s*(ba)?sh",
]


class PermissionGate:
    def __init__(self, workspace: str, auto_approve: bool = False,
                 ask: Callable[[str], bool] | None = None):
        self.root = Path(workspace).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.auto_approve = auto_approve
        self.ask = ask or self._ask_terminal
        self.denials = 0

    @staticmethod
    def _ask_terminal(prompt: str) -> bool:
        try:
            return input(f"\n[NeuroOS] {prompt}\nAllow? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    def resolve(self, path: str) -> Path:
        p = (self.root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if p != self.root and self.root not in p.parents:
            self.denials += 1
            raise PermissionDenied(f"Path outside workspace: {path}")
        return p

    def check_shell(self, cmd: str) -> None:
        for pat in _BLOCKED:
            if re.search(pat, cmd):
                self.denials += 1
                raise PermissionDenied(f"Blocked command pattern: {cmd}")

    def approve(self, action: str, risk: Risk) -> None:
        if risk is Risk.SAFE or self.auto_approve:
            return
        if not self.ask(action):
            self.denials += 1
            raise PermissionDenied(f"User declined: {action}")
