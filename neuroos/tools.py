"""Tools the agent can call. Each tool is a plain Python function plus a JSON schema.
Tools go through the PermissionGate; they are the only way the agent touches the system."""

from __future__ import annotations

import fnmatch
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable

from .permissions import PermissionGate, Risk

MAX_OUT = 4000  # characters returned to the model per tool call


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    risk: Risk
    fn: Callable[..., str]

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


def _clip(s: str) -> str:
    return s if len(s) <= MAX_OUT else s[:MAX_OUT] + f"\n...[truncated {len(s) - MAX_OUT} chars]"


def bwrap_available() -> bool:
    if not shutil.which("bwrap"):
        return False
    try:
        r = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "true"],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def sandboxed_cmd(command: str, root) -> list[str]:
    """Whole filesystem read-only, home hidden, only the workspace writable, no network."""
    home = os.path.expanduser("~")
    return ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp", "--tmpfs", home, "--bind", str(root), str(root),
            "--unshare-net", "--die-with-parent", "--chdir", str(root),
            "/bin/sh", "-c", command]


def build_tools(gate: PermissionGate, shell_mode: str = "sandbox") -> dict[str, Tool]:
    root = gate.root
    if shell_mode == "sandbox" and not bwrap_available():
        shell_mode = "off"          # never silently fall back to an unsandboxed shell

    def list_dir(path: str = ".") -> str:
        p = gate.resolve(path)
        if not p.is_dir():
            return f"Not a directory: {path}"
        rows = []
        for e in sorted(p.iterdir()):
            kind = "dir " if e.is_dir() else "file"
            size = e.stat().st_size if e.is_file() else 0
            rows.append(f"{kind}  {size:>10}  {e.relative_to(root)}")
        return _clip("\n".join(rows) or "(empty)")

    def read_file(path: str) -> str:
        p = gate.resolve(path)
        if not p.is_file():
            return f"Not a file: {path}"
        return _clip(p.read_text(errors="replace"))

    def write_file(path: str, content: str) -> str:
        p = gate.resolve(path)
        gate.approve(f"Write {len(content)} chars to {p.relative_to(root)}", Risk.RISKY)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {p.relative_to(root)}"

    def move_path(src: str, dst: str) -> str:
        s, d = gate.resolve(src), gate.resolve(dst)
        gate.approve(f"Move {s.relative_to(root)} -> {d.relative_to(root)}", Risk.RISKY)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        return f"Moved {s.relative_to(root)} -> {d.relative_to(root)}"

    def find_files(pattern: str, path: str = ".") -> str:
        p = gate.resolve(path)
        hits = []
        for dirpath, _, files in os.walk(p):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    hits.append(os.path.relpath(os.path.join(dirpath, f), root))
        return _clip("\n".join(sorted(hits)) or "(no matches)")

    def run_shell(command: str) -> str:
        gate.check_shell(command)
        gate.approve(f"Run shell command ({shell_mode}): {command}", Risk.RISKY)
        if shell_mode == "sandbox":
            r = subprocess.run(sandboxed_cmd(command, root), capture_output=True, text=True, timeout=60,
                               stdin=subprocess.DEVNULL)
        else:
            r = subprocess.run(command, shell=True, cwd=root, capture_output=True, text=True, timeout=60,
                               stdin=subprocess.DEVNULL)
        return _clip(f"exit={r.returncode}\n{r.stdout}{r.stderr}")

    def file_stats(path: str) -> str:
        p = gate.resolve(path)
        if not p.is_file():
            return f"Not a file: {path}"
        text = p.read_text(errors="replace")
        return (f"{path}: lines={len(text.splitlines())} words={len(text.split())} "
                f"chars={len(text)} bytes={p.stat().st_size}")

    def diagnose_system() -> str:
        from .doctor import diagnose, format_report
        return format_report(diagnose(0.5))

    def security_check() -> str:
        from .guard import format_report, security_check as run
        return format_report(run())

    def system_info() -> str:
        du = shutil.disk_usage(root)
        load = os.getloadavg() if hasattr(os, "getloadavg") else ("n/a",)
        return (f"os={platform.system()} {platform.release()} cpu_count={os.cpu_count()} "
                f"load={load} disk_free_gb={du.free / 1e9:.1f}")

    tools = [
        Tool("list_dir", "List files and folders in a workspace directory.",
             _obj({"path": {"type": "string", "description": "Directory relative to workspace"}}, []),
             Risk.SAFE, list_dir),
        Tool("read_file", "Read a text file from the workspace.",
             _obj({"path": {"type": "string"}}, ["path"]), Risk.SAFE, read_file),
        Tool("write_file", "Create or overwrite a text file in the workspace.",
             _obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
             Risk.RISKY, write_file),
        Tool("move_path", "Move or rename a file or folder inside the workspace.",
             _obj({"src": {"type": "string"}, "dst": {"type": "string"}}, ["src", "dst"]),
             Risk.RISKY, move_path),
        Tool("find_files", "Find files by glob pattern, e.g. '*.pdf'.",
             _obj({"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
             Risk.SAFE, find_files),
        Tool("file_stats", "Exact line, word and character counts of a file. Always use this for counting.",
             _obj({"path": {"type": "string"}}, ["path"]), Risk.SAFE, file_stats),
        Tool("system_info", "Basic system information: OS, CPUs, load, free disk.",
             _obj({}, []), Risk.SAFE, system_info),
        Tool("diagnose_system", "Full health check of this computer: CPU, memory, which programs use the most, "
             "disks, disk health, temperatures, battery, failed services, errors. Use for any question about the "
             "computer being slow, hot, full, crashing or unhealthy.",
             _obj({}, []), Risk.SAFE, diagnose_system),
        Tool("security_check", "Security check of this computer: programs open to the network, suspicious "
             "processes, new startup entries, changed system files, admin accounts, failed logins. Use for any "
             "question about hacking, attacks, malware or suspicious activity.",
             _obj({}, []), Risk.SAFE, security_check),
    ]
    from . import apps
    have = apps.available()

    def read_document(path: str) -> str:
        p = gate.resolve(path)
        if not p.is_file():
            return f"Not a file: {path}"
        try:
            return apps.read_document(p)
        except ImportError as e:
            return f"Error: library missing for {p.suffix} ({e.name}); install with pip install -r requirements-apps.txt"

    def _writer(fn, kind):
        def w(path: str, **kw) -> str:
            p = gate.resolve(path)
            gate.approve(f"Create {kind} {p.relative_to(root)}", Risk.RISKY)
            return fn(p, **kw)
        return w

    tools.append(Tool("read_document", "Read the text of a document: Word (.docx), Excel (.xlsx), PDF, or text file.",
                      _obj({"path": {"type": "string"}}, ["path"]), Risk.SAFE, read_document))
    doc_args = _obj({"path": {"type": "string"}, "title": {"type": "string"},
                     "content": {"type": "string", "description": "Text with '# ' headings, '- ' bullets, "
                                                                  "and paragraphs separated by new lines"}},
                    ["path", "content"])
    if have["word"]:
        tools.append(Tool("write_docx", "Create a Word document (.docx).", doc_args, Risk.RISKY,
                          lambda path, content, title="": _writer(apps.write_docx, "Word document")(
                              path, title=title, content=content)))
    if have["pdf_write"]:
        tools.append(Tool("write_pdf", "Create a PDF document.", doc_args, Risk.RISKY,
                          lambda path, content, title="": _writer(apps.write_pdf, "PDF")(
                              path, title=title, content=content)))
    if have["excel"]:
        tools.append(Tool("write_xlsx", "Create an Excel spreadsheet (.xlsx) from CSV text (first row = headers).",
                          _obj({"path": {"type": "string"}, "csv": {"type": "string"},
                                "sheet": {"type": "string"}}, ["path", "csv"]), Risk.RISKY,
                          lambda path, csv, sheet="Sheet1": _writer(apps.write_xlsx, "Excel file")(
                              path, csv_text=csv, sheet=sheet)))
    tools.append(Tool("web_search", "Search the web. Returns titles, links and snippets.",
                      _obj({"query": {"type": "string"}}, ["query"]), Risk.SAFE, apps.web_search))
    tools.append(Tool("web_read", "Open a web page and return its readable text.",
                      _obj({"url": {"type": "string"}}, ["url"]), Risk.SAFE, apps.web_read))

    if shell_mode != "off":
        tools.append(Tool("run_shell", "Run a shell command inside the workspace (asks the user first).",
                          _obj({"command": {"type": "string"}}, ["command"]), Risk.RISKY, run_shell))
    return {t.name: t for t in tools}
