"""Runtime result validators: catch silent failures WITHOUT knowing the correct answer.

A model often says "done" when it isn't. These checks use only what the OS can see:
the request text, which files existed beforehand, the tool calls the model made and their
results, and the files it wrote. Each rule came from a failure observed in real runs.

  issues = validate(request, tool_calls, answer, preexisting)
  -> [] if nothing suspicious, else a list of short issue codes with details

Rules (all generic: no task-specific knowledge):
  placeholder_path   a tool argument contains a template path such as 'path/to/...'
  tool_error         a file-changing tool failed and was never retried successfully
  blind_overwrite    an existing file was overwritten without being read first
  invalid_json       a .json file was written with content that is not valid JSON
  invalid_python     a .py file was written with content that does not compile
  no_action          the request asks for a change but no change was made
  missing_output     a new file named in the request was never created
  unread_input       an existing file named in the request was never looked at
  unread_source      wrote a file with the same name as an existing file it never read (fake copy)
"""

from __future__ import annotations

import json
import re

WRITERS = {"write_file", "write_docx", "write_pdf", "write_xlsx"}
MUTATING = WRITERS | {"move_path", "run_shell"}
READERS = {"read_file", "read_document", "file_stats", "run_shell"}
LOOKING = READERS | {"list_dir", "find_files", "move_path"}
_FILE = re.compile(r"(?<![\w/.-])([\w./-]*[\w-]\.(?:txt|md|csv|json|py|ini|log|jpe?g|png|mp3|pdf|dat|cfg|yaml|yml))\b",
                   re.I)
_WRITE_INTENT = re.compile(r"\b(create|write|save|move|rename|copy|add|append|remove|delete|fix|convert|replace|"
                           r"change|split|combine|mark|organi[sz]e|sort|make|extract|rewrite|update|edit)\b", re.I)
_SHELL_WRITES = re.compile(r"(>|\bmv\b|\bcp\b|\bmkdir\b|\btouch\b|\bsed\s+-i|\brm\b|\btee\b)")


def _err(tool: str, result: str) -> bool:
    r = (result or "").lstrip()
    if tool == "run_shell":
        m = re.match(r"exit=(-?\d+)", r)
        return (m is not None and m.group(1) != "0") or r.startswith("Error")
    return r.startswith(("Error", "Not a file", "Not a directory", "Permission denied", "User declined"))


def _paths(call: dict) -> list[str]:
    a = call.get("args") or {}
    out = [str(a[k]) for k in ("path", "src", "dst") if a.get(k)]
    if call["tool"] == "run_shell":
        out += _FILE.findall(str(a.get("command", "")))
    return out


def _norm(p: str) -> str:
    return p.strip().lstrip("./")


def validate(request: str, tool_calls: list[dict], answer: str = "", preexisting: set[str] | None = None) -> list[str]:
    pre = {_norm(p) for p in (preexisting or set())}
    pre_names = {p.rsplit("/", 1)[-1] for p in pre}
    issues: list[str] = []

    # placeholder paths
    for c in tool_calls:
        blob = json.dumps(c.get("args") or {})
        if re.search(r"path/to/|/your/|<[a-z_]+>", blob, re.I):
            issues.append(f"placeholder_path:{c['tool']}")
            break

    # mutating tool errors not followed by a successful call of the same tool
    for i, c in enumerate(tool_calls):
        if c["tool"] in MUTATING and _err(c["tool"], c.get("result_preview", "")):
            later_ok = any(d["tool"] == c["tool"] and not _err(d["tool"], d.get("result_preview", ""))
                           for d in tool_calls[i + 1:])
            if not later_ok:
                issues.append(f"tool_error:{c['tool']}")
                break

    # existing file overwritten without being read
    seen: set[str] = set()
    for c in tool_calls:
        if c["tool"] in WRITERS:
            p = _norm(str((c.get("args") or {}).get("path", "")))
            if p in pre and p not in seen:
                issues.append(f"blind_overwrite:{p}")
                break
        if c["tool"] in READERS:
            for p in _paths(c):
                seen.add(_norm(p))

    # 'copy' written from imagination: new path shares a name with an existing, never-read file
    read_names = {p.rsplit("/", 1)[-1] for c in tool_calls if c["tool"] in READERS
                  for p in map(_norm, _paths(c))}
    for c in tool_calls:
        if c["tool"] in WRITERS:
            p = _norm(str((c.get("args") or {}).get("path", "")))
            name = p.rsplit("/", 1)[-1]
            if p not in pre and name in pre_names and name not in read_names:
                issues.append(f"unread_source:{name}")
                break

    # written content that is syntactically invalid
    last_written: dict[str, str] = {}
    for c in tool_calls:
        if c["tool"] == "write_file":
            a = c.get("args") or {}
            last_written[_norm(str(a.get("path", "")))] = str(a.get("content", ""))
    for p, content in last_written.items():
        if p.endswith(".json"):
            try:
                json.loads(content)
            except Exception:
                issues.append(f"invalid_json:{p}")
        elif p.endswith(".py"):
            try:
                compile(content, p, "exec")
            except SyntaxError:
                issues.append(f"invalid_python:{p}")

    # change requested, nothing changed
    changed = any(
        (c["tool"] in WRITERS | {"move_path"} and not _err(c["tool"], c.get("result_preview", "")))
        or (c["tool"] == "run_shell" and _SHELL_WRITES.search(str((c.get("args") or {}).get("command", "")))
            and not _err("run_shell", c.get("result_preview", "")))
        for c in tool_calls)
    question = request.strip().endswith("?")
    if _WRITE_INTENT.search(request) and not question and not changed:
        issues.append("no_action")

    # files named in the request
    named = {_norm(f) for f in _FILE.findall(request)}
    touched = {_norm(p) for c in tool_calls for p in _paths(c)}
    touched_names = {p.rsplit("/", 1)[-1] for p in touched}
    listed = any(c["tool"] in ("list_dir", "find_files") for c in tool_calls)
    for f in sorted(named):
        name = f.rsplit("/", 1)[-1]
        exists_before = f in pre or name in pre_names
        if exists_before:
            if f not in touched and name not in touched_names and not (listed and question):
                issues.append(f"unread_input:{f}")
        elif _WRITE_INTENT.search(request) and not question:
            if f not in touched and name not in touched_names:
                issues.append(f"missing_output:{f}")
    return issues
