"""NeuroOS benchmark: 60 everyday computer tasks with verified checkers.

Each task has:
  id, category, difficulty, request   - what the user says
  setup(root)                         - creates the starting files
  check(answer, root) -> bool         - inspects the REAL filesystem / answer; never trusts the model
  solve(root) -> str                  - reference solution (used by tests to prove each checker is correct)
  split                               - "train" or "test" (stratified by category; fixed for the paper)

Categories: files, data, text, code, multistep, questions (10 each).
"""

from __future__ import annotations

import ast
import csv
import io
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class BenchTask:
    id: str
    category: str
    difficulty: str          # easy | medium | hard
    request: str
    setup: Callable[[Path], None]
    check: Callable[[str, Path], bool]
    solve: Callable[[Path], str] | None = None
    split: str = field(default="train")


# ---------------------------------------------------------------- helpers
def _files(root: Path, files: dict[str, str | bytes]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        (p.write_bytes if isinstance(content, bytes) else p.write_text)(content)


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _r(root: Path, rel: str) -> str | None:
    p = root / rel
    return p.read_text(errors="replace") if p.is_file() else None


def _has_num(answer: str, n) -> bool:
    """Number appears as a whole token (so '5' doesn't match '15')."""
    a = answer.replace(",", "")
    return re.search(rf"(?<![\d.]){re.escape(str(n))}(?![\d])", a) is not None


def _has_word(answer: str, w: str) -> bool:
    return re.search(rf"\b{re.escape(w)}\b", answer, re.I) is not None


def _lines(text: str | None) -> list[str]:
    return [l.strip() for l in (text or "").splitlines() if l.strip()]


def _exec(root: Path, rel: str) -> dict | None:
    src = _r(root, rel)
    if src is None:
        return None
    ns: dict = {}
    try:
        exec(compile(src, rel, "exec"), ns)
        return ns
    except Exception:
        return None


def _csv_rows(text: str | None) -> list[dict]:
    if not text:
        return []
    return list(csv.DictReader(io.StringIO(text.strip())))


# ---------------------------------------------------------------- fixtures
SALES = "region,month,amount\nnorth,jan,120\nsouth,jan,80\nnorth,feb,150\neast,feb,95\nsouth,mar,60\n"
NOTES = "Meeting notes\n- ship v2 by Friday\n- TODO: email Dana\n- budget ok\n- TODO: book room\n"
DOWNLOADS = {"downloads/report.pdf": "%PDF", "downloads/photo1.jpg": "jpg", "downloads/photo2.png": "png",
             "downloads/song.mp3": "mp3", "downloads/notes.txt": "hi", "downloads/data.csv": "a,b"}
CONFIG = "[server]\nhost = localhost\nport = 8080\ndebug = true\n\n[db]\nname = shop\n"
PEOPLE = "name,age\nAna,31\nBen,25\nChloe,40\n"
SCORES = "16\n4\n42\n8\n23\n15\n"
APPLOG = ("INFO start\nERROR disk full\nWARN slow\nerror: retry failed\nINFO ok\n"
          "Error in module x\nINFO done\n")
INVENTORY = json.dumps({"items": [{"name": "pens", "stock": 40}, {"name": "paper", "stock": 7},
                                  {"name": "stapler", "stock": 12}]}, indent=2)
ORDERS = "item,qty,price\npen,3,1.5\nbook,2,12\nbag,1,30\n"
CONTACTS = ("Team list\nAna - ana@example.com\nBen <ben.k@work.org>\nno email here\n"
            "Chloe: chloe_99@mail.net, backup chloe@example.com\n")
ESSAY = "The colour of the sky is blue. My favourite colour is green.\nColour matters.\n"
REPORT = "# Quarterly Report\n\nIntro text.\n\n## Revenue\nUp 4%.\n\n## Costs\nFlat.\n\n## Outlook\nPositive.\n"
JOURNAL = "2026-01-04 started gym\nmet Sam on 2026-02-11.\nno date\nTrip planned 2026-03-30 and 2026-04-02\n"
MESSY = "first line\n\n\nsecond line\n   \nthird line\n\n"
NAMES = "Ana\nZoe\nMike\nNina\nBen\nOscar\nliam\n"


# ---------------------------------------------------------------- reference-solution helpers
def _mv(root, src, dst):
    d = root / dst
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root / src), str(d))


# ================================================================ FILES (10)
FILES = [
    BenchTask("rename", "files", "easy", "Rename draft.txt to final.txt",
              lambda r: _files(r, {"draft.txt": "x"}),
              lambda a, r: (r / "final.txt").exists() and not (r / "draft.txt").exists(),
              lambda r: (_mv(r, "draft.txt", "final.txt"), "done")[1]),
    BenchTask("create-file", "files", "easy", "Create a file hello.txt containing exactly: Hello NeuroOS",
              lambda r: None,
              lambda a, r: (_r(r, "hello.txt") or "").strip() == "Hello NeuroOS",
              lambda r: (_w(r, "hello.txt", "Hello NeuroOS"), "done")[1]),
    BenchTask("archive-log", "files", "easy", "Create a folder named archive and move old_log.txt into it",
              lambda r: _files(r, {"old_log.txt": "log"}),
              lambda a, r: (r / "archive/old_log.txt").is_file() and not (r / "old_log.txt").exists(),
              lambda r: (_mv(r, "old_log.txt", "archive/old_log.txt"), "done")[1]),
    BenchTask("copy-config", "files", "easy",
              "Make a copy of settings.ini named settings.backup.ini (keep the original)",
              lambda r: _files(r, {"settings.ini": "[a]\nx = 1\n"}),
              lambda a, r: _r(r, "settings.ini") == "[a]\nx = 1\n"
              and (_r(r, "settings.backup.ini") or "").strip() == "[a]\nx = 1",
              lambda r: (_w(r, "settings.backup.ini", "[a]\nx = 1\n"), "done")[1]),
    BenchTask("logs-folder", "files", "medium", "Move all .log files into a folder named logs",
              lambda r: _files(r, {"a.log": "1", "b.log": "2", "readme.md": "r"}),
              lambda a, r: (r / "logs/a.log").is_file() and (r / "logs/b.log").is_file()
              and not list(r.glob("*.log")) and (r / "readme.md").is_file(),
              lambda r: (_mv(r, "a.log", "logs/a.log"), _mv(r, "b.log", "logs/b.log"), "done")[2]),
    BenchTask("jpeg-to-jpg", "files", "medium", "Rename every .jpeg file to use the .jpg extension instead",
              lambda r: _files(r, {"x.jpeg": "1", "y.jpeg": "2", "z.png": "3"}),
              lambda a, r: (r / "x.jpg").is_file() and (r / "y.jpg").is_file()
              and not list(r.glob("*.jpeg")) and (r / "z.png").is_file(),
              lambda r: (_mv(r, "x.jpeg", "x.jpg"), _mv(r, "y.jpeg", "y.jpg"), "done")[2]),
    BenchTask("readme-title", "files", "easy",
              "Create README.md whose first line is exactly: # My Project",
              lambda r: None,
              lambda a, r: (_r(r, "README.md") or "\n").splitlines()[0:1] == ["# My Project"],
              lambda r: (_w(r, "README.md", "# My Project\n"), "done")[1]),
    BenchTask("append-line", "files", "easy",
              "Add the line 'Reviewed' to the end of status.txt, keeping the existing lines",
              lambda r: _files(r, {"status.txt": "Draft\nChecked\n"}),
              lambda a, r: _lines(_r(r, "status.txt")) == ["Draft", "Checked", "Reviewed"],
              lambda r: (_w(r, "status.txt", "Draft\nChecked\nReviewed\n"), "done")[1]),
    BenchTask("organize", "files", "hard",
              "Organize the downloads folder: move images into downloads/images, audio into "
              "downloads/audio, and pdf, txt and csv files into downloads/documents.",
              lambda r: _files(r, DOWNLOADS),
              lambda a, r: all((r / "downloads" / f).is_file() for f in
                               ["images/photo1.jpg", "images/photo2.png", "audio/song.mp3",
                                "documents/report.pdf", "documents/notes.txt", "documents/data.csv"]),
              lambda r: ([_mv(r, f"downloads/{s}", f"downloads/{d}/{s}") for s, d in
                          [("photo1.jpg", "images"), ("photo2.png", "images"), ("song.mp3", "audio"),
                           ("report.pdf", "documents"), ("notes.txt", "documents"),
                           ("data.csv", "documents")]], "done")[1]),
    BenchTask("project-skeleton", "files", "medium",
              "Create the folders project/src, project/tests and project/docs, each containing an empty "
              "file named .keep",
              lambda r: None,
              lambda a, r: all((r / f"project/{d}/.keep").is_file() for d in ("src", "tests", "docs")),
              lambda r: ([_w(r, f"project/{d}/.keep", "") for d in ("src", "tests", "docs")], "done")[1]),
]

# ================================================================ DATA (10)
DATA = [
    BenchTask("csv-max", "data", "medium",
              "In sales.csv, which region had the single largest amount, and what was it?",
              lambda r: _files(r, {"sales.csv": SALES}),
              lambda a, r: _has_word(a, "north") and _has_num(a, 150),
              lambda r: "north, 150"),
    BenchTask("totals-json", "data", "hard",
              "Sum the amount per region in sales.csv and save the result to totals.json "
              "as a JSON object mapping region to total.",
              lambda r: _files(r, {"sales.csv": SALES}),
              lambda a, r: _json_eq(_r(r, "totals.json"), {"north": 270, "south": 140, "east": 95}),
              lambda r: (_w(r, "totals.json", json.dumps({"north": 270, "south": 140, "east": 95})), "ok")[1]),
    BenchTask("csv-rowcount", "data", "easy",
              "How many data rows (not counting the header) are in sales.csv?",
              lambda r: _files(r, {"sales.csv": SALES}),
              lambda a, r: _has_num(a, 5), lambda r: "5"),
    BenchTask("csv-sum", "data", "medium", "What is the total of the amount column in sales.csv?",
              lambda r: _files(r, {"sales.csv": SALES}),
              lambda a, r: _has_num(a, 505), lambda r: "505"),
    BenchTask("csv-filter", "data", "medium",
              "Write the rows of sales.csv where region is north into north.csv, keeping the header row",
              lambda r: _files(r, {"sales.csv": SALES}),
              lambda a, r: _lines(_r(r, "north.csv")) == ["region,month,amount", "north,jan,120",
                                                          "north,feb,150"],
              lambda r: (_w(r, "north.csv", "region,month,amount\nnorth,jan,120\nnorth,feb,150\n"), "ok")[1]),
    BenchTask("csv-to-json", "data", "medium",
              "Convert people.csv into people.json: a JSON list of objects with keys name and age",
              lambda r: _files(r, {"people.csv": PEOPLE}),
              lambda a, r: _people_ok(_r(r, "people.json")),
              lambda r: (_w(r, "people.json", json.dumps([{"name": "Ana", "age": 31}, {"name": "Ben", "age": 25},
                                                          {"name": "Chloe", "age": 40}])), "ok")[1]),
    BenchTask("best-month", "data", "medium",
              "Which month in sales.csv had the highest total amount across all regions?",
              lambda r: _files(r, {"sales.csv": SALES}),
              lambda a, r: _has_word(a, "feb") or _has_word(a, "february"),
              lambda r: "feb"),
    BenchTask("sort-numbers", "data", "easy",
              "scores.txt has one number per line. Save them sorted from smallest to largest in sorted.txt",
              lambda r: _files(r, {"scores.txt": SCORES}),
              lambda a, r: _lines(_r(r, "sorted.txt")) == ["4", "8", "15", "16", "23", "42"],
              lambda r: (_w(r, "sorted.txt", "4\n8\n15\n16\n23\n42\n"), "ok")[1]),
    BenchTask("count-errors", "data", "medium",
              "How many lines in app.log mention an error (ignore upper/lower case)?",
              lambda r: _files(r, {"app.log": APPLOG}),
              lambda a, r: _has_num(a, 3), lambda r: "3"),
    BenchTask("order-totals", "data", "hard",
              "Add a column named total to orders.csv equal to qty times price, and save the result "
              "as orders_total.csv",
              lambda r: _files(r, {"orders.csv": ORDERS}),
              lambda a, r: _orders_ok(_r(r, "orders_total.csv")),
              lambda r: (_w(r, "orders_total.csv",
                            "item,qty,price,total\npen,3,1.5,4.5\nbook,2,12,24\nbag,1,30,30\n"), "ok")[1]),
]


def _json_eq(text, want) -> bool:
    try:
        return json.loads(text or "") == want
    except Exception:
        return False


def _people_ok(text) -> bool:
    try:
        data = json.loads(text or "")
        got = [(d["name"], int(d["age"])) for d in data]
        return got == [("Ana", 31), ("Ben", 25), ("Chloe", 40)]
    except Exception:
        return False


def _orders_ok(text) -> bool:
    rows = _csv_rows(text)
    try:
        return [(r["item"], float(r["total"])) for r in rows] == [("pen", 4.5), ("book", 24.0), ("bag", 30.0)]
    except Exception:
        return False


# ================================================================ TEXT (10)
TEXT = [
    BenchTask("extract-todos", "text", "medium",
              "Read notes.md and write only the TODO items, one per line, into todo.txt",
              lambda r: _files(r, {"notes.md": NOTES}),
              lambda a, r: _todos_ok(_r(r, "todo.txt")),
              lambda r: (_w(r, "todo.txt", "email Dana\nbook room\n"), "ok")[1]),
    BenchTask("count-lines", "text", "easy", "How many lines are in notes.md?",
              lambda r: _files(r, {"notes.md": NOTES}),
              lambda a, r: _has_num(a, 5), lambda r: "5"),
    BenchTask("extract-emails", "text", "medium",
              "Extract every email address from contacts.txt into emails.txt, one per line",
              lambda r: _files(r, {"contacts.txt": CONTACTS}),
              lambda a, r: set(_lines(_r(r, "emails.txt"))) == {"ana@example.com", "ben.k@work.org",
                                                                "chloe_99@mail.net", "chloe@example.com"},
              lambda r: (_w(r, "emails.txt", "ana@example.com\nben.k@work.org\nchloe_99@mail.net\n"
                                             "chloe@example.com\n"), "ok")[1]),
    BenchTask("replace-word", "text", "medium",
              "In essay.txt, replace every 'colour' with 'color' (match the capitalisation) and save it",
              lambda r: _files(r, {"essay.txt": ESSAY}),
              lambda a, r: (_r(r, "essay.txt") or "").strip() == ESSAY.replace("colour", "color").replace("Colour", "Color").strip(),
              lambda r: (_w(r, "essay.txt", ESSAY.replace("colour", "color").replace("Colour", "Color")), "ok")[1]),
    BenchTask("doc-title", "text", "easy", "What is the title on the first line of report.md?",
              lambda r: _files(r, {"report.md": REPORT}),
              lambda a, r: "quarterly report" in a.lower(), lambda r: "Quarterly Report"),
    BenchTask("lowercase", "text", "easy", "Convert all the text in shout.txt to lowercase and save it back",
              lambda r: _files(r, {"shout.txt": "HELLO World\nTHIS Is LOUD\n"}),
              lambda a, r: (_r(r, "shout.txt") or "").strip() == "hello world\nthis is loud",
              lambda r: (_w(r, "shout.txt", "hello world\nthis is loud\n"), "ok")[1]),
    BenchTask("extract-dates", "text", "medium",
              "Extract every date written as YYYY-MM-DD in journal.txt into dates.txt, one per line, "
              "in the order they appear",
              lambda r: _files(r, {"journal.txt": JOURNAL}),
              lambda a, r: _lines(_r(r, "dates.txt")) == ["2026-01-04", "2026-02-11", "2026-03-30", "2026-04-02"],
              lambda r: (_w(r, "dates.txt", "2026-01-04\n2026-02-11\n2026-03-30\n2026-04-02\n"), "ok")[1]),
    BenchTask("word-count", "text", "easy", "How many words are in essay.txt?",
              lambda r: _files(r, {"essay.txt": ESSAY}),
              lambda a, r: _has_num(a, len(ESSAY.split())), lambda r: str(len(ESSAY.split()))),
    BenchTask("list-headings", "text", "medium",
              "Write the section headings of report.md (the lines starting with '## ', without the '## ') "
              "into headings.txt, one per line",
              lambda r: _files(r, {"report.md": REPORT}),
              lambda a, r: _lines(_r(r, "headings.txt")) == ["Revenue", "Costs", "Outlook"],
              lambda r: (_w(r, "headings.txt", "Revenue\nCosts\nOutlook\n"), "ok")[1]),
    BenchTask("remove-blank", "text", "easy",
              "Remove the blank lines from messy.txt and save the result as clean.txt",
              lambda r: _files(r, {"messy.txt": MESSY}),
              lambda a, r: _blank_ok(_r(r, "clean.txt")),
              lambda r: (_w(r, "clean.txt", "first line\nsecond line\nthird line\n"), "ok")[1]),
]


def _todos_ok(text) -> bool:
    t = (text or "").lower()
    return "email dana" in t and "book room" in t and "budget" not in t and "ship" not in t


# ================================================================ CODE (10)
CODE = [
    BenchTask("fix-bug", "code", "medium",
              "stats.py has a bug in average(). Fix it so it returns the correct mean.",
              lambda r: _files(r, {"stats.py": "def average(xs):\n    return sum(xs) / len(xs) + 1\n"}),
              lambda a, r: _call(r, "stats.py", "average", [2, 4, 6]) == 4,
              lambda r: (_w(r, "stats.py", "def average(xs):\n    return sum(xs) / len(xs)\n"), "ok")[1]),
    BenchTask("fix-syntax", "code", "easy",
              "greet.py has a syntax error. Fix it so greet('Ana') returns 'Hello, Ana'",
              lambda r: _files(r, {"greet.py": "def greet(name)\n    return 'Hello, ' + name\n"}),
              lambda a, r: _call(r, "greet.py", "greet", "Ana") == "Hello, Ana",
              lambda r: (_w(r, "greet.py", "def greet(name):\n    return 'Hello, ' + name\n"), "ok")[1]),
    BenchTask("write-add", "code", "easy",
              "Create add.py with a function add(a, b) that returns a + b",
              lambda r: None,
              lambda a, r: _call(r, "add.py", "add", 2, 3) == 5 and _call(r, "add.py", "add", -1, 1) == 0,
              lambda r: (_w(r, "add.py", "def add(a, b):\n    return a + b\n"), "ok")[1]),
    BenchTask("fix-even", "code", "easy", "In utils.py, is_even() gives wrong answers. Fix it.",
              lambda r: _files(r, {"utils.py": "def is_even(n):\n    return n % 2 == 1\n"}),
              lambda a, r: _call(r, "utils.py", "is_even", 4) is True and _call(r, "utils.py", "is_even", 7) is False,
              lambda r: (_w(r, "utils.py", "def is_even(n):\n    return n % 2 == 0\n"), "ok")[1]),
    BenchTask("palindrome", "code", "medium",
              "Create pal.py with is_palindrome(s) that returns True if s reads the same backwards, "
              "ignoring case",
              lambda r: None,
              lambda a, r: _call(r, "pal.py", "is_palindrome", "Level") is True
              and _call(r, "pal.py", "is_palindrome", "hello") is False,
              lambda r: (_w(r, "pal.py", "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]\n"), "ok")[1]),
    BenchTask("rename-func", "code", "medium",
              "In shop.py, rename the function calc to calculate_total everywhere it is defined or used",
              lambda r: _files(r, {"shop.py": "def calc(p, q):\n    return p * q\n\nresult = calc(2, 5)\n"}),
              lambda a, r: _call(r, "shop.py", "calculate_total", 3, 4) == 12
              and "calc(" not in (_r(r, "shop.py") or "").replace("calculate_total(", "")
              and (_exec(r, "shop.py") or {}).get("result") == 10,
              lambda r: (_w(r, "shop.py", "def calculate_total(p, q):\n    return p * q\n\n"
                                          "result = calculate_total(2, 5)\n"), "ok")[1]),
    BenchTask("add-docstring", "code", "easy",
              "Add a docstring to the function area in geometry.py (don't change what it does)",
              lambda r: _files(r, {"geometry.py": "def area(w, h):\n    return w * h\n"}),
              lambda a, r: _has_docstring(_r(r, "geometry.py"), "area") and _call(r, "geometry.py", "area", 2, 3) == 6,
              lambda r: (_w(r, "geometry.py", 'def area(w, h):\n    """Rectangle area."""\n    return w * h\n'), "ok")[1]),
    BenchTask("fizzbuzz", "code", "medium",
              "Create fizzbuzz.py with fizzbuzz(n) that returns 'FizzBuzz' if n is divisible by 15, 'Fizz' "
              "if by 3, 'Buzz' if by 5, otherwise str(n)",
              lambda r: None,
              lambda a, r: [_call(r, "fizzbuzz.py", "fizzbuzz", n) for n in (3, 5, 15, 7)]
              == ["Fizz", "Buzz", "FizzBuzz", "7"],
              lambda r: (_w(r, "fizzbuzz.py", "def fizzbuzz(n):\n    if n % 15 == 0: return 'FizzBuzz'\n"
                                              "    if n % 3 == 0: return 'Fizz'\n    if n % 5 == 0: return 'Buzz'\n"
                                              "    return str(n)\n"), "ok")[1]),
    BenchTask("fix-max", "code", "medium",
              "biggest.py's biggest(xs) should return the largest number in a list but fails for "
              "negative numbers. Fix it.",
              lambda r: _files(r, {"biggest.py": "def biggest(xs):\n    m = 0\n    for x in xs:\n"
                                                 "        if x > m:\n            m = x\n    return m\n"}),
              lambda a, r: _call(r, "biggest.py", "biggest", [-5, -2, -9]) == -2
              and _call(r, "biggest.py", "biggest", [1, 7, 3]) == 7,
              lambda r: (_w(r, "biggest.py", "def biggest(xs):\n    return max(xs)\n"), "ok")[1]),
    BenchTask("count-vowels", "code", "easy",
              "Create vowels.py with count_vowels(s) returning how many of a, e, i, o, u (any case) are in s",
              lambda r: None,
              lambda a, r: _call(r, "vowels.py", "count_vowels", "Hello World") == 3
              and _call(r, "vowels.py", "count_vowels", "AEIOU") == 5,
              lambda r: (_w(r, "vowels.py", "def count_vowels(s):\n    return sum(c in 'aeiou' for c in s.lower())\n"), "ok")[1]),
]


def _call(root, rel, fn, *args):
    ns = _exec(root, rel)
    try:
        return ns[fn](*args) if ns and fn in ns else None
    except Exception:
        return None


def _has_docstring(src, fn) -> bool:
    try:
        for node in ast.walk(ast.parse(src or "")):
            if isinstance(node, ast.FunctionDef) and node.name == fn:
                return bool(ast.get_docstring(node))
    except SyntaxError:
        pass
    return False


# ================================================================ MULTISTEP (10)
MULTI = [
    BenchTask("todo-count", "multistep", "medium",
              "Read notes.md, count the TODO items, and write just that number into todo_count.txt",
              lambda r: _files(r, {"notes.md": NOTES}),
              lambda a, r: (_r(r, "todo_count.txt") or "").strip() == "2",
              lambda r: (_w(r, "todo_count.txt", "2\n"), "ok")[1]),
    BenchTask("txt-index", "multistep", "medium",
              "Find every .txt file in the workspace (including subfolders) and write their file names, "
              "sorted alphabetically, one per line, into index.md",
              lambda r: _files(r, {"b.txt": "1", "notes/a.txt": "2", "c.md": "3", "x/y/z.txt": "4"}),
              lambda a, r: _index_ok(_lines(_r(r, "index.md"))),
              lambda r: (_w(r, "index.md", "a.txt\nb.txt\nz.txt\n"), "ok")[1]),
    BenchTask("monthly-csv", "multistep", "hard",
              "Read sales.csv, total the amount for each month, and write monthly.csv with the header "
              "month,total and one row per month",
              lambda r: _files(r, {"sales.csv": SALES}),
              lambda a, r: {(x.get("month", "").strip().lower(), _num(x.get("total"))) for x in _csv_rows(_r(r, "monthly.csv"))}
              == {("jan", 200), ("feb", 245), ("mar", 60)},
              lambda r: (_w(r, "monthly.csv", "month,total\njan,200\nfeb,245\nmar,60\n"), "ok")[1]),
    BenchTask("combine-parts", "multistep", "easy",
              "Combine part1.txt and part2.txt, in that order, into full.txt",
              lambda r: _files(r, {"part1.txt": "alpha\nbeta\n", "part2.txt": "gamma\n"}),
              lambda a, r: _lines(_r(r, "full.txt")) == ["alpha", "beta", "gamma"],
              lambda r: (_w(r, "full.txt", "alpha\nbeta\ngamma\n"), "ok")[1]),
    BenchTask("big-files", "multistep", "hard",
              "Move every file larger than 100 bytes into a folder named big (leave smaller files where they are)",
              lambda r: _files(r, {"small1.txt": "x" * 10, "large1.txt": "y" * 500, "large2.dat": "z" * 300,
                                   "small2.txt": "w" * 99}),
              lambda a, r: (r / "big/large1.txt").is_file() and (r / "big/large2.dat").is_file()
              and (r / "small1.txt").is_file() and (r / "small2.txt").is_file()
              and not (r / "large1.txt").exists(),
              lambda r: (_mv(r, "large1.txt", "big/large1.txt"), _mv(r, "large2.dat", "big/large2.dat"), "ok")[2]),
    BenchTask("change-port", "multistep", "medium",
              "In config.ini change the port from 8080 to 9090 and keep every other line exactly the same",
              lambda r: _files(r, {"config.ini": CONFIG}),
              lambda a, r: (_r(r, "config.ini") or "").strip() == CONFIG.replace("8080", "9090").strip(),
              lambda r: (_w(r, "config.ini", CONFIG.replace("8080", "9090")), "ok")[1]),
    BenchTask("backup-md", "multistep", "medium",
              "Create a folder named backup and copy every .md file into it, keeping the originals",
              lambda r: _files(r, {"a.md": "A", "b.md": "B", "c.txt": "C"}),
              lambda a, r: _r(r, "backup/a.md") == "A" and _r(r, "backup/b.md") == "B"
              and (r / "a.md").is_file() and (r / "b.md").is_file() and not (r / "backup/c.txt").exists(),
              lambda r: (_w(r, "backup/a.md", "A"), _w(r, "backup/b.md", "B"), "ok")[2]),
    BenchTask("mark-done", "multistep", "medium",
              "In todo.txt, mark 'buy milk' as done by adding ' [x]' to the end of that line only",
              lambda r: _files(r, {"todo.txt": "call mom\nbuy milk\npay rent\n"}),
              lambda a, r: _lines(_r(r, "todo.txt")) == ["call mom", "buy milk [x]", "pay rent"],
              lambda r: (_w(r, "todo.txt", "call mom\nbuy milk [x]\npay rent\n"), "ok")[1]),
    BenchTask("data-report", "multistep", "hard",
              "Look in the data folder and create report.txt stating how many files it contains and the "
              "name of the largest one",
              lambda r: _files(r, {"data/a.csv": "1" * 50, "data/b.csv": "2" * 400, "data/c.json": "3" * 120,
                                   "data/d.txt": "4"}),
              lambda a, r: _has_num(_r(r, "report.txt") or "", 4) and "b.csv" in (_r(r, "report.txt") or ""),
              lambda r: (_w(r, "report.txt", "Files: 4\nLargest: b.csv\n"), "ok")[1]),
    BenchTask("split-names", "multistep", "hard",
              "Split names.txt into a_to_m.txt (names starting with A to M, any case) and n_to_z.txt "
              "(the rest), keeping one name per line",
              lambda r: _files(r, {"names.txt": NAMES}),
              lambda a, r: set(_lines(_r(r, "a_to_m.txt"))) == {"Ana", "Mike", "Ben", "liam"}
              and set(_lines(_r(r, "n_to_z.txt"))) == {"Zoe", "Nina", "Oscar"},
              lambda r: (_w(r, "a_to_m.txt", "Ana\nMike\nBen\nliam\n"), _w(r, "n_to_z.txt", "Zoe\nNina\nOscar\n"), "ok")[2]),
]


def _index_ok(lines: list[str]) -> bool:
    """Accept bare names sorted, or relative paths sorted (both are reasonable readings of the request)."""
    names = [Path(l.lstrip("-* ").strip()).name for l in lines]
    return sorted(names) == ["a.txt", "b.txt", "z.txt"] and (lines == sorted(lines) or names == sorted(names))


def _blank_ok(text) -> bool:
    """No empty lines left; a whitespace-only line may count as blank or not (both accepted)."""
    if text is None:
        return False
    lines = text.rstrip("\n").split("\n")
    return "" not in lines and [l for l in lines if l.strip()] == ["first line", "second line", "third line"]


def _num(x):
    try:
        f = float(x)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return None


# ================================================================ QUESTIONS (10)
QUESTIONS = [
    BenchTask("list-files", "questions", "easy", "list files",
              lambda r: _files(r, {"a.txt": "1", "b.txt": "2"}),
              lambda a, r: "a.txt" in a and "b.txt" in a, lambda r: "a.txt b.txt"),
    BenchTask("find-pdfs", "questions", "easy", "Which PDF files are in my workspace?",
              lambda r: _files(r, DOWNLOADS),
              lambda a, r: "report.pdf" in a, lambda r: "report.pdf"),
    BenchTask("largest-file", "questions", "easy", "Which file in the workspace is the largest?",
              lambda r: _files(r, {"tiny.txt": "a", "video.bin": "v" * 900, "photo.jpg": "p" * 300}),
              lambda a, r: "video.bin" in a, lambda r: "video.bin"),
    BenchTask("count-py", "questions", "medium",
              "How many .py files are in the project folder, including its subfolders?",
              lambda r: _files(r, {"project/main.py": "", "project/lib/util.py": "", "project/lib/io.py": "",
                                   "project/README.md": "", "other.py": ""}),
              lambda a, r: _has_num(a, 3), lambda r: "3"),
    BenchTask("has-debug", "questions", "easy",
              "Does config.ini have a setting called debug? Answer yes or no.",
              lambda r: _files(r, {"config.ini": CONFIG}),
              lambda a, r: re.match(r"\W*yes\b", a, re.I) is not None or (_has_word(a, "yes") and not _has_word(a, "no")),
              lambda r: "yes"),
    BenchTask("which-port", "questions", "easy", "What port is set in config.ini?",
              lambda r: _files(r, {"config.ini": CONFIG}),
              lambda a, r: _has_num(a, 8080), lambda r: "8080"),
    BenchTask("busiest-folder", "questions", "medium",
              "Which folder has the most files in it: alpha, beta or gamma?",
              lambda r: _files(r, {"alpha/1": "", "alpha/2": "", "beta/1": "", "beta/2": "", "beta/3": "",
                                   "beta/4": "", "gamma/1": ""}),
              lambda a, r: _has_word(a, "beta") and (not (_has_word(a, "alpha") or _has_word(a, "gamma"))
                                                     or _has_num(a, 4)),
              lambda r: "beta"),
    BenchTask("pkg-version", "questions", "easy", "What version is listed in package.json?",
              lambda r: _files(r, {"package.json": json.dumps({"name": "app", "version": "2.4.1"})}),
              lambda a, r: "2.4.1" in a, lambda r: "2.4.1"),
    BenchTask("lowest-stock", "questions", "medium",
              "In inventory.json, which item has the lowest stock?",
              lambda r: _files(r, {"inventory.json": INVENTORY}),
              lambda a, r: _has_word(a, "paper"), lambda r: "paper"),
    BenchTask("avg-score", "questions", "medium",
              "What is the average of the numbers in scores.txt?",
              lambda r: _files(r, {"scores.txt": SCORES}),
              lambda a, r: _has_num(a, 18) or _has_num(a, "18.0"), lambda r: "18"),
]

TASKS: list[BenchTask] = FILES + DATA + TEXT + CODE + MULTI + QUESTIONS

# Fixed, stratified split: in each category, every 3rd task (positions 2, 5, 8) is test -> 18 test, 42 train.
for _cat in (FILES, DATA, TEXT, CODE, MULTI, QUESTIONS):
    for _i, _t in enumerate(_cat):
        _t.split = "test" if _i % 3 == 2 else "train"

assert len({t.id for t in TASKS}) == len(TASKS), "duplicate task ids"


def export_csv(path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "difficulty", "split", "request"])
        for t in TASKS:
            w.writerow([t.id, t.category, t.difficulty, t.split, t.request])
