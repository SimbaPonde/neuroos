# NeuroOS (prototype v0.1)

Talk to your Linux computer in plain language. NeuroOS decides **where** each task should run
(a deterministic tool, a small local model, a larger local model, or the cloud), carries it out
through tools behind a permission gate, and **measures the energy of every task**.

Research question: *how much agent work can run locally at acceptable quality, and what does
an OS-level router save in joules per successfully completed task?*

```
you ──► router ──► TOOL  (no model)
                ├─► SMALL local model ─┐
                ├─► LARGE local model ─┼─► tools (files, shell…) ─► permission gate ─► Linux kernel
                └─► CLOUD (optional) ──┘
          every task ──► energy meter (RAPL / nvidia-smi) ──► JSONL log ──► learned router
```

## Setup (Ubuntu 22.04/24.04, or a Linux VM for development)

```bash
# 1. Local models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b        # SMALL pathway
ollama pull qwen2.5:7b        # LARGE pathway (needs ~6 GB free RAM/VRAM)

# 2. NeuroOS (standard library only; pytest for tests)
cd neuroos
pip install pytest
python -m pytest -q           # 127 tests, no models needed

# 3. Check everything
python -m neuroos --check
```

Optional cloud pathway: `export NEUROOS_CLOUD_API_KEY=...` (any OpenAI-compatible endpoint;
set `NEUROOS_CLOUD_URL` and `NEUROOS_CLOUD_MODEL` to change provider).

## Doctor: diagnose this computer

```bash
python -m neuroos --doctor                          # health check (no AI needed)
python -m neuroos --doctor "why is my laptop slow?"  # health check, explained by the local model
python -m neuroos                                   # or just ask in chat: "is my battery healthy?"
sudo .venv/bin/python -m neuroos --doctor           # with sudo: disk health (SMART) and full system logs
```

Checks CPU, memory and swap, which programs use the most (grouped: "chrome x34, 6 GB"),
disk space, disk health (`sudo apt install smartmontools`), temperatures, battery wear,
failed system services, errors and out-of-memory kills since startup, and uptime.
Fixed thresholds decide what is a problem; the model only explains, using only the
measured findings, and the measurements are always shown under its answer.
Inside WSL, Windows hides temperatures, battery and disk health: run on native Ubuntu for those.

## Guard: security check, explained

```bash
python -m neuroos --guard                              # first run records what is normal on this computer
python -m neuroos --guard                              # later runs report what is new or suspicious
python -m neuroos --guard "has anyone tried to hack me?"  # explained by the local model
python -m neuroos --guard --accept                     # after reviewing: accept the current state as normal
sudo .venv/bin/python -m neuroos --guard               # with sudo: sees root's programs and files too
```

Always flags: programs running from /tmp or from memory only ("fileless"), programs whose file was
deleted, /etc/ld.so.preload (rootkit trick), extra root accounts, setuid programs in temp folders,
listeners on known attack ports (4444, 1337, 31337, ...), and password guessing in the logs.
Compared with your baseline: new programs open to the network, new or changed startup entries
(cron, systemd, autostart, shell profiles), changed sensitive files (/etc/passwd, sudoers, hosts,
SSH config, authorized_keys), new login accounts, new setuid programs.

Fixed rules decide; the model only explains. Everything Guard quotes (process names, log lines) is
given to the model as untrusted data, so text planted by an attacker can't change the verdict, and
the rule-based report is always shown under the explanation. Under WSL Guard sees the Linux side only;
live kernel monitoring (eBPF/auditd) is planned for native Ubuntu.

## Apps: documents, web, AI documents

```bash
pip install -r requirements-apps.txt      # Word, Excel, PDF and Google Docs support
python -m neuroos --check                 # shows which app tools are available
```

Ask in plain English, e.g. "make a Word document summarizing notes.md", "turn sales.csv into an
Excel file", "make a one-page PDF of my plan", "search the web for ...", "read report.pdf".
Documents are created in the workspace and every write asks for approval. Web pages are read-only
and treated as untrusted (their text can't instruct NeuroOS). The web reader doesn't run
JavaScript, so some modern sites return little text.

**AI documents.** Write a line starting with `@neuro` anywhere in a document:

```
@neuro summarize the notes above in 3 bullets
@neuro why is my computer slow?
@neuro has anyone tried to hack me?
```

then run the watcher. Answers appear under each request, starting with `NeuroOS:`.

```bash
python -m neuroos --watch ~/neuroos_workspace/notes.md                       # local file
python -m neuroos --watch "https://docs.google.com/document/d/<id>/edit"     # Google Doc
```

### Google Docs setup (once, about 10 minutes, free)

1. Go to console.cloud.google.com and create a project (e.g. "NeuroOS").
2. APIs & Services -> Library -> enable **Google Docs API**.
3. Google Auth Platform (OAuth consent screen): choose External, fill in an app name and your email,
   and add your own Google account under **Test users**.
4. Clients -> Create client -> **Desktop app** -> Download the JSON file.
5. In Ubuntu:
   `mkdir -p ~/.neuroos/google && cp /mnt/c/Users/<you>/Downloads/client_secret_*.json ~/.neuroos/google/credentials.json`
6. Run `--watch` with your doc link. It prints a sign-in link once: open it, allow access, done.
   The sign-in is saved in `~/.neuroos/google/token.json`. Keep both files private.

## Use

```bash
python -m neuroos                                   # chat
python -m neuroos "organize my downloads by file type"
python -m neuroos --router always-small "..."       # force a pathway
```

The agent can only touch `~/neuroos_workspace` (change with `NEUROOS_WORKSPACE`).
Writes, moves and shell commands ask for approval; `sudo`, `rm -rf /`, `curl | sh` etc. are always blocked.

## Run the experiment

**Benchmark:** 60 tasks in 6 categories (files, data, text, code, multistep, questions), each with
a checker that inspects the real result. Every checker is proven by a test against a reference
solution. Fixed split: 42 train / 18 test, stratified by category.

**Step 1: collect outcomes** (every task on every pathway; several hours on CPU, so run overnight):

```bash
python -m bench.run_bench --routers always-small always-large --reps 3 --idle-seconds 0
# native Linux with energy: sudo .venv/bin/python -m bench.run_bench --routers always-small always-large --reps 3
```
Results are saved after every task. If it stops, continue with `--resume bench/results/<timestamp>`.

**Step 2: evaluate routers on unseen tasks** (seconds, no models rerun: counterfactual replay):

```bash
python -m bench.eval_router --cv 5
```
This compares always-small, always-large, heuristic, the learned router at several success
thresholds, and the oracle, on held-out tasks (fixed split and 5-fold cross-validation).
Output: success rate, cost per success, share routed to the small model, cost relative to oracle.
Saved to `router_eval.csv`.

`bench.analyze` summarizes any single run (`python -m bench.analyze`).
`bench.inspect_task <id> ...` shows exactly what each model did on a task.

**Runtime validators** (`neuroos/validators.py`) flag suspicious results without knowing the
right answer: placeholder paths, ignored tool errors, overwriting a file without reading it,
invalid JSON/Python, no change made when one was asked for, a named output never created,
a named input never read, and "copies" written without reading the source. In normal use a
flagged result counts as a failure and escalates to the next model. `eval_router` reports how
many silent failures they catch and the `cascade+validate` policy, replayed from the logs.

To rerun a few tasks after a fix and combine with an earlier run (later runs replace earlier
outcomes for the same task): `python -m bench.eval_router <old_run> <new_run> --cv 5`.

## Energy measurement: read this before claiming numbers

| Source | Measures | Notes |
|---|---|---|
| RAPL (`/sys/class/powercap`) | CPU package (+ DRAM on some CPUs) | Intel and recent AMD. Needs root. |
| `nvidia-smi` | GPU board power, integrated at 10 Hz | Laptop dGPUs may under-report. |
| Cloud | **Estimated** from tokens × Wh/1k tokens | Never mixed silently; kept in its own column. Report a sensitivity range. |
| None found | `null` | The code never invents a number. |

- Run on **real hardware** for energy results; VMs are for debugging only.
- Measure an idle baseline (the runner does) and report energy **above idle**.
- Plug in, fix screen brightness, close other apps, use repetitions, report confidence intervals.
- Best practice: validate RAPL against an external wall meter (e.g. a USB/AC power meter) once.
- Apple Silicon: not supported yet (`powermetrics` needs sudo and a parser); use Linux first.

## Code map

| File | Role |
|---|---|
| `neuroos/agent.py` | Agent loop, escalation cascade, per-task measurement and logging |
| `neuroos/router.py` | Fixed, Heuristic, feature k-NN, and TextKNN (learned, default) routers |
| `neuroos/backends.py` | Ollama, OpenAI-compatible cloud, and Mock backends (stdlib only) |
| `neuroos/tools.py` | list/read/write/move/find files, shell, system info |
| `neuroos/permissions.py` | Workspace sandbox, risky-action approval, blocked commands |
| `neuroos/energy.py` | RAPL + NVIDIA energy meter, idle baseline |
| `bench/tasks.py` | 60 benchmark tasks, verified checkers, reference solutions, train/test split |
| `bench/eval_router.py` | Main experiment: routers on unseen tasks (fixed split + k-fold CV) |
| `bench/analyze.py` | Per-run summary and oracle |
| `bench/run_bench.py` | Experiment runner and metrics |

## Roadmap (one paper per stage)

1. **Now:** grow `bench/tasks.py` to 100+ verified multi-step tasks; run baselines on one laptop.
2. Learned router vs. heuristic vs. fixed, on held-out tasks; locality term in the objective.
3. **Model residency:** which models stay loaded (Ollama `keep_alive`, `/api/ps`) and its joule cost.
4. OS resource policy: cgroups/CPU affinity, later `sched_ext`, driven by what 1–3 show matters.
5. Shared memory across agents; integrated NeuroOS evaluation.

See `docs/related_work.md` for how this differs from AIOS, AgentOS, HybridInfer and carbon-aware routing.
