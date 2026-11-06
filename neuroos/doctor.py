"""NeuroOS Doctor: diagnose the whole computer and explain what's wrong.

Two layers, on purpose:
  1. Collect + rules (this file): read the real system state and apply fixed thresholds.
     Always correct, needs no AI, works offline. Every finding carries its evidence.
  2. Explain (doctor_explain): the local model turns the findings into plain English.
     It is told to use ONLY the findings, so it cannot invent problems.

Reads /proc and /sys directly (standard library only) and uses system commands when present
(systemctl, journalctl, smartctl). Anything unavailable is reported as "not checked" with
the reason, never guessed. WSL is detected: it hides real hardware (temperatures, battery,
disk health), so those checks need native Linux.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

SEVERITY = {"ok": 0, "info": 1, "warn": 2, "critical": 3}
ICON = {"ok": "OK  ", "info": "INFO", "warn": "WARN", "critical": "CRIT"}


@dataclass
class Finding:
    severity: str            # ok | info | warn | critical
    area: str                # cpu | memory | apps | disk | temperature | battery | services | logs | system
    title: str
    evidence: str
    advice: str = ""


@dataclass
class Report:
    status: str
    findings: list[Finding] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    not_checked: dict = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


Runner = Callable[[list[str]], tuple[int, str]]


def run_cmd(cmd: list[str], timeout: float = 10) -> tuple[int, str]:
    if not shutil.which(cmd[0]):
        return 127, f"{cmd[0]} not installed"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def cores_str(cpu_pct: float, cores: int | None = None) -> str:
    """'786% of one core' -> 'about 8 cores (all of them)'."""
    if cpu_pct >= 150:
        n = cpu_pct / 100
        whole = f" (all of them)" if cores and n >= cores - 0.5 else (f" of {cores}" if cores else "")
        return f"about {n:.0f} cores{whole}"
    return f"{cpu_pct:.0f}% of one core"


def gb(n_bytes: float) -> str:
    """Human-readable size: 30 MB stays '30 MB', not '0.0 GB'."""
    if n_bytes >= 1e9:
        return f"{n_bytes / 1e9:.1f} GB"
    if n_bytes >= 1e6:
        return f"{n_bytes / 1e6:.0f} MB"
    return f"{n_bytes / 1e3:.0f} KB"


class Doctor:
    def __init__(self, root: str = "/", runner: Runner = run_cmd, sample_seconds: float = 1.0):
        self.root = Path(root)
        self.run = runner
        self.sample = sample_seconds
        self.facts: dict = {}
        self.not_checked: dict = {}
        self.findings: list[Finding] = []

    # ------------------------------------------------------------------ helpers
    def _p(self, rel: str) -> Path:
        return self.root / rel.lstrip("/")

    def _read(self, rel: str) -> str | None:
        try:
            return self._p(rel).read_text(errors="replace")
        except OSError:
            return None

    def _add(self, *a, **k) -> None:
        self.findings.append(Finding(*a, **k))

    # ------------------------------------------------------------------ main
    def run_all(self) -> Report:
        t0 = time.perf_counter()
        for check in (self.system, self.cpu_and_apps, self.memory, self.disks, self.disk_health,
                      self.temperature, self.battery, self.services, self.logs):
            try:
                check()
            except Exception as e:                     # a broken check must never break the report
                self.not_checked[check.__name__] = f"error: {type(e).__name__}: {e}"
        worst = max((SEVERITY[f.severity] for f in self.findings), default=0)
        status = next(k for k, v in SEVERITY.items() if v == worst)
        self.findings.sort(key=lambda f: -SEVERITY[f.severity])
        return Report(status, self.findings, self.facts, self.not_checked, time.perf_counter() - t0)

    # ------------------------------------------------------------------ checks
    def system(self) -> None:
        ver = self._read("/proc/version") or ""
        wsl = "microsoft" in ver.lower()
        self.facts["wsl"] = wsl
        up = self._read("/proc/uptime")
        if up:
            days = float(up.split()[0]) / 86400
            self.facts["uptime_days"] = round(days, 1)
            if days > 30:
                self._add("info", "system", f"Running for {days:.0f} days without a restart",
                          f"uptime {days:.1f} days",
                          "Restart to apply pending updates and clear accumulated memory leaks.")
        if wsl:
            self.not_checked["hardware"] = ("running inside WSL: temperatures, battery and disk health "
                                            "are hidden by Windows; run on native Linux for these")

    def _cpu_total(self) -> int | None:
        s = self._read("/proc/stat")
        if not s:
            return None
        parts = s.splitlines()[0].split()[1:]
        return sum(int(x) for x in parts[:8])

    def _cpu_idle(self) -> int:
        parts = (self._read("/proc/stat") or "cpu 0").splitlines()[0].split()[1:]
        return int(parts[3]) + (int(parts[4]) if len(parts) > 4 else 0)

    def _procs(self) -> dict[int, tuple[str, int, int]]:
        """pid -> (name, cpu ticks, rss bytes)"""
        out = {}
        proc = self._p("/proc")
        page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
        for d in proc.iterdir() if proc.exists() else []:
            if not d.name.isdigit():
                continue
            try:
                stat = (d / "stat").read_text()
                name = stat[stat.index("(") + 1:stat.rindex(")")]
                rest = stat[stat.rindex(")") + 2:].split()
                ticks = int(rest[11]) + int(rest[12])
                rss = int(rest[21]) * page
                state = rest[0]
                out[int(d.name)] = (name, ticks, rss, state)
            except (OSError, ValueError, IndexError):
                continue
        return out

    def cpu_and_apps(self) -> None:
        cores = os.cpu_count() or 1
        load = self._read("/proc/loadavg")
        if load:
            l1, l5, l15 = (float(x) for x in load.split()[:3])
            self.facts["load_1_5_15"] = [l1, l5, l15]
            self.facts["cores"] = cores
        t1, i1, p1 = self._cpu_total(), self._cpu_idle(), self._procs()
        if t1 is None:
            self.not_checked["cpu"] = "/proc/stat unreadable"
            return
        time.sleep(self.sample)
        t2, i2, p2 = self._cpu_total(), self._cpu_idle(), self._procs()
        if t2 - t1 <= 0:                        # counters didn't move: can't measure, don't guess
            self.not_checked["cpu_usage"] = "CPU counters did not change during the sample"
            return
        dt = t2 - t1
        busy = 100 * (1 - (i2 - i1) / dt)
        self.facts["cpu_busy_pct"] = round(busy, 1)

        # group by program name: "chrome x34" is more useful than 34 rows
        apps: dict[str, dict] = {}
        zombies = 0
        for pid, (name, ticks, rss, state) in p2.items():
            if state == "Z":
                zombies += 1
            prev = p1.get(pid)
            cpu = 100 * cores * (ticks - prev[1]) / dt if prev and prev[0] == name else 0.0
            a = apps.setdefault(name, {"name": name, "procs": 0, "cpu_pct": 0.0, "rss": 0})
            a["procs"] += 1; a["cpu_pct"] += cpu; a["rss"] += rss
        top_cpu = sorted(apps.values(), key=lambda a: -a["cpu_pct"])[:5]
        top_mem = sorted(apps.values(), key=lambda a: -a["rss"])[:5]
        self.facts["top_cpu"] = [{**a, "cpu_pct": round(a["cpu_pct"], 1), "rss": gb(a["rss"])} for a in top_cpu]
        self.facts["top_memory"] = [{**a, "cpu_pct": round(a["cpu_pct"], 1), "rss": gb(a["rss"])} for a in top_mem]

        if busy > 85:
            hog = top_cpu[0] if top_cpu else None
            who = f"; biggest user: {hog['name']} (using {cores_str(hog['cpu_pct'], cores)})" if hog else ""
            self._add("warn", "cpu", "CPU is almost fully busy", f"{busy:.0f}% busy{who}",
                      "Close or pause the program using the most CPU; if it's unexpected, check what it is.")
        elif load and l5 / cores > 1.5:
            self._add("warn", "cpu", "More work queued than the CPU can handle",
                      f"5-minute load {l5:.1f} on {cores} cores",
                      "Programs are waiting for CPU time; close heavy background tasks.")
        for a in top_cpu[:1]:
            if a["cpu_pct"] > 50 * 1 and busy <= 85:
                self._add("info", "apps", f"{a['name']} is using a lot of CPU",
                          f"{cores_str(a['cpu_pct'], cores)} across {a['procs']} process(es)",
                          "Normal for builds, video or AI models; otherwise it may be stuck.")
        if zombies > 5:
            self._add("info", "system", f"{zombies} zombie processes", "finished processes not cleaned up by their parent",
                      "Usually harmless; restarting the parent program clears them.")

    def memory(self) -> None:
        mi = self._read("/proc/meminfo")
        if not mi:
            self.not_checked["memory"] = "/proc/meminfo unreadable"
            return
        m = {k: int(v.split()[0]) * 1024 for k, v in (l.split(":", 1) for l in mi.splitlines() if ":" in l)}
        total, avail = m.get("MemTotal", 0), m.get("MemAvailable", m.get("MemFree", 0))
        st, sf = m.get("SwapTotal", 0), m.get("SwapFree", 0)
        used_pct = 100 * (1 - avail / total) if total else 0
        self.facts["memory"] = {"total": gb(total), "available": gb(avail), "used_pct": round(used_pct, 1),
                                "swap_used": gb(st - sf), "swap_total": gb(st)}
        top = (self.facts.get("top_memory") or [{}])[0]
        hog = f"; biggest user: {top['name']} ({top['rss']}, {top['procs']} process(es))" if top.get("name") else ""
        if total and avail / total < 0.05:
            self._add("critical", "memory", "Almost out of memory", f"only {gb(avail)} of {gb(total)} free{hog}",
                      "Close the biggest program now; the system may freeze or start killing programs.")
        elif total and avail / total < 0.15:
            self._add("warn", "memory", "Memory is running low", f"{gb(avail)} of {gb(total)} free{hog}",
                      "Close programs or browser tabs you don't need.")
        if st and (st - sf) / st > 0.5:
            self._add("warn", "memory", "Heavy use of swap (disk used as memory)",
                      f"{gb(st - sf)} of {gb(st)} swap in use",
                      "This makes everything slow. Free memory, or add RAM if it happens often.")

    def _mounts(self) -> list[tuple[str, str, str]]:
        keep = {"ext4", "ext3", "xfs", "btrfs", "vfat", "exfat", "ntfs", "ntfs3", "fuseblk", "9p", "drvfs",
                "zfs", "f2fs", "overlay"}
        seen, out = set(), []
        for line in (self._read("/proc/mounts") or "").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            dev, mnt, fs = parts[0], parts[1].replace("\\040", " "), parts[2]
            opts = parts[3].split(",") if len(parts) > 3 else []
            if "ro" in opts and mnt != "/":          # read-only images can't be "cleaned up"; not a user problem
                continue
            if fs not in keep or dev in seen or mnt.startswith(("/snap", "/boot/efi", "/usr/lib/wsl", "/proc", "/sys")):
                continue
            if fs == "overlay" and mnt != "/":
                continue
            seen.add(dev)
            out.append((dev, mnt, fs))
        return out

    def disks(self) -> None:
        rows = []
        for dev, mnt, fs in self._mounts():
            try:
                u = shutil.disk_usage(self._p(mnt) if str(self.root) != "/" else mnt)
            except OSError:
                continue
            if u.total < 1e9:                        # tiny system partitions
                continue
            pct = 100 * u.used / u.total if u.total else 0
            label = f"{mnt} (Windows drive)" if fs in ("9p", "drvfs") else mnt
            rows.append({"mount": label, "used_pct": round(pct, 1), "free": gb(u.free), "total": gb(u.total)})
            if pct > 97:
                self._add("critical", "disk", f"Disk {label} is full", f"{pct:.0f}% used, {gb(u.free)} free",
                          "Free space now: programs fail to save and the system can stop working.")
            elif pct > 90:
                self._add("warn", "disk", f"Disk {label} is nearly full", f"{pct:.0f}% used, {gb(u.free)} free",
                          "Delete large unused files, empty the trash, clear caches (e.g. old downloads).")
        self.facts["disks"] = rows
        if not rows:
            self.not_checked["disks"] = "no mounted disks found"

    def disk_health(self) -> None:
        if self.facts.get("wsl"):
            return
        if not shutil.which("smartctl"):
            self.not_checked["disk_health"] = "install smartmontools (sudo apt install smartmontools)"
            return
        code, out = self.run(["smartctl", "--scan"])
        devs = [l.split()[0] for l in out.splitlines() if l.startswith("/dev/")]
        if not devs:
            self.not_checked["disk_health"] = "no disks found by smartctl (needs root: run with sudo)"
            return
        res = {}
        for d in devs:
            code, out = self.run(["smartctl", "-H", d])
            if "PASSED" in out or ": OK" in out:
                res[d] = "healthy"
            elif "FAILED" in out:
                res[d] = "FAILING"
                self._add("critical", "disk", f"Disk {d} reports it is failing", "SMART health check: FAILED",
                          "Back up your files immediately and replace the disk.")
            else:
                res[d] = "unknown (run with sudo)"
        self.facts["disk_health"] = res

    def temperature(self) -> None:
        temps = {}
        for z in sorted(self._p("/sys/class/thermal").glob("thermal_zone*")):
            t, typ = self._read(str(z.relative_to(self.root)) + "/temp"), self._read(str(z.relative_to(self.root)) + "/type")
            if t and t.strip().lstrip("-").isdigit():
                temps[(typ or z.name).strip()] = int(t) / 1000
        for h in sorted(self._p("/sys/class/hwmon").glob("hwmon*")):
            name = (self._read(str(h.relative_to(self.root)) + "/name") or h.name).strip()
            for f in sorted(h.glob("temp*_input")):
                v = self._read(str(f.relative_to(self.root)))
                if v and v.strip().isdigit():
                    temps[f"{name}/{f.name.split('_')[0]}"] = int(v) / 1000
        temps = {k: v for k, v in temps.items() if 0 < v < 150}
        if not temps:
            if not self.facts.get("wsl"):
                self.not_checked["temperature"] = "no temperature sensors exposed"
            return
        self.facts["temperatures_c"] = {k: round(v, 1) for k, v in temps.items()}
        hot_name, hot = max(temps.items(), key=lambda kv: kv[1])
        if hot >= 95:
            self._add("critical", "temperature", "Overheating", f"{hot_name} at {hot:.0f} °C",
                      "Stop heavy work, check the fan and vents for dust, use a hard flat surface.")
        elif hot >= 85:
            self._add("warn", "temperature", "Running hot", f"{hot_name} at {hot:.0f} °C",
                      "The CPU will slow itself down to cool off. Clean vents; avoid soft surfaces.")

    def battery(self) -> None:
        if self.facts.get("wsl"):
            return            # WSL shows a virtual battery (always "healthy"); real values need native Linux
        bats = sorted(self._p("/sys/class/power_supply").glob("BAT*"))
        if not bats:
            return
        b = str(bats[0].relative_to(self.root))
        rd = lambda f: (self._read(f"{b}/{f}") or "").strip()
        cap, status = rd("capacity"), rd("status")
        full = rd("energy_full") or rd("charge_full")
        design = rd("energy_full_design") or rd("charge_full_design")
        info = {"charge_pct": int(cap) if cap.isdigit() else None, "status": status}
        if full.isdigit() and design.isdigit() and int(design):
            health = 100 * int(full) / int(design)
            info["health_pct"] = round(health, 1)
            if health < 60:
                self._add("warn", "battery", "Battery is badly worn",
                          f"holds {health:.0f}% of its original capacity",
                          "Expect much shorter battery life; consider a replacement battery.")
            elif health < 80:
                self._add("info", "battery", "Battery has aged",
                          f"holds {health:.0f}% of its original capacity",
                          "Normal for an older laptop; battery life will be shorter than when new.")
        if info["charge_pct"] is not None and info["charge_pct"] < 15 and status == "Discharging":
            self._add("warn", "battery", "Battery almost empty", f"{cap}% and discharging", "Plug in the charger.")
        self.facts["battery"] = info

    def services(self) -> None:
        code, out = self.run(["systemctl", "--failed", "--no-legend", "--plain", "--no-pager"])
        if code == 127 or "System has not been booted with systemd" in out or "Failed to connect" in out:
            self.not_checked["services"] = "systemd not available"
            return
        failed = [l.split()[0] for l in out.splitlines() if l.strip() and l.split()[0].endswith((".service", ".mount", ".socket", ".timer"))]
        self.facts["failed_services"] = failed
        if failed:
            self._add("warn", "services", f"{len(failed)} system service(s) failed to run",
                      ", ".join(failed[:6]) + (" ..." if len(failed) > 6 else ""),
                      "Ask: 'why did <service> fail?' — details are in its log (journalctl -u <service>).")

    def logs(self) -> None:
        code, out = self.run(["journalctl", "-b", "-p", "3", "--no-pager", "-o", "cat", "-n", "300"])
        if code == 127 or "No journal files" in out:
            self.not_checked["logs"] = "system journal not available"
            return
        if code != 0 and not out.strip():
            self.not_checked["logs"] = "journal unreadable (try with sudo)"
            return
        lines = [l for l in out.splitlines() if l.strip() and not l.startswith("-- ")]
        self.facts["errors_this_boot"] = len(lines)
        # most frequent messages, numbers stripped so repeats group together
        counts: dict[str, int] = {}
        for l in lines:
            key = re.sub(r"\d+", "#", l)[:120]
            counts[key] = counts.get(key, 0) + 1
        self.facts["top_errors"] = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        oom = [l for l in lines if "Out of memory" in l or "oom-kill" in l or "Killed process" in l]
        code2, kout = self.run(["journalctl", "-k", "-b", "--no-pager", "-o", "cat", "-g", "Killed process"])
        oom += [l for l in kout.splitlines() if "Killed process" in l and l not in oom]
        if oom:
            names = sorted({m.group(1) for l in oom for m in [re.search(r"\(([^)]+)\)", l)] if m})
            self._add("warn", "memory", "The system ran out of memory and killed programs",
                      f"{len(oom)} kill(s) this boot" + (f": {', '.join(names[:5])}" if names else ""),
                      "Something used all memory. Check which program grows large (ask: 'what is using memory?').")
        if len(lines) > 50:
            self._add("info", "logs", f"{len(lines)} errors logged since startup",
                      f"most common: {self.facts['top_errors'][0][0][:90]}" if self.facts["top_errors"] else "",
                      "Often harmless driver noise; ask about a specific one if something misbehaves.")


# ---------------------------------------------------------------------- output
def format_report(r: Report) -> str:
    out = [f"System check: {r.status.upper()}  ({r.seconds:.1f}s)"]
    f = r.facts
    if "cpu_busy_pct" in f:
        out.append(f"  CPU {f['cpu_busy_pct']:.0f}% busy on {f.get('cores', '?')} cores"
                   + (f", load {f['load_1_5_15'][1]:.1f}" if "load_1_5_15" in f else ""))
    if "memory" in f:
        m = f["memory"]
        out.append(f"  Memory {m['used_pct']:.0f}% used ({m['available']} free of {m['total']}), swap {m['swap_used']}")
    for d in f.get("disks", []):
        out.append(f"  Disk {d['mount']}: {d['used_pct']:.0f}% used, {d['free']} free")
    if f.get("battery"):
        b = f["battery"]
        out.append(f"  Battery {b.get('charge_pct')}% {b.get('status', '')}"
                   + (f", health {b['health_pct']:.0f}%" if b.get("health_pct") else ""))
    if f.get("temperatures_c"):
        k, v = max(f["temperatures_c"].items(), key=lambda kv: kv[1])
        out.append(f"  Hottest sensor {k}: {v:.0f} °C")
    if f.get("top_memory"):
        out.append("  Top memory: " + ", ".join(f"{a['name']} {a['rss']}" for a in f["top_memory"][:3]))
    if f.get("top_cpu"):
        out.append("  Top CPU:    " + ", ".join(f"{a['name']} ({cores_str(a['cpu_pct'], f.get('cores'))})"
                                              for a in f["top_cpu"][:3]))
    out.append("")
    problems = [x for x in r.findings if x.severity != "ok"]
    if not problems:
        out.append("No problems found.")
    for x in problems:
        out.append(f"[{ICON[x.severity]}] {x.title} — {x.evidence}")
        if x.advice:
            out.append(f"       -> {x.advice}")
    if r.not_checked:
        out.append("")
        out.append("Not checked: " + "; ".join(f"{k} ({v})" for k, v in r.not_checked.items()))
    return "\n".join(out)


def diagnose(sample_seconds: float = 1.0) -> Report:
    return Doctor(sample_seconds=sample_seconds).run_all()


EXPLAIN_PROMPT = """You are the diagnosis assistant of the user's computer.
Answer the user's question using ONLY the measurements and findings below. Do not invent problems,
numbers or programs that are not listed. If the data doesn't answer the question, say what was not checked.
Be short and concrete: what is wrong, why it matters, and what to do, most important first.

User question: {question}

System report:
{report}"""
