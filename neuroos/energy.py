"""Energy measurement. This is the part that makes the research credible, so it is
explicit about what it can and cannot measure.

Sources (auto-detected, combined):
  - Intel/AMD RAPL via /sys/class/powercap  -> CPU package (+DRAM if exposed). Usually needs root.
  - NVIDIA GPU via `nvidia-smi` power polling -> integrated to joules.
  - Anything else -> reports `None` (never invents a number).

Cloud energy cannot be measured from the client; it is estimated per token in agent.py
and kept in a separate column so the two are never mixed silently.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

RAPL_ROOT = Path("/sys/class/powercap")


@dataclass
class EnergyReading:
    joules: float | None
    seconds: float
    by_source: dict[str, float] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


class _Rapl:
    """Top-level RAPL domains (package-N, and psys if present)."""

    def __init__(self):
        self.domains: list[tuple[str, Path, int]] = []
        if not RAPL_ROOT.exists():
            return
        for d in sorted(RAPL_ROOT.glob("intel-rapl:*")):
            if d.name.count(":") != 1:          # skip subdomains (core/uncore/dram) to avoid double counting
                continue
            f, mx = d / "energy_uj", d / "max_energy_range_uj"
            try:
                int(f.read_text()); rng = int(mx.read_text())
            except (OSError, ValueError):
                continue                          # unreadable without root
            self.domains.append(((d / "name").read_text().strip(), f, rng))

    @property
    def available(self) -> bool:
        return bool(self.domains)

    def read(self) -> dict[str, int]:
        return {name: int(f.read_text()) for name, f, _ in self.domains}

    def delta_j(self, a: dict[str, int], b: dict[str, int]) -> float:
        total = 0
        for name, _, rng in self.domains:
            d = b[name] - a[name]
            if d < 0:
                d += rng                          # counter wrapped
            total += d
        return total / 1e6


class _NvidiaPoller(threading.Thread):
    def __init__(self, hz: float):
        super().__init__(daemon=True)
        self.dt, self.joules, self._stop = 1.0 / hz, 0.0, threading.Event()

    @staticmethod
    def available() -> bool:
        if not shutil.which("nvidia-smi"):
            return False
        try:
            _NvidiaPoller._watts()
            return True
        except Exception:
            return False

    @staticmethod
    def _watts() -> float:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=2).stdout
        return sum(float(x) for x in out.split() if x.replace(".", "", 1).isdigit())

    def run(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.dt)
            now = time.perf_counter()
            try:
                self.joules += self._watts() * (now - last)   # rectangle rule; fine at >=10 Hz
            except Exception:
                pass
            last = now

    def stop(self):
        self._stop.set()


class EnergyMeter:
    """Usage:
        meter = EnergyMeter()
        with meter.measure() as m:
            ...work...
        print(m.reading.joules)
    """

    def __init__(self, sample_hz: float = 10.0):
        self.hz = sample_hz
        self.rapl = _Rapl()
        self.nvidia = _NvidiaPoller.available()

    def describe(self) -> str:
        srcs = (["rapl:" + ",".join(n for n, _, _ in self.rapl.domains)] if self.rapl.available else []) \
               + (["nvidia-smi"] if self.nvidia else [])
        return ", ".join(srcs) if srcs else "none (run as root for RAPL, or use an NVIDIA GPU / external meter)"

    def measure(self) -> "_Measurement":
        return _Measurement(self)


class _Measurement:
    def __init__(self, meter: EnergyMeter):
        self.m, self.reading = meter, EnergyReading(None, 0.0)

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.r0 = self.m.rapl.read() if self.m.rapl.available else None
        self.gpu = _NvidiaPoller(self.m.hz) if self.m.nvidia else None
        if self.gpu:
            self.gpu.start()
        return self

    def __exit__(self, *exc):
        secs = time.perf_counter() - self.t0
        by: dict[str, float] = {}
        if self.r0 is not None:
            by["cpu_rapl"] = self.m.rapl.delta_j(self.r0, self.m.rapl.read())
        if self.gpu:
            self.gpu.stop(); self.gpu.join(timeout=1)
            by["gpu_nvidia"] = self.gpu.joules
        self.reading = EnergyReading(joules=sum(by.values()) if by else None,
                                     seconds=secs, by_source=by, sources=list(by))
        return False


def idle_baseline_watts(meter: EnergyMeter, seconds: float = 10.0) -> float | None:
    """Measure idle power so experiments can report energy ABOVE idle (subtract watts * duration)."""
    with meter.measure() as m:
        time.sleep(seconds)
    return None if m.reading.joules is None else m.reading.joules / m.reading.seconds
