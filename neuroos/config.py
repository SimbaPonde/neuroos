"""Central configuration. Override any field with environment variables or a JSON file."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Local model server (Ollama)
    ollama_url: str = "http://localhost:11434"
    small_model: str = "qwen2.5:3b"      # fast, cheap pathway
    large_model: str = "qwen2.5:7b"      # stronger local pathway

    # Optional cloud pathway (any OpenAI-compatible endpoint). Disabled if no key.
    cloud_url: str = "https://api.openai.com/v1"
    cloud_model: str = "gpt-4o-mini"
    cloud_api_key_env: str = "NEUROOS_CLOUD_API_KEY"

    # Cloud energy is NOT measurable from your laptop, so it is estimated.
    # Wh per 1,000 tokens. Report results across a range (sensitivity analysis).
    cloud_wh_per_1k_tokens: float = 0.3

    # Agent behaviour
    max_steps: int = 8                    # tool-call iterations per pathway
    request_timeout_s: float = 600.0      # per model call; a timeout counts as a failed attempt
    attempt_budget_s: float = 300.0       # wall-clock budget per pathway attempt; exceeded -> failed attempt
    max_output_tokens: int = 1024         # cap per model call (prevents runaway generation)
    escalate_on_failure: bool = True      # small -> large -> cloud cascade
    workspace: str = str(Path.home() / "neuroos_workspace")  # agent may only touch this tree
    auto_approve: bool = False            # NEVER true outside benchmarks in a sandbox
    # Shell: "sandbox" = bubblewrap (workspace writable, rest of system read-only, home hidden,
    # no network); "off" = shell tool disabled; "host" = plain shell (asks first).
    shell_mode: str = "sandbox"
    # After a model says it is done, ask it to check its own work; FAIL -> escalate.
    self_verify: bool = False
    # Runtime validators (neuroos/validators.py): suspicious result -> attempt counts as failed -> escalate.
    validate: bool = True

    # Logging
    log_dir: str = str(Path.home() / ".neuroos" / "logs")

    # Energy sampling
    energy_sample_hz: float = 10.0

    extra: dict = field(default_factory=dict)

    @property
    def cloud_api_key(self) -> str | None:
        return os.environ.get(self.cloud_api_key_env)

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        if path and Path(path).exists():
            for k, v in json.loads(Path(path).read_text()).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        for k in asdict(cfg):
            env = os.environ.get(f"NEUROOS_{k.upper()}")
            if env is not None:
                cur = getattr(cfg, k)
                if isinstance(cur, bool):
                    setattr(cfg, k, env.lower() in ("1", "true", "yes"))
                elif isinstance(cur, int):
                    setattr(cfg, k, int(env))
                elif isinstance(cur, float):
                    setattr(cfg, k, float(env))
                elif isinstance(cur, str):
                    setattr(cfg, k, env)
        return cfg
