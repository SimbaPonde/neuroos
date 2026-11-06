"""Model backends. Messages are kept in OpenAI chat format internally;
each backend converts as needed. Standard library only (no SDKs)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class BackendError(RuntimeError):
    pass


def _post(url: str, payload: dict, headers: dict | None = None, timeout: float = 300) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise BackendError(f"HTTP {e.code} from {url}: {e.read().decode(errors='replace')[:300]}") from e
    except urllib.error.URLError as e:
        raise BackendError(f"Cannot reach {url}: {e.reason}") from e
    except (TimeoutError, OSError) as e:      # slow model / dropped connection = failed attempt
        raise BackendError(f"Timeout or connection error after {timeout}s from {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise BackendError(f"Bad JSON from {url}: {e}") from e


class Backend:
    name: str = "backend"
    is_local: bool = True

    max_output_tokens: int = 1024

    def chat(self, messages: list[dict], tools: list[dict], timeout: float | None = None) -> ModelResponse:
        raise NotImplementedError


class OllamaBackend(Backend):
    """Local model via Ollama's /api/chat (supports tool calling)."""

    is_local = True

    def __init__(self, model: str, url: str = "http://localhost:11434", keep_alive: str = "10m",
                 timeout: float = 600):
        self.model, self.url, self.keep_alive, self.timeout = model, url.rstrip("/"), keep_alive, timeout
        self.name = f"ollama:{model}"

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [
                    {"function": {"name": tc["function"]["name"],
                                  "arguments": json.loads(tc["function"]["arguments"] or "{}")}}
                    for tc in m["tool_calls"]
                ]
            m.pop("tool_call_id", None)
            out.append(m)
        return out

    def chat(self, messages, tools, timeout=None):
        t0 = time.perf_counter()
        r = _post(f"{self.url}/api/chat", {
            "model": self.model, "messages": self._convert(messages), "tools": tools,
            "stream": False, "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_predict": self.max_output_tokens},   # stops runaway generation
        }, timeout=min(self.timeout, timeout) if timeout else self.timeout)
        msg = r.get("message", {})
        calls = [ToolCall(id=f"call_{i}", name=tc["function"]["name"],
                          arguments=tc["function"].get("arguments") or {})
                 for i, tc in enumerate(msg.get("tool_calls") or [])]
        return ModelResponse(text=msg.get("content", ""), tool_calls=calls,
                             prompt_tokens=r.get("prompt_eval_count", 0),
                             completion_tokens=r.get("eval_count", 0),
                             latency_s=time.perf_counter() - t0)

    def loaded_models(self) -> list[str]:
        """Models currently resident in memory (useful for model-residency experiments)."""
        try:
            with urllib.request.urlopen(f"{self.url}/api/ps", timeout=5) as r:
                return [m["name"] for m in json.loads(r.read()).get("models", [])]
        except Exception:
            return []


class OpenAICompatBackend(Backend):
    """Cloud (or any remote) model through an OpenAI-compatible /chat/completions endpoint."""

    is_local = False

    def __init__(self, model: str, url: str, api_key: str):
        self.model, self.url, self.api_key = model, url.rstrip("/"), api_key
        self.name = f"cloud:{model}"

    def chat(self, messages, tools, timeout=None):
        t0 = time.perf_counter()
        r = _post(f"{self.url}/chat/completions",
                  {"model": self.model, "messages": messages, "tools": tools, "temperature": 0,
                   "max_tokens": self.max_output_tokens},
                  headers={"Authorization": f"Bearer {self.api_key}"}, timeout=timeout or 300)
        msg = r["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.get("id", ""), name=tc["function"]["name"], arguments=args))
        usage = r.get("usage", {})
        return ModelResponse(text=msg.get("content") or "", tool_calls=calls,
                             prompt_tokens=usage.get("prompt_tokens", 0),
                             completion_tokens=usage.get("completion_tokens", 0),
                             latency_s=time.perf_counter() - t0)


class MockBackend(Backend):
    """Scripted backend for tests and for developing without a GPU.
    `script` is a list of ModelResponse objects returned in order."""

    def __init__(self, name: str, script: list[ModelResponse], is_local: bool = True):
        self.name, self._script, self.is_local = name, list(script), is_local
        self.calls = 0

    def chat(self, messages, tools, timeout=None):
        self.calls += 1
        if not self._script:
            return ModelResponse(text="(mock: no more scripted responses)")
        return self._script.pop(0)
