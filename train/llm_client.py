from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_LLM_CONFIG_PATH = Path("config/llm.json")


@dataclass(frozen=True, slots=True)
class LlmConfig:
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1"
    timeout: int = 120
    max_retries: int = 2
    retry_backoff: float = 2.0


class LlmRequestError(RuntimeError):
    pass


class LlmTimeoutError(LlmRequestError):
    def __init__(self, message: str, timeout: int, elapsed: float, url: str) -> None:
        super().__init__(message)
        self.timeout = timeout
        self.elapsed = elapsed
        self.url = url


class JsonTransport(Protocol):
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: int,
    ) -> dict[str, Any]:
        ...


class UrllibTransport:
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: int,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started_at = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            if _is_timeout_error(exc.reason):
                elapsed = time.monotonic() - started_at
                raise _build_timeout_error(url, timeout, elapsed) from exc
            raise
        except OSError as exc:
            if _is_timeout_error(exc):
                elapsed = time.monotonic() - started_at
                raise _build_timeout_error(url, timeout, elapsed) from exc
            raise


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        transport: JsonTransport | None = None,
        timeout: int = 120,
        max_retries: int = 2,
        retry_backoff: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.transport = transport or UrllibTransport()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        return cls.from_config()

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_LLM_CONFIG_PATH) -> "OpenAICompatibleClient":
        config = load_llm_config(config_path)
        return cls(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout,
            max_retries=config.max_retries,
            retry_backoff=config.retry_backoff,
        )

    def generate(self, prompt: str, temperature: float = 0.4) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        response = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.transport.post_json(
                    f"{self.base_url}/chat/completions",
                    headers,
                    payload,
                    self.timeout,
                )
                break
            except LlmTimeoutError:
                if attempt >= self.max_retries:
                    raise
                if self.retry_backoff > 0:
                    time.sleep(self.retry_backoff * (attempt + 1))
        if response is None:
            raise LlmRequestError("LLM request failed without a response")
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected LLM response: {response}") from exc


def load_llm_config(config_path: str | Path = DEFAULT_LLM_CONFIG_PATH) -> LlmConfig:
    path = Path(config_path)
    raw: dict[str, Any] = {}
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))

    return LlmConfig(
        api_key=os.getenv("LLM_API_KEY", str(raw.get("api_key", ""))),
        base_url=os.getenv("LLM_BASE_URL", str(raw.get("base_url", "https://api.openai.com/v1"))),
        model=os.getenv("LLM_MODEL", str(raw.get("model", "gpt-4.1"))),
        timeout=int(os.getenv("LLM_TIMEOUT", raw.get("timeout", raw.get("request_timeout", 120)))),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", raw.get("max_retries", 2))),
        retry_backoff=float(os.getenv("LLM_RETRY_BACKOFF", raw.get("retry_backoff", 2.0))),
    )


def _is_timeout_error(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, socket.timeout))


def _build_timeout_error(url: str, timeout: int, elapsed: float) -> LlmTimeoutError:
    return LlmTimeoutError(
        (
            f"LLM request timed out: configured timeout={timeout}s, elapsed={elapsed:.1f}s, "
            "url={url}. If elapsed is much lower than configured timeout, the upstream API "
            "gateway or proxy may have a shorter timeout; reduce --batch-size or use another endpoint."
        ).format(url=url),
        timeout=timeout,
        elapsed=elapsed,
        url=url,
    )
