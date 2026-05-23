from __future__ import annotations

import json
import os
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
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        transport: JsonTransport | None = None,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.transport = transport or UrllibTransport()
        self.timeout = timeout

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
        response = self.transport.post_json(
            f"{self.base_url}/chat/completions",
            headers,
            payload,
            self.timeout,
        )
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
        timeout=int(os.getenv("LLM_TIMEOUT", raw.get("timeout", 120))),
    )
