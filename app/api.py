from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

from app.desensitizer import Desensitizer, NullNerDetector
from app.ner import OnnxNerDetector
from app.serialization import result_to_dict


class DesensitizeRequest(BaseModel):
    text: str = Field(min_length=1)
    mode: Literal["case_display", "contract_generation"] = "case_display"
    strict_level: Literal["low", "medium", "high"] = "medium"


class DesensitizeResponse(BaseModel):
    masked_text: str
    risk_level: str
    need_review: bool
    entities: list[dict]


@lru_cache(maxsize=1)
def get_desensitizer() -> Desensitizer:
    model_dir = os.getenv("NER_MODEL_DIR", "").strip()
    ner_detector = OnnxNerDetector(model_dir) if model_dir else NullNerDetector()
    return Desensitizer(ner_detector=ner_detector)


def create_app():
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise ImportError("Install API dependencies with: pip install -r requirements.txt") from exc

    app = FastAPI(title="Legal PII Redactor", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/desensitize", response_model=DesensitizeResponse)
    def desensitize(request: DesensitizeRequest) -> dict:
        result = get_desensitizer().desensitize(
            request.text,
            mode=request.mode,
            strict_level=request.strict_level,
        )
        return result_to_dict(result)

    return app


app = create_app()
