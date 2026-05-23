from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.entities import Entity
from app.span import merge_entities


class OnnxNerDetector:
    def __init__(self, model_dir: str | Path, max_length: int = 256) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install model dependencies with: pip install -r requirements-ml.txt"
            ) from exc

        self.np = np
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.id2label = self._load_labels(self.model_dir)
        self.input_names = {item.name for item in self.session.get_inputs()}

    def detect(self, text: str) -> list[Entity]:
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
        )
        offsets = encoded.pop("offset_mapping")[0]
        feed = {name: value for name, value in encoded.items() if name in self.input_names}
        logits = self.session.run(None, feed)[0][0]
        pred_ids = self.np.argmax(logits, axis=-1)
        scores = _softmax(logits, self.np).max(axis=-1)
        return merge_entities(self._bio_to_entities(text, offsets, pred_ids, scores))

    def _bio_to_entities(self, text: str, offsets: Any, pred_ids: Any, scores: Any) -> list[Entity]:
        entities: list[Entity] = []
        current_label: str | None = None
        current_start: int | None = None
        current_end: int | None = None
        current_scores: list[float] = []

        def close_current() -> None:
            nonlocal current_label, current_start, current_end, current_scores
            if current_label is not None and current_start is not None and current_end is not None:
                entities.append(
                    Entity(
                        start=current_start,
                        end=current_end,
                        label=current_label,
                        text=text[current_start:current_end],
                        source="ner",
                        score=sum(current_scores) / len(current_scores),
                    )
                )
            current_label = None
            current_start = None
            current_end = None
            current_scores = []

        for offset, pred_id, score in zip(offsets, pred_ids, scores):
            start, end = int(offset[0]), int(offset[1])
            if start == end:
                continue
            tag = self.id2label.get(int(pred_id), "O")
            if tag == "O":
                close_current()
                continue
            prefix, _, label = tag.partition("-")
            if prefix == "B" or current_label != label or current_end != start:
                close_current()
                current_label = label
                current_start = start
                current_end = end
                current_scores = [float(score)]
            else:
                current_end = end
                current_scores.append(float(score))
        close_current()
        return entities

    @staticmethod
    def _load_labels(model_dir: Path) -> dict[int, str]:
        labels_path = model_dir / "labels.json"
        if labels_path.exists():
            raw = json.loads(labels_path.read_text(encoding="utf-8"))
            return {int(key): value for key, value in raw.items()}
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        return {int(key): value for key, value in config["id2label"].items()}


def _softmax(values: Any, np: Any) -> Any:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)

