from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from app.entities import Entity
from app.span import merge_entities

logger = logging.getLogger(__name__)


class OnnxNerDetector:
    def __init__(
        self,
        model_dir: str | Path,
        max_length: int = 256,
        stride: int = 128,
        enable_sliding_window: bool = True,
    ) -> None:
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
        self.stride = min(stride, max_length // 2)
        self.enable_sliding_window = enable_sliding_window
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.id2label = self._load_labels(self.model_dir)
        self.input_names = {item.name for item in self.session.get_inputs()}

    def detect(self, text: str) -> list[Entity]:
        start_time = time.time()
        all_entities: list[Entity] = []
        chunks = self._build_chunks(text)

        for chunk_idx, (chunk_text, char_offset) in enumerate(chunks):
            chunk_entities = self._detect_chunk(chunk_text, char_offset)
            all_entities.extend(chunk_entities)
            logger.debug(
                "chunk %d/%d: offset=%d, len=%d, entities=%d",
                chunk_idx + 1,
                len(chunks),
                char_offset,
                len(chunk_text),
                len(chunk_entities),
            )

        merged = merge_entities(all_entities)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "NER detected %d entities from %d chunks in %.1fms",
            len(merged),
            len(chunks),
            elapsed_ms,
        )
        return merged

    def _build_chunks(self, text: str) -> list[tuple[str, int]]:
        if not text:
            return []

        if not self.enable_sliding_window:
            return [(text, 0)]

        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        offsets = encoded["offset_mapping"]

        # 预留 [CLS] 和 [SEP] 的位置
        usable = self.max_length - 2
        if len(offsets) <= usable:
            return [(text, 0)]

        step = max(usable - self.stride, 1)
        chunks: list[tuple[str, int]] = []
        seen_starts: set[int] = set()
        for start in range(0, len(offsets), step):
            end = min(start + usable, len(offsets))
            char_start = int(offsets[start][0])
            char_end = int(offsets[end - 1][1])
            if char_start in seen_starts:
                continue
            seen_starts.add(char_start)
            chunks.append((text[char_start:char_end], char_start))
            if end >= len(offsets):
                break
        return chunks

    def _detect_chunk(self, chunk_text: str, char_offset: int) -> list[Entity]:
        encoded = self.tokenizer(
            chunk_text,
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
        chunk_entities = self._bio_to_entities(chunk_text, offsets, pred_ids, scores)
        # 将 chunk 内的相对偏移转换为原始文本的绝对偏移
        return [
            Entity(
                start=e.start + char_offset,
                end=e.end + char_offset,
                label=e.label,
                text=e.text,
                source=e.source,
                score=e.score,
                replacement=e.replacement,
            )
            for e in chunk_entities
        ]

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
