from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="INT8 dynamic quantization for CPU ONNX inference.")
    parser.add_argument("--onnx-dir", default="models/legal-ner-onnx")
    parser.add_argument("--output-dir", default="models/legal-ner-onnx-int8")
    args = parser.parse_args()

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise SystemExit("Install ML dependencies with: pip install -r requirements-ml.txt") from exc

    source = Path(args.onnx_dir)
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "labels.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]:
        source_file = source / name
        if source_file.exists():
            shutil.copy2(source_file, target / name)
    quantize_dynamic(
        model_input=str(source / "model.onnx"),
        model_output=str(target / "model.onnx"),
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    main()

