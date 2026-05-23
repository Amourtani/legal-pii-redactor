from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the trained NER model to ONNX.")
    parser.add_argument("--model-dir", default="models/legal-ner")
    parser.add_argument("--output", default="models/legal-ner-onnx")
    args = parser.parse_args()

    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install ML dependencies with: pip install -r requirements-ml.txt") from exc

    model = ORTModelForTokenClassification.from_pretrained(args.model_dir, export=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()

