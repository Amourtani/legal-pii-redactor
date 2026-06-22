from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.dataset import jsonl_record_to_bio
from train.llm_client import OpenAICompatibleClient
from train.synthetic import build_generation_prompt, parse_generated_records_lenient
from train.validate_dataset import validate_dataset


def generate_synthetic_dataset(
    client,
    output: str | Path,
    count: int,
    batch_size: int,
    mode: str,
    append: bool = False,
    max_batches: int = 0,
):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(output.name + ".tmp")
    if append and output.exists():
        temp_output.write_text(output.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        temp_output.write_text("", encoding="utf-8")

    written = 0
    batch_index = 0
    max_batches = max_batches or max(10, ((count + batch_size - 1) // batch_size) * 5)

    print(
        "using LLM "
        f"base_url={getattr(client, 'base_url', '<custom>')}, "
        f"model={getattr(client, 'model', '<custom>')}, "
        f"timeout={getattr(client, 'timeout', '<custom>')}s, "
        f"max_retries={getattr(client, 'max_retries', '<custom>')}, "
        f"retry_backoff={getattr(client, 'retry_backoff', '<custom>')}s"
    )

    with temp_output.open("a", encoding="utf-8") as file:
        while written < count:
            if batch_index >= max_batches:
                raise SystemExit(
                    f"stopped after {batch_index} batches: only {written}/{count} valid records generated"
                )
            batch_index += 1
            batch_count = min(batch_size, count - written)
            prompt = build_generation_prompt(batch_count, mode, batch_index=batch_index)
            payload = client.generate(prompt)
            records, errors = parse_generated_records_lenient(payload)
            for error in errors[:20]:
                print(f"skipped batch {batch_index} line {error.line}: {error.message}")
            for record in records:
                if written >= count:
                    break
                jsonl_record_to_bio(record)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            print(f"wrote {written}/{count} valid records to {temp_output}")

    report = validate_dataset(temp_output)
    if not report.ok:
        print(f"validation failed for {temp_output}")
        for error in report.errors[:100]:
            print(f"line {error.line}: {error.message}")
        raise SystemExit(1)
    temp_output.replace(output)
    print(f"validated {report.record_count} records and wrote {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate legal NER JSONL with a strong LLM.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=["case_display", "contract_generation", "general_redaction", "mixed_redaction"],
        default="case_display",
    )
    parser.add_argument("--output", default="data/generated/legal_ner.jsonl")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=None, help="Override config/llm.json timeout.")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override config/llm.json max_retries for timeout retries.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=None,
        help="Override config/llm.json retry_backoff seconds.",
    )
    args = parser.parse_args()

    client = OpenAICompatibleClient.from_config()
    if args.timeout is not None:
        client.timeout = args.timeout
    if args.max_retries is not None:
        client.max_retries = args.max_retries
    if args.retry_backoff is not None:
        client.retry_backoff = args.retry_backoff
    generate_synthetic_dataset(
        client=client,
        output=args.output,
        count=args.count,
        batch_size=args.batch_size,
        mode=args.mode,
        append=args.append,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
