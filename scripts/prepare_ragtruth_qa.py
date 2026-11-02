"""Convert official RAGTruth QA records into a response-level evaluation table.

RAGTruth labels hallucinated *spans*. This script derives a transparent
response-level label: 1 when a good-quality response contains at least one
officially annotated span, otherwise 0. It preserves the official train/test
partition and source_id for group-safe validation splitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-non-good-quality", action="store_true")
    args = parser.parse_args()

    responses = load_jsonl(args.responses)
    sources = {str(row["source_id"]): row for row in load_jsonl(args.sources)}
    records: list[dict[str, object]] = []
    for response in responses:
        source = sources.get(str(response["source_id"]))
        if source is None or source.get("task_type") != "QA":
            continue
        if not args.include_non_good_quality and response.get("quality") != "good":
            continue
        source_info = source.get("source_info")
        if not isinstance(source_info, dict):
            raise ValueError(f"QA source {source['source_id']} has non-dict source_info")
        question, evidence = source_info.get("question"), source_info.get("passages")
        if not isinstance(question, str) or not isinstance(evidence, str) or not isinstance(response.get("response"), str):
            raise ValueError(f"Malformed QA record for source_id={source['source_id']}")
        spans = response.get("labels", [])
        if not isinstance(spans, list):
            raise ValueError(f"labels must be a list for response id={response.get('id')}")
        records.append({
            "question_id": f"ragtruth_{source['source_id']}",
            "response_id": str(response["id"]),
            "variant": f"response_{response['id']}",
            "official_split": response["split"],
            "task_type": "QA",
            "generator_model": response.get("model"),
            "temperature": response.get("temperature"),
            "quality": response.get("quality"),
            "question": question,
            "evidence": evidence,
            "answer": response["response"],
            "label": int(len(spans) > 0),
            "hallucination_span_count": len(spans),
            "label_types": json.dumps(sorted({str(span.get("label_type", "")) for span in spans})),
            "implicit_true_span_count": sum(bool(span.get("implicit_true", False)) for span in spans),
        })
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No QA records selected. Check dataset paths and quality filter.")
    if not set(frame.official_split.unique()) <= {"train", "test"}:
        raise ValueError(f"Unexpected official split labels: {frame.official_split.unique()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    manifest = {
        "task": "RAGTruth QA response-level hallucination detection",
        "label_definition": "1 iff a retained response has one or more official RAGTruth hallucination-span annotations.",
        "quality_policy": "good only" if not args.include_non_good_quality else "all quality labels retained",
        "source_hashes": {"responses": sha256(args.responses), "sources": sha256(args.sources)},
        "records": int(len(frame)),
        "question_groups": int(frame.question_id.nunique()),
        "official_split_counts": frame.official_split.value_counts().to_dict(),
        "label_counts": {str(key): int(value) for key, value in frame.label.value_counts().items()},
        "important_caveat": "This response-level label discards span location and severity. Report it as a derived response-level evaluation, not as the official token/span metric.",
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
