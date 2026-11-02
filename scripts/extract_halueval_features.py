"""Extract reproducible HaluEval QA features from the full source dataset.

Design decisions fixed by the research rebuild:
* `knowledge` is the evidence; the question is never concatenated onto it.
* Direct length features are written for diagnostic and artifact baselines only.
* Feature extraction is model-versioned through the accompanying manifest.
* Any optional subset is sampled by question group with a recorded seed.

Run this in an isolated environment created from requirements-experiment.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline


TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
REQUIRED_KEYS = ("knowledge", "question", "right_answer", "hallucinated_answer")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def lexical_overlap(evidence: str, answer: str) -> float:
    evidence_tokens, answer_tokens = set(tokenize(evidence)), set(tokenize(answer))
    return len(evidence_tokens & answer_tokens) / len(answer_tokens) if answer_tokens else 0.0


def jaccard(evidence: str, answer: str) -> float:
    evidence_tokens, answer_tokens = set(tokenize(evidence)), set(tokenize(answer))
    union = evidence_tokens | answer_tokens
    return len(evidence_tokens & answer_tokens) / len(union) if union else 0.0


def uncertainty_score(answer: str) -> float:
    markers = (
        "i think", "probably", "maybe", "it is likely", "it seems", "as an ai",
        "i cannot verify", "not sure", "unknown",
    )
    text = str(answer).lower()
    return min(1.0, sum(marker in text for marker in markers) / 2.0)


def load_source(path: Path, limit_questions: int | None, sample_seed: int) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8").strip()
    rows = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line]
    for index, row in enumerate(rows):
        missing = [key for key in REQUIRED_KEYS if not isinstance(row.get(key), str) or not row[key].strip()]
        if missing:
            raise ValueError(f"Invalid record {index}; missing/non-text fields: {missing}")

    question_ids = np.arange(len(rows))
    if limit_questions is not None:
        if limit_questions <= 0 or limit_questions > len(rows):
            raise ValueError("--limit-questions must be between 1 and source question count")
        rng = np.random.default_rng(sample_seed)
        question_ids = np.sort(rng.choice(question_ids, size=limit_questions, replace=False))

    records: list[dict[str, object]] = []
    for question_id in question_ids:
        row = rows[int(question_id)]
        records.extend(
            [
                {"question_id": int(question_id), "variant": "factual", "label": 0,
                 "question": row["question"], "evidence": row["knowledge"], "answer": row["right_answer"]},
                {"question_id": int(question_id), "variant": "hallucinated", "label": 1,
                 "question": row["question"], "evidence": row["knowledge"], "answer": row["hallucinated_answer"]},
            ]
        )
    return pd.DataFrame.from_records(records)


def load_standardized_records(path: Path) -> pd.DataFrame:
    """Load an externally prepared response-level table, e.g. RAGTruth QA."""
    frame = pd.read_csv(path)
    required = {"question_id", "variant", "label", "question", "evidence", "answer"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Standardized record file lacks columns: {missing}")
    if frame[["question", "evidence", "answer"]].isna().any().any():
        raise ValueError("Standardized record file contains missing text fields")
    return frame


def add_basic_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["lexical_overlap"] = [lexical_overlap(e, a) for e, a in zip(result.evidence, result.answer)]
    result["lexical_mismatch"] = 1.0 - result["lexical_overlap"]
    result["jaccard"] = [jaccard(e, a) for e, a in zip(result.evidence, result.answer)]
    # Diagnostic-only / length-aware-baseline columns; exclude from the direct-length-excluded model.
    result["answer_len"] = result.answer.map(lambda value: len(tokenize(value)))
    result["evidence_len"] = result.evidence.map(lambda value: len(tokenize(value)))
    result["question_len"] = result.question.map(lambda value: len(tokenize(value)))
    result["answer_evidence_len_ratio"] = np.divide(
        result.answer_len, result.evidence_len, out=np.zeros(len(result)), where=result.evidence_len.ne(0)
    )
    result["uncertainty"] = result.answer.map(uncertainty_score)
    return result


def add_semantic_features(frame: pd.DataFrame, model_name: str, device: str, batch_size: int) -> pd.DataFrame:
    model = SentenceTransformer(model_name, device=device)
    encoded = {
        "evidence": model.encode(frame.evidence.tolist(), batch_size=batch_size, normalize_embeddings=True,
                                  show_progress_bar=True, convert_to_numpy=True),
        "answer": model.encode(frame.answer.tolist(), batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=True, convert_to_numpy=True),
        "question": model.encode(frame.question.tolist(), batch_size=batch_size, normalize_embeddings=True,
                                   show_progress_bar=True, convert_to_numpy=True),
    }
    result = frame.copy()
    answer_evidence = np.einsum("ij,ij->i", encoded["answer"], encoded["evidence"])
    answer_question = np.einsum("ij,ij->i", encoded["answer"], encoded["question"])
    result["semantic_answer_evidence_sim"] = answer_evidence
    result["semantic_answer_question_sim"] = answer_question
    # This is intentionally defined exactly as stated in the rebuilt manuscript protocol.
    result["semantic_mismatch"] = 1.0 - answer_evidence
    return result


def normalize_nli(labels: list[dict[str, object]]) -> dict[str, float]:
    probabilities = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
    for item in labels:
        label = str(item["label"]).lower()
        if "entail" in label:
            probabilities["entailment"] = float(item["score"])
        elif "neutral" in label:
            probabilities["neutral"] = float(item["score"])
        elif "contrad" in label:
            probabilities["contradiction"] = float(item["score"])
    if not np.isclose(sum(probabilities.values()), 1.0, atol=1e-3):
        raise ValueError(f"Unexpected NLI labels/probabilities: {labels}")
    return probabilities


def add_nli_features(frame: pd.DataFrame, model_name: str, device: int, batch_size: int, max_length: int) -> pd.DataFrame:
    classifier = pipeline("text-classification", model=model_name, device=device, top_k=None, truncation=True,
                          max_length=max_length)
    pairs = [{"text": evidence, "text_pair": answer} for evidence, answer in zip(frame.evidence, frame.answer)]
    rows: list[dict[str, float]] = []
    for start in range(0, len(pairs), batch_size):
        output = classifier(pairs[start : start + batch_size])
        rows.extend(normalize_nli(labels) for labels in output)
    result = frame.copy()
    nli = pd.DataFrame(rows)
    result["nli_entailment"] = nli["entailment"].to_numpy()
    result["nli_neutral"] = nli["neutral"].to_numpy()
    result["nli_contradiction"] = nli["contradiction"].to_numpy()
    result["nli_unsupported"] = np.clip(result.nli_contradiction + 0.35 * result.nli_neutral, 0, 1)
    return result


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="Raw HaluEval JSONL source")
    parser.add_argument("--records", type=Path, help="Prepared standard CSV, e.g. RAGTruth QA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--nli-model", default="facebook/bart-large-mnli")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--nli-max-length", type=int, default=512)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--sample-seed", type=int, default=11)
    args = parser.parse_args()

    if (args.source is None) == (args.records is None):
        raise ValueError("Provide exactly one of --source or --records")
    input_path = args.source if args.source is not None else args.records
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.records is not None and args.limit_questions is not None:
        raise ValueError("--limit-questions is only supported with --source")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    embedding_device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    nli_device = 0 if embedding_device == "cuda" else -1

    frame = (
        load_source(args.source, args.limit_questions, args.sample_seed)
        if args.source is not None
        else load_standardized_records(args.records)
    )
    frame = add_basic_features(frame)
    frame = add_semantic_features(frame, args.embedding_model, embedding_device, args.embedding_batch_size)
    frame = add_nli_features(frame, args.nli_model, nli_device, args.nli_batch_size, args.nli_max_length)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(input_path), "sha256": file_sha256(input_path)},
        "output": str(args.output),
        "question_groups": int(frame.question_id.nunique()),
        "examples": int(len(frame)),
        "sampling": {"limit_questions": args.limit_questions, "sample_seed": args.sample_seed},
        "models": {"embedding": args.embedding_model, "nli": args.nli_model},
        "inference": {"device": embedding_device, "embedding_batch_size": args.embedding_batch_size,
                      "nli_batch_size": args.nli_batch_size, "nli_max_length": args.nli_max_length},
        "software": {name: package_version(name) for name in ("numpy", "pandas", "torch", "transformers", "sentence-transformers")},
        "python": sys.version,
        "platform": platform.platform(),
        "direct_length_columns": ["answer_len", "evidence_len", "question_len", "answer_evidence_len_ratio"],
        "direct_length_excluded_features": [
            "lexical_overlap", "lexical_mismatch", "jaccard", "semantic_answer_evidence_sim",
            "semantic_answer_question_sim", "semantic_mismatch", "nli_entailment", "nli_neutral",
            "nli_contradiction", "nli_unsupported", "uncertainty",
        ],
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
