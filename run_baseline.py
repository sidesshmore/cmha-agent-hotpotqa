#!/usr/bin/env python3
"""Entry point for the CMHA-Agent baseline.

Example:
    python run_baseline.py --strategy cmha --limit 10

See README.md for full setup instructions (dependencies, env vars, expected
runtime, where the input/output live).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cmha_agent import run_batch  # noqa: E402
from embedder import Embedder  # noqa: E402
from llm_client import LLMClient  # noqa: E402

# Small, cross-organization models chosen to each individually fit well under
# 8GB RAM when Ollama loads them (Ollama loads one model at a time, so peak
# RAM is the size of the largest single model below, not their sum — see
# README.md "Local models and RAM budget").
DEFAULT_HYPOTHESIS_MODELS = [
    "qwen2.5:3b",  # Alibaba, ~1.9GB
    "llama3.2:3b",  # Meta, ~2.0GB
    "gemma2:2b",  # Google, ~1.6GB
    "phi3.5:3.8b",  # Microsoft, ~2.2GB
]
DEFAULT_ANSWER_MODEL = "qwen2.5:3b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"  # ~274MB, served by the same local Ollama process


def parse_args():
    p = argparse.ArgumentParser(description="CMHA-Agent baseline on a frozen HotpotQA slice.")
    p.add_argument("--data", default="data/hotpotqa_sample.json", help="Path to the frozen question set.")
    p.add_argument(
        "--strategy",
        choices=["direct", "single_hyde", "cmha"],
        default="cmha",
        help="Retrieval strategy: direct query embedding, single-model HyDE, or full CMHA (default).",
    )
    p.add_argument("--k", type=int, default=4, help="Number of paragraphs to retrieve per question.")
    p.add_argument("--limit", type=int, default=None, help="Only run the first N questions (for a quick smoke test).")
    p.add_argument("--hypothesis-models", default=",".join(DEFAULT_HYPOTHESIS_MODELS), help="Comma-separated model list for CMHA hypothesis generation.")
    p.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL, help="Model used for the final answer-generation call.")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model name (served locally, e.g. nomic-embed-text).")
    p.add_argument("--out", default=None, help="Output JSONL path (default: results/<strategy>_<timestamp>.jsonl).")
    p.add_argument("--mock", action="store_true", help="Skip Ollama entirely (no server, no models needed); verify the pipeline runs end-to-end with canned text (does NOT produce meaningful scores).")
    return p.parse_args()


def load_completed_ids(out_path):
    completed = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    completed.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def main():
    args = parse_args()

    with open(args.data) as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]

    out_path = args.out
    if out_path is None:
        os.makedirs("results", exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = f"results/{args.strategy}_{ts}.jsonl"

    print(f"[setup] strategy={args.strategy} k={args.k} n_questions={len(items)} mock={args.mock}")
    print(f"[setup] embedding model: {args.embed_model} (via local Ollama, ~274MB, pulled once)")
    print(f"[setup] output: {out_path}")

    embedder = Embedder(model=args.embed_model, mock=args.mock)
    llm = LLMClient(mock=args.mock)
    hypothesis_models = [m.strip() for m in args.hypothesis_models.split(",") if m.strip()]

    completed_ids = load_completed_ids(out_path)
    todo_items = [item for item in items if item["id"] not in completed_ids]
    if completed_ids:
        print(f"[resume] {len(completed_ids)} questions already logged in {out_path}, skipping those.")

    t0 = time.time()
    if todo_items:
        # Batched model-major (see cmha_agent.run_batch docstring): results for the
        # whole todo_items list are computed together, then written all at once below.
        # This trades true per-question incremental flushing for a large local-model
        # swap-time reduction (see README "Local models and RAM budget"). Previously
        # completed questions from an earlier, interrupted run are still preserved —
        # only the remaining `todo_items` are recomputed.
        new_records = run_batch(
            todo_items,
            embedder=embedder,
            llm=llm,
            hypothesis_models=hypothesis_models,
            answer_model=args.answer_model,
            k=args.k,
            strategy=args.strategy,
        )
        with open(out_path, "a") as out_f:
            for record in new_records:
                out_f.write(json.dumps(record) + "\n")

    # Reload the full file (covers resumed runs where earlier results weren't in this run)
    all_records = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    ok = [r for r in all_records if "error" not in r]
    errs = [r for r in all_records if "error" in r]

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"CMHA-Agent baseline — strategy={args.strategy}, k={args.k}")
    print("=" * 60)
    print(f"questions scored : {len(ok)} / {len(all_records)}  ({len(errs)} errors)")
    if ok:
        em = sum(r["em"] for r in ok) / len(ok)
        f1 = sum(r["f1"] for r in ok) / len(ok)
        recall = sum(r["retrieval_recall"] for r in ok) / len(ok)
        conf = sum(r["confidence"] for r in ok) / len(ok)
        print(f"exact match      : {em:.3f}")
        print(f"token F1         : {f1:.3f}")
        print(f"retrieval recall : {recall:.3f}  (gold-paragraph hit rate @ k={args.k})")
        print(f"mean confidence  : {conf:.3f}")
    if todo_items:
        print(f"elapsed          : {elapsed:.1f}s  ({elapsed / len(todo_items):.1f}s/question, {len(todo_items)} newly run)")
    else:
        print(f"elapsed          : {elapsed:.1f}s  (nothing new to run — all questions already in {out_path})")
    print(f"full log         : {out_path}")
    print("=" * 60)

    if args.mock:
        print("\n[note] --mock was used: no Ollama server or models were needed, and the")
        print("       numbers above are NOT meaningful QA results — they only confirm the")
        print("       pipeline runs end-to-end. Re-run without --mock (with Ollama running")
        print("       and the models pulled) for the graded baseline output.")


if __name__ == "__main__":
    main()
