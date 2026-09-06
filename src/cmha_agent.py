"""CMHA-Agent: cross-model hypothesis retrieval + single-hop generation.

This is the retrieval half (Cross-Model Hypothesis Aggregation, CMHA) from
"Beyond HyDE" wired into a generation step, evaluated end-to-end on answer
exact-match/F1 rather than only Recall@K. See ../README.md for the full
Problem Definition / Motivation this baseline supports.

Three --strategy modes let a single script reproduce the three-way
comparison the proposal's evaluation plan needs:
  - direct : embed the raw question (no LLM call at retrieval time)
  - single_hyde : one LLM generates one hypothesis, embed that
  - cmha : N diverse LLMs each generate a hypothesis; embed the centroid

Pipeline is staged MODEL-MAJOR, not question-major: every call to model X
across all questions happens before we move to model Y. Ollama loads one
model into RAM at a time, and swapping models costs ~3-5s (confirmed by
timing on this machine) versus ~0.3s for another call to an already-loaded
model. Looping question-major (load A, B, C, D, embed, answer — repeat per
question) would pay that swap cost 6-7 times *per question*; looping
model-major pays it 6-7 times for the *entire run*, regardless of question
count. This is the difference between a 30-question run taking ~15 minutes
vs. ~2-3 minutes.

The tradeoff: results are written to disk once per run_batch() call, not
incrementally per question (unlike a question-major loop, which could flush
after each item). For a local CPU run of a few dozen questions this is an
acceptable trade — see README "Local models and RAM budget" for the honest
accounting.
"""

import itertools

import numpy as np

from embedder import Embedder
from hotpot_metrics import exact_match_score, f1_score
from llm_client import LLMClient

HYPOTHESIS_PROMPT = (
    "Write a short passage (2-4 sentences) that would directly answer the "
    "following question. Write only the passage, no preamble.\n\n"
    "Question: {q}"
)

ANSWER_PROMPT = (
    "Answer the question using ONLY the evidence passages below. "
    "Respond with the shortest possible answer (a name, date, number, or "
    "short phrase) and nothing else — no explanation, no full sentence.\n\n"
    "Evidence:\n{evidence}\n\n"
    "Question: {q}\n"
    "Answer:"
)


def _centroid(vecs: np.ndarray) -> np.ndarray:
    c = vecs.sum(axis=0)
    norm = np.linalg.norm(c)
    return c / norm if norm > 0 else c


def _diversity_score(vecs: np.ndarray) -> float:
    """Mean pairwise cosine distance between hypothesis embeddings.

    Same statistic as div(q) in Beyond-HyDE §3.3 — there it predicted per-query
    retrieval difficulty (Pearson r=-0.53 on HotpotQA). We reuse it directly as
    this baseline's confidence proxy instead of inventing a new signal.
    """
    n = len(vecs)
    if n < 2:
        return 0.0
    dists = [1.0 - float(np.dot(vecs[i], vecs[j])) for i, j in itertools.combinations(range(n), 2)]
    return sum(dists) / len(dists)


def _embed_paragraphs(items: list[dict], embedder: Embedder) -> dict:
    """One contiguous embedding-model pass over every paragraph of every item."""
    flat_texts, spans, cursor = [], {}, 0
    for item in items:
        texts = [f"{p['title']}: {p['text']}" for p in item["paragraphs"]]
        flat_texts.extend(texts)
        spans[item["id"]] = (cursor, cursor + len(texts))
        cursor += len(texts)
    print(f"[embed] {len(flat_texts)} paragraphs across {len(items)} questions...")
    vecs = embedder.embed(flat_texts)
    return {iid: vecs[s:e] for iid, (s, e) in spans.items()}


def _embed_questions(items: list[dict], embedder: Embedder) -> dict:
    vecs = embedder.embed([item["question"] for item in items])
    return {item["id"]: vecs[i] for i, item in enumerate(items)}


def _generate_hypotheses(items: list[dict], llm: LLMClient, models: list[str], errors: dict) -> dict:
    """Model-major: model A finishes every item before model B starts."""
    hyps = {item["id"]: [] for item in items}
    for model in models:
        print(f"[hypotheses] {model}: generating for {len(items)} questions...")
        for i, item in enumerate(items, 1):
            try:
                h = llm.complete(model, HYPOTHESIS_PROMPT.format(q=item["question"]), max_tokens=120, temperature=0.7)
                hyps[item["id"]].append(h)
            except Exception as e:  # one bad model call shouldn't sink the whole item
                errors.setdefault(item["id"], []).append(f"hypothesis[{model}]: {e}")
            if i % 10 == 0 or i == len(items):
                print(f"    {i}/{len(items)}")
    return hyps


def _embed_hypotheses(hyps_by_id: dict, embedder: Embedder) -> dict:
    flat_texts, spans, cursor = [], {}, 0
    for iid, texts in hyps_by_id.items():
        flat_texts.extend(texts)
        spans[iid] = (cursor, cursor + len(texts))
        cursor += len(texts)
    if not flat_texts:
        return {iid: np.zeros((0, 0), dtype=np.float32) for iid in hyps_by_id}
    vecs = embedder.embed(flat_texts)
    return {iid: vecs[s:e] for iid, (s, e) in spans.items()}


def _retrieve(items: list[dict], para_vecs_by_id: dict, query_vec_by_id: dict, k: int) -> dict:
    out = {}
    for item in items:
        iid = item["id"]
        paragraphs = item["paragraphs"]
        sims = para_vecs_by_id[iid] @ query_vec_by_id[iid]
        order = np.argsort(-sims)[:k]
        retrieved = [paragraphs[i] for i in order]
        retrieved_titles = [p["title"] for p in retrieved]
        gold_titles = set(item["gold_titles"])
        recall = len(set(retrieved_titles) & gold_titles) / len(gold_titles) if gold_titles else 0.0
        out[iid] = {
            "retrieved_titles": retrieved_titles,
            "gold_titles": sorted(gold_titles),
            "retrieval_recall": round(recall, 4),
            "evidence_block": "\n\n".join(f"[{p['title']}] {p['text']}" for p in retrieved),
        }
    return out


def _generate_answers(items: list[dict], evidence_by_id: dict, llm: LLMClient, answer_model: str, errors: dict) -> dict:
    print(f"[answers] {answer_model}: generating for {len(items)} questions...")
    answers = {}
    for i, item in enumerate(items, 1):
        iid = item["id"]
        try:
            answers[iid] = llm.complete(
                answer_model,
                ANSWER_PROMPT.format(evidence=evidence_by_id[iid]["evidence_block"], q=item["question"]),
                max_tokens=32,
                temperature=0.0,
            )
        except Exception as e:
            errors.setdefault(iid, []).append(f"answer[{answer_model}]: {e}")
            answers[iid] = None
        if i % 10 == 0 or i == len(items):
            print(f"    {i}/{len(items)}")
    return answers


def run_batch(
    items: list[dict],
    embedder: Embedder,
    llm: LLMClient,
    hypothesis_models: list[str],
    answer_model: str,
    k: int,
    strategy: str,
) -> list[dict]:
    """Runs the full CMHA-Agent pipeline over `items`, model-major. Returns one
    scored result dict per item, in the same order as `items`."""
    errors: dict = {}

    para_vecs_by_id = _embed_paragraphs(items, embedder)

    diversity_by_id = {item["id"]: 0.0 for item in items}
    hyps_by_id = {item["id"]: [] for item in items}

    if strategy == "direct":
        query_vec_by_id = _embed_questions(items, embedder)
    else:
        models = hypothesis_models if strategy == "cmha" else hypothesis_models[:1]
        hyps_by_id = _generate_hypotheses(items, llm, models, errors)
        hyp_vecs_by_id = _embed_hypotheses(hyps_by_id, embedder)

        query_vec_by_id = {}
        fallback_needed = [item for item in items if len(hyp_vecs_by_id[item["id"]]) == 0]
        if fallback_needed:
            # every hypothesis call failed for this item — fall back to the raw question
            fallback_vecs = _embed_questions(fallback_needed, embedder)
            query_vec_by_id.update(fallback_vecs)
        for item in items:
            iid = item["id"]
            if iid in query_vec_by_id:
                continue
            vecs = hyp_vecs_by_id[iid]
            query_vec_by_id[iid] = _centroid(vecs)
            diversity_by_id[iid] = _diversity_score(vecs)

    evidence_by_id = _retrieve(items, para_vecs_by_id, query_vec_by_id, k)
    answers_by_id = _generate_answers(items, evidence_by_id, llm, answer_model, errors)

    results = []
    for item in items:
        iid = item["id"]
        predicted = answers_by_id.get(iid)
        if predicted is None:
            results.append(
                {
                    "id": iid,
                    "question": item["question"],
                    "error": "; ".join(errors.get(iid, ["unknown error"])),
                    "strategy": strategy,
                }
            )
            continue
        confidence = 1.0 / (1.0 + diversity_by_id[iid])
        record = {
            "id": iid,
            "question": item["question"],
            "gold_answer": item["answer"],
            "predicted_answer": predicted,
            "em": exact_match_score(predicted, item["answer"]),
            "f1": round(f1_score(predicted, item["answer"]), 4),
            "retrieval_recall": evidence_by_id[iid]["retrieval_recall"],
            "retrieved_titles": evidence_by_id[iid]["retrieved_titles"],
            "gold_titles": evidence_by_id[iid]["gold_titles"],
            "diversity_score": round(diversity_by_id[iid], 4),
            "confidence": round(confidence, 4),
            "hypotheses": hyps_by_id.get(iid, []),
            "strategy": strategy,
        }
        if iid in errors:
            record["partial_errors"] = errors[iid]  # e.g. one of four hypothesis models failed but answer still succeeded
        results.append(record)
    return results
