"""One-off script that fills the CSE598 capstone proposal template with the
CMHA-Agent proposal content. Not part of the baseline pipeline — kept here
only for provenance/reproducibility of how the .docx was generated.

Usage:
    python scripts/fill_proposal_docx.py <path-to-docx-copy>
"""

import sys

import docx


def delete_paragraph(paragraph):
    p = paragraph._p
    p.getparent().remove(p)
    paragraph._p = paragraph._element = None


def set_cell_text(cell, text):
    p = cell.paragraphs[0]
    for run in list(p.runs):
        run.text = ""
    if p.runs:
        p.runs[0].text = text
    else:
        p.add_run(text)
    # remove any extra paragraphs the cell might have had
    for extra in cell.paragraphs[1:]:
        delete_paragraph(extra)


def insert_lines(anchor, lines):
    """Insert (text, style) tuples immediately before `anchor`, in order."""
    for text, style in lines:
        anchor.insert_paragraph_before(text, style=style)


def main(path):
    d = docx.Document(path)
    paras = list(d.paragraphs)  # snapshot; index-stable for the deletes below

    # ---------- Basic Information table ----------
    t0 = d.tables[0]
    set_cell_text(t0.rows[1].cells[1], "Siddhesh More")
    set_cell_text(
        t0.rows[2].cells[1],
        "CMHA-Agent: A Confidence-Aware Cross-Model Retrieval Agent for Multi-Hop Question Answering",
    )
    set_cell_text(t0.rows[3].cells[1], "https://github.com/sidesshmore/cmha-agent-hotpotqa")
    set_cell_text(
        t0.rows[4].cells[1],
        ".env.example (repo root) and README.md, Setup section — no API key required by default (fully local via Ollama).",
    )

    BODY = "Body Text"
    BULLET = "Compact"

    # ---------- Section 1: Problem Definition (paras 6..15, keep 15 as spacer) ----------
    for i in range(7, 15):
        delete_paragraph(paras[i])
    insert_lines(
        paras[15],
        [
            (
                "The agent reads a multi-hop question together with a pool of candidate evidence "
                "passages, decides which passages actually support the answer, and returns a short "
                "answer string plus a numeric confidence score.",
                BODY,
            ),
            ("- Task: retrieve-then-answer over passages that require chaining two separate facts (bridge-type multi-hop QA), not single-paragraph lookup.", BULLET),
            ("- Intended user: someone building a QA assistant over a document collection (research literature, internal docs, etc.) who needs the system to work on questions that require connecting two facts, and to know when it doesn't know.", BULLET),
            ("- Input: a natural-language question plus a pool of candidate paragraphs (in this baseline, HotpotQA's own 10-paragraph distractor set per question; in a later phase, a larger real document corpus).", BULLET),
            ("- Output: a short answer string, the titles of the paragraphs used as evidence, and a confidence score in [0,1].", BULLET),
            (
                "- Success: the answer matches the gold answer (exact match / F1) AND confidence tracks correctness "
                "(high confidence should coincide with correct answers, low confidence with incorrect ones — "
                "measurable via correlation, not just accuracy alone). Failure: a wrong answer, or worse, a "
                "confidently wrong answer.",
                BULLET,
            ),
        ],
    )

    # ---------- Section 2: Motivation and Scope (paras 16..24, keep 24 spacer) ----------
    for i in range(17, 24):
        delete_paragraph(paras[i])
    insert_lines(
        paras[24],
        [
            (
                "Multi-hop QA is where retrieval-augmented systems actually break in practice: many real "
                "questions (\"who is the CEO of the company that acquired X\") need chaining facts across "
                "documents, and single-vector dense retrieval famously struggles here. My own prior work "
                "(“Beyond HyDE”) found that every single-model HyDE retrieval variant tested actually "
                "degrades accuracy below a no-retrieval baseline on multi-hop questions — this project "
                "closes the loop that paper left open by measuring whether cross-model retrieval improves the "
                "final answer, not just what gets retrieved.",
                BODY,
            ),
            (
                "- Why agentic: the right amount of retrieval effort is not fixed per question — some need one "
                "paragraph, others need evidence chained across documents. An agent that adaptively decides how "
                "much evidence to gather, and whether to abstain when uncertain, is a genuine control-flow "
                "problem, not a fixed pipeline.",
                BULLET,
            ),
            (
                "- In scope this semester: single-hop CMHA retrieval → generation → confidence estimate "
                "(built as this proposal's baseline); an adaptive stopping/second-hop retrieval policy (the "
                "actual agent-loop contribution — decide from confidence whether to retrieve again); a "
                "larger, paired-significance-tested evaluation on HotpotQA.",
                BULLET,
            ),
            (
                "- Out of scope: building a new document corpus/index from scratch, fine-tuning any model, "
                "production-grade serving, and live web tool-use (GAIA-style search) — flagged only as a "
                "possible stretch goal if time allows.",
                BULLET,
            ),
        ],
    )

    # ---------- Section 3: Runnable Baseline (paras 25..38, keep 38 spacer) ----------
    for i in range(26, 38):
        delete_paragraph(paras[i])
    insert_lines(
        paras[38],
        [
            (
                "This is a tool-use/workflow baseline: a retrieval step (Cross-Model Hypothesis Aggregation, "
                "CMHA) feeding a generation step, running entirely on local open-weight models via Ollama — "
                "no API key, no cloud dependency, and every model chosen to fit comfortably under 8GB RAM.",
                BODY,
            ),
            (
                "- Models/tools: four small open-weight LLMs served by Ollama for cross-model hypothesis "
                "generation (qwen2.5:3b, llama3.2:3b, gemma2:2b, phi3.5:3.8b — chosen to mirror the "
                "Alibaba/Meta/Google/Mistral-family organizational diversity of the original CMHA paper), "
                "nomic-embed-text (also via Ollama) for embeddings, and one of the four models for final "
                "answer generation. Plain Python + numpy for the retrieval math; no PyTorch, no GPU required.",
                BULLET,
            ),
            (
                "- Step by step: (1) each of the 4 models generates a short hypothetical answer passage for the "
                "question; (2) every hypothesis and every candidate paragraph is embedded; (3) the centroid of "
                "the 4 hypothesis embeddings ranks the paragraphs by cosine similarity; (4) the top-k paragraphs "
                "are handed to an answer-generation call; (5) the predicted answer is scored (EM/F1) and a "
                "confidence score is derived from how much the 4 hypotheses agreed with each other.",
                BULLET,
            ),
            (
                "- Why reasonable: it operationalizes and extends a retrieval method I already built and "
                "validated in peer-reviewed work (Beyond HyDE) rather than starting from an untested idea, "
                "reuses a validated confidence proxy (the same diversity score that predicted per-query "
                "difficulty at r=−0.53 in that paper) instead of inventing an uncalibrated one, and the code "
                "already runs the three conditions (direct query / single-model HyDE / full CMHA) the "
                "semester's evaluation plan needs.",
                BULLET,
            ),
            (
                "- Files: run_baseline.py (entry point); src/cmha_agent.py (retrieval + generation pipeline); "
                "src/embedder.py and src/llm_client.py (local Ollama clients); src/hotpot_metrics.py (EM/F1 "
                "scoring); data/hotpotqa_sample.json (30 frozen real HotpotQA questions). Full repo: "
                "https://github.com/sidesshmore/cmha-agent-hotpotqa",
                BULLET,
            ),
        ],
    )

    # ---------- Section 4: Test Case and Baseline Output (paras 39..47, keep 47 spacer) ----------
    for i in range(40, 47):
        delete_paragraph(paras[i])
    insert_lines(
        paras[47],
        [
            (
                "Sample input: “Which American film director hosted the 18th Independent Spirit Awards in "
                "2002?” — a real HotpotQA validation-set bridge question (id 5ac3165c5542995ef918c10a), "
                "given with its full 10-paragraph distractor pool.",
                BULLET,
            ),
            (
                "Expected behavior: the agent should retrieve both gold-supporting paragraphs (“18th "
                "Independent Spirit Awards” and “John Waters”) out of the 10 candidates and answer "
                "“John Waters.”",
                BULLET,
            ),
            (
                "Actual output (real run, not mocked): predicted_answer = “John Waters” (EM = 1.0, F1 = "
                "1.0), retrieval_recall = 1.0 (both gold paragraphs retrieved in the top-4), confidence = 0.838. "
                "Full JSON record: examples/test_case_real_output.jsonl in the repository.",
                BULLET,
            ),
            (
                "[SCREENSHOT PLACEHOLDER — paste a terminal screenshot here of: "
                "`python run_baseline.py --strategy cmha --limit 1` producing the output above. "
                "See examples/test_case.md in the repo for the exact reproduction command.]",
                BODY,
            ),
            (
                "What worked: even though all four individual hypothesis models hallucinated a different wrong "
                "host name (Kevin Smith, Jon Favreau, Quentin Tarantino, Spike Jonze), the cross-model centroid "
                "still retrieved the correct evidence, and the answer model correctly read the true answer off "
                "that evidence rather than trusting any single model's guess — concrete, real evidence for "
                "the cross-model-retrieval thesis, not just a citation of the original paper's claim.",
                BODY,
            ),
            (
                "What didn't: on a slightly larger real comparison (n=10 questions), full CMHA (EM 0.400) did "
                "not clearly beat the cheaper single-model-HyDE ablation (EM 0.500) — an honest, inconclusive "
                "result at this small sample size and with much smaller local models than the original paper "
                "used. This is exactly why Section 6's evaluation plan calls for a paired significance test on a "
                "larger sample rather than trusting a point estimate.",
                BODY,
            ),
        ],
    )

    # ---------- Section 5: Reproducibility (paras 48..59, keep 59 spacer) ----------
    for i in range(49, 59):
        delete_paragraph(paras[i])
    insert_lines(
        paras[59],
        [
            (
                "- Dependencies: Python 3.10+; pip packages in requirements.txt (requests, numpy — no "
                "PyTorch, no openai SDK, deliberately minimal); Ollama (local LLM server, https://ollama.com) with 5 small models pulled once "
                "(~8GB disk, one-time download).",
                BULLET,
            ),
            (
                "- API keys / env vars: none required by default — everything runs locally. An optional "
                "LLM_BASE_URL / LLM_API_KEY override is documented in .env.example only if you'd rather point "
                "the chat calls at a remote OpenAI-compatible endpoint instead.",
                BULLET,
            ),
            (
                "- Exact command: python run_baseline.py --strategy cmha --limit 10 (drop --limit for the full "
                "30-question set). A --mock flag is available for a zero-dependency pipeline smoke test that "
                "needs neither Ollama nor any models installed.",
                BULLET,
            ),
            ("- Input location: data/hotpotqa_sample.json (frozen, checked into the repo).", BULLET),
            ("- Output location: results/<strategy>_<timestamp>.jsonl — one JSON record per question.", BULLET),
            (
                "- Known setup limitations: the first run downloads ~8GB of Ollama models (one-time, needs "
                "internet once); each model swap costs ~3–5s so a full CMHA run over 30 questions takes "
                "~6 minutes (mitigated in the code by batching calls model-major rather than per-question — "
                "documented in README.md).",
                BULLET,
            ),
            ("Full step-by-step setup, command reference, and troubleshooting: see README.md in the repository.", BODY),
        ],
    )

    # ---------- Section 6: Initial Evaluation Plan (paras 60..73, keep 73 spacer) ----------
    for i in range(61, 73):
        delete_paragraph(paras[i])
    insert_lines(
        paras[73],
        [
            (
                "The baseline already runs three conditions with one flag change — direct query, "
                "single-model HyDE, and full CMHA — which is the core comparison for evaluating any future "
                "improvement:",
                BODY,
            ),
            ("- Task correctness: exact match and token F1 against gold HotpotQA answers.", BULLET),
            ("- Retrieval quality: recall of the gold supporting-paragraph titles at k=4.", BULLET),
            (
                "- Calibration: correlation between the confidence score and per-question correctness (reusing "
                "the same statistic Beyond-HyDE validated at r=−0.53 for difficulty prediction).",
                BULLET,
            ),
            (
                "- Cost: LLM calls per question (direct=1, single_hyde=2, cmha=5) and wall-clock time — "
                "already logged by the pipeline, since CMHA's accuracy gain (if any) has to be weighed against "
                "a real 4–5x compute cost.",
                BULLET,
            ),
            (
                "- Statistical rigor: because a small-sample run already produced a counter-intuitive result "
                "(single-hyde edging out full CMHA at n=10), any later claim that an improved system beats this "
                "baseline will use a paired bootstrap / significance test over a larger question set, following "
                "the same protocol as my prior work (Transfer or Noise?), rather than a bare point estimate.",
                BULLET,
            ),
            (
                "Later, once the adaptive multi-hop stopping policy is built, an LLM-as-judge or human-rated "
                "qualitative check will be added to evaluate whether the agent's decisions to re-retrieve (or "
                "abstain) are themselves reasonable, not just whether the final answer is correct.",
                BODY,
            ),
        ],
    )

    # ---------- Section 7: Limitations and Next Steps (paras 74..82, keep 82 spacer) ----------
    for i in range(75, 82):
        delete_paragraph(paras[i])
    insert_lines(
        paras[82],
        [
            (
                "- Known weaknesses: single-hop only right now (no adaptive re-retrieval yet); the confidence "
                "proxy is a validated correlate from prior work but not calibrated on this exact pipeline/dataset; "
                "small local models likely hallucinate more uniformly than the much larger models used in "
                "Beyond HyDE, which may explain why CMHA's advantage wasn't clean at n=10; results are written "
                "once per run rather than flushed per question, so a crash mid-run loses the whole in-flight batch.",
                BULLET,
            ),
            (
                "- Expected failure cases: bridging-entity questions where the first hop is already wrong "
                "(documented in Beyond-HyDE's own error analysis); questions needing information absent from "
                "the given paragraph pool entirely.",
                BULLET,
            ),
            (
                "- Next phase: build the adaptive stopping/second-hop retrieval policy (the actual agent-loop "
                "contribution); run the larger, paired-significance-tested evaluation from Section 6; possibly "
                "extend to a larger real document corpus or live tool-use (GAIA-style search) as a stretch goal.",
                BULLET,
            ),
            (
                "- Risks: small local models may put a hard ceiling on achievable accuracy regardless of "
                "pipeline improvements; the larger evaluation needs careful compute-time budgeting.",
                BULLET,
            ),
            (
                "- Help/resources: ASU Sol HPC access (already available via an existing group allocation) if "
                "scaling past small local models becomes necessary for the multi-hop extension.",
                BULLET,
            ),
        ],
    )

    d.save(path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/Users/sidessh/Agentic-AI/CSE598-capstone-proposal-Siddhesh-More.docx")
