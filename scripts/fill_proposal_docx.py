"""One-off script that fills the CSE598 capstone proposal template with the
CMHA-Agent proposal content. Not part of the baseline pipeline — kept here
only for provenance/reproducibility of how the .docx was generated.

Usage:
    python scripts/fill_proposal_docx.py <path-to-blank-template> <path-to-output-docx>
"""

import shutil
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
    for extra in cell.paragraphs[1:]:
        delete_paragraph(extra)


def insert_lines(anchor, lines):
    """Insert (text, style) tuples immediately before `anchor`, in order."""
    for text, style in lines:
        anchor.insert_paragraph_before(text, style=style)


def main(template_path, out_path):
    shutil.copy(template_path, out_path)
    d = docx.Document(out_path)
    paras = list(d.paragraphs)  # snapshot; index-stable for the deletes below

    BODY = "Body Text"
    BULLET = "Compact"

    # ---------- Basic Information table ----------
    t0 = d.tables[0]
    set_cell_text(t0.rows[1].cells[1], "Siddhesh More")
    set_cell_text(
        t0.rows[2].cells[1],
        "CMHA-Agent: An Adaptive Multi-Hop Retrieval Agent for Open-Domain Question Answering",
    )
    set_cell_text(t0.rows[3].cells[1], "https://github.com/sidesshmore/cmha-agent-hotpotqa")
    set_cell_text(
        t0.rows[4].cells[1],
        ".env.example (repo root) and README.md, Setup section — no API key required by default (fully local via Ollama).",
    )

    # ---------- Section 1: Problem Definition (paras 6..15, keep 15 as spacer) ----------
    for i in range(7, 15):
        delete_paragraph(paras[i])
    insert_lines(
        paras[15],
        [
            (
                "Task: answer a multi-hop question by retrieving evidence, deciding whether it's enough, "
                "and retrieving a further targeted hop if not — the continue-vs-answer decision is the "
                "agentic behavior being proposed. User: someone building QA over a document collection who "
                "needs multi-fact questions answered and needs the system to know when it lacks evidence.",
                BULLET,
            ),
            ("- Input: a question + a pool of candidate paragraphs (HotpotQA's 10-paragraph set per question here; a larger corpus later).", BULLET),
            ("- Output: a short answer, the evidence actually used (possibly across 2 hops), why it stopped, and a confidence score.", BULLET),
            (
                "- Success: correct answer (EM/F1) AND a sound stopping decision (takes a 2nd hop only when "
                "hop-1 evidence is genuinely incomplete). Failure: wrong answer, or a confident wrong answer.",
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
                "Multi-hop QA is where RAG systems break in practice — my own prior work (“Beyond HyDE”) "
                "found every single-model HyDE variant tested degrades accuracy below no-retrieval on "
                "multi-hop questions. A fixed-depth retriever can't adapt: some questions need one paragraph, "
                "others need a second chained fact the first pass misses.",
                BODY,
            ),
            (
                "- Why agentic: this is control flow, not a fixed pipeline — observe evidence, decide if it's "
                "enough, act. Already measured, not just planned: with an identical hop-1 retrieval, letting "
                "the model decide whether to take a 2nd hop raised exact match from 0.400 to 0.600 on a real "
                "10-question run (Section 6).",
                BULLET,
            ),
            (
                "- In scope: the adaptive stopping policy (built, measured); a larger significance-tested "
                "evaluation; improving the sufficiency-check's format reliability (a real observed failure — Section 7).",
                BULLET,
            ),
            ("- Out of scope: a new corpus/index, fine-tuning, production serving, live web tool-use (stretch goal only).", BULLET),
        ],
    )

    # ---------- Section 3: Runnable Baseline (paras 25..38, keep 38 spacer) ----------
    for i in range(26, 38):
        delete_paragraph(paras[i])
    insert_lines(
        paras[38],
        [
            (
                "Tool-use/workflow baseline built around a real agent loop (observe → decide → act → repeat, "
                "bounded), running entirely on local open-weight models via Ollama — no API key, no cloud "
                "dependency, every model under 8GB RAM.",
                BODY,
            ),
            (
                "- Models: 4 small LLMs for cross-model hop-1 hypotheses (qwen2.5:3b, llama3.2:3b, gemma2:2b, "
                "phi3.5:3.8b — mirroring Beyond-HyDE's org diversity), nomic-embed-text for embeddings, one "
                "model for the sufficiency check/follow-up/final answer. Plain Python + numpy; no PyTorch, no GPU.",
                BULLET,
            ),
            (
                "- Steps: (1) 4 models generate hop-1 hypotheses; (2) their embedding centroid ranks "
                "paragraphs (= hop-1 evidence, identical to the cmha ablation); (3) agent judges if that's "
                "enough to answer; (4) if not, names the missing fact, retrieves a targeted 2nd hop "
                "(excluding seen paragraphs), re-checks — capped at 2 hops (HotpotQA bridge questions need "
                "exactly two facts); (5) final answer generated and scored (EM/F1) with a confidence score.",
                BULLET,
            ),
            (
                "- Why reasonable: extends peer-reviewed retrieval work (Beyond HyDE) rather than an untested "
                "idea, reuses a validated confidence proxy (r=−0.53 there), and the loop is already measurably "
                "working — +20pp exact match over identical hop-1 retrieval once the model can ask for more evidence.",
                BULLET,
            ),
            (
                "- Files: run_baseline.py (entry, --strategy agent/cmha/single_hyde/direct); "
                "src/cmha_agent.py (run_agent_batch = agent loop); src/embedder.py, src/llm_client.py (Ollama "
                "clients); data/hotpotqa_sample.json. Repo: https://github.com/sidesshmore/cmha-agent-hotpotqa",
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
                "Case A (1 hop, real run): “Which American film director hosted the 18th Independent Spirit "
                "Awards in 2002?” (gold: “John Waters”). Output: “John Waters”, EM=1.0, F1=1.0, "
                "retrieval_recall=1.0, hops_used=1, stop_reason=“sufficient”. All 4 hop-1 models individually "
                "hallucinated a different wrong host — centroid retrieval still found the right evidence, and "
                "the agent correctly judged 1 hop was enough.",
                BULLET,
            ),
            (
                "Case B (2 hops, real run): “What movie did Pitof direct which had an action-adventure tie-in "
                "video game based off of it in 2004?” (gold: “Catwoman”). Hop 1 was insufficient; the agent "
                "named the missing fact, retrieved a targeted 2nd hop, recovered the 2 paragraphs hop-1 "
                "missed, and answered correctly: EM=1.0, hops_used=2.",
                BULLET,
            ),
            (
                "[SCREENSHOT PLACEHOLDER — terminal screenshot of Case A: "
                "`python run_baseline.py --strategy agent --limit 1`. Exact reproduction command in "
                "examples/test_case.md.]",
                BODY,
            ),
            (
                "What worked: both cases are real, unedited output — evidence that cross-model retrieval "
                "survives individual hallucination, and that the 2nd-hop mechanism recovers evidence a "
                "fixed-depth retriever would miss.",
                BODY,
            ),
            (
                "What didn't: the sufficiency check's format isn't always followed by these small models — one "
                "run had a model echo the prompt's own instructions back as its \"missing fact\" (parser "
                "degrades safely, but that follow-up query was useless). At n=10, the fixed-depth ablations "
                "alone don't show a clean cmha-over-single_hyde win — see Section 6.",
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
            ("- Dependencies: Python 3.10+; requirements.txt (requests, numpy only); Ollama with 5 small models pulled once (~8GB disk, one-time).", BULLET),
            ("- API keys: none required — everything runs locally. Optional remote-endpoint override documented in .env.example.", BULLET),
            ("- Command: python run_baseline.py --strategy agent --limit 10 (default strategy; drop --limit for all 30). --mock available for a zero-dependency smoke test.", BULLET),
            ("- Input: data/hotpotqa_sample.json (frozen, in repo). Output: results/<strategy>_<timestamp>.jsonl, one record per question.", BULLET),
            (
                "- Setup limitations: ~8GB one-time model download; hop-1 models are batched to minimize "
                "swap overhead, but the adaptive loop is sequential per question, so a full agent run over 30 "
                "questions takes ~7–8 minutes (documented in README.md).",
                BULLET,
            ),
            ("Full setup, commands, troubleshooting: README.md in the repository.", BODY),
        ],
    )

    # ---------- Section 6: Initial Evaluation Plan (paras 60..73, keep 73 spacer) ----------
    for i in range(61, 73):
        delete_paragraph(paras[i])
    insert_lines(
        paras[73],
        [
            (
                "Four conditions, one flag change: direct, single_hyde, cmha (fixed-depth) vs. agent "
                "(adaptive) — the core comparison for this proposal's claim that adaptive stopping, not just "
                "better retrieval, improves multi-hop QA. Real n=10 result: direct EM 0.400, single_hyde 0.500, "
                "cmha 0.400, agent 0.600 (retrieval recall 0.850, 60% used a 2nd hop). Since agent's hop 1 is "
                "identical to cmha's, the 0.400→0.600 gap isolates the agent's own decision-making from the "
                "retrieval method underneath it.",
                BODY,
            ),
            ("- Task correctness (EM/F1) and retrieval recall across all four strategies.", BULLET),
            ("- Agent-specific: hops_used distribution, stop-reason breakdown, and whether hops_used correlates with correctness (i.e. is the 2nd hop triggered on questions that actually need it).", BULLET),
            ("- Calibration: confidence vs. correctness correlation (reusing Beyond-HyDE's validated r=−0.53 statistic).", BULLET),
            ("- Cost: LLM calls/question (1/2/5/5–7) and wall-clock time, since any gain must be weighed against real extra compute.", BULLET),
            (
                "- Statistical rigor: n=10 isn't enough to trust these point estimates (single_hyde already "
                "flips against the original paper's finding at this size) — the full evaluation uses a paired "
                "bootstrap/significance test, following my prior work (Transfer or Noise?).",
                BULLET,
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
                "- Weaknesses: small models don't reliably follow the sufficiency check's format (observed "
                "real failure — Section 4); max_hops fixed at 2 (justified for HotpotQA, won't generalize "
                "elsewhere); follow-up hop uses 1 model not the 4-way ensemble (cost tradeoff, untested "
                "accuracy cost); results flush once per run, not per question.",
                BULLET,
            ),
            ("- Expected failures: model confidently misjudges hop-1 as sufficient; a 2nd-hop query built on a wrong guess about what's missing.", BULLET),
            (
                "- Next: the larger significance-tested evaluation (Section 6); more reliable sufficiency-check "
                "parsing/model; possibly a larger corpus or live tool-use (GAIA-style) as a stretch goal.",
                BULLET,
            ),
            ("- Risks: small models may cap achievable accuracy and format-following regardless of pipeline improvements; larger eval needs compute-time budgeting given the agent's per-question sequential loop.", BULLET),
            ("- Help/resources: ASU Sol HPC access (existing group allocation) if scaling past small local models is needed.", BULLET),
        ],
    )

    d.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    template = sys.argv[1] if len(sys.argv) > 1 else "/Users/sidessh/Agentic-AI/CSE598-capstone-proposal-template.docx"
    out = sys.argv[2] if len(sys.argv) > 2 else "/Users/sidessh/Agentic-AI/CSE598-capstone-proposal-Siddhesh-More.docx"
    main(template, out)
