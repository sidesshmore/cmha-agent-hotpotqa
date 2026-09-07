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
        ".env.example (repo root) and README.md, Setup section. No API key required by default, since it runs fully locally via Ollama.",
    )

    # ---------- Section 1: Problem Definition (paras 6..15, keep 15 as spacer) ----------
    for i in range(7, 15):
        delete_paragraph(paras[i])
    insert_lines(
        paras[15],
        [
            (
                "I'm building an agent that answers questions requiring two connected facts, the kind of "
                "question where one search isn't enough. It retrieves some evidence, decides for itself "
                "whether that's actually enough to answer confidently, and if not, figures out what's missing "
                "and retrieves again before answering. That decision, whether to continue searching or "
                "answer now, is the agentic part.",
                BODY,
            ),
            (
                "The intended user is anyone building QA over a document collection who needs multi-fact "
                "questions handled correctly, and would rather the system say \"I need more information\" "
                "than guess. The input is a question plus candidate paragraphs (HotpotQA supplies 10 per "
                "question here). The output is a short answer, the evidence used, why it stopped searching, "
                "and a confidence score.",
                BODY,
            ),
            (
                "It succeeds when the answer is correct and its stopping decision was sound, meaning it took "
                "a second look only when the first one genuinely wasn't enough. It fails on a wrong answer, "
                "or worse, a confidently wrong one.",
                BODY,
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
                "Multi-hop questions are exactly where retrieval-augmented systems tend to break. In my own "
                "earlier work (“Beyond HyDE”), every single-model retrieval variant I tested actually did "
                "worse than not retrieving at all on multi-hop questions. A retriever that always grabs the "
                "same fixed number of paragraphs can't adapt: some questions need one paragraph, others need "
                "a second fact the first search will miss.",
                BODY,
            ),
            (
                "That's why this needs to be agentic rather than a fixed pipeline, and this isn't just a "
                "plan. Using the identical first-round retrieval, letting the model decide whether to look "
                "again took exact match from 0.400 to 0.600 on a real 10-question test run, detailed in "
                "Section 6.",
                BODY,
            ),
            (
                "This semester I'm scoping to the adaptive stopping behavior, already built and measured, a "
                "properly sized statistical evaluation, and making the model's reports of what's missing more "
                "reliable, since a real failure case shows up in Section 7. I'm leaving out a new document "
                "collection, fine-tuning, production deployment, and live web search, with that last one kept "
                "only as a possible stretch goal.",
                BODY,
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
                "This is a tool-use baseline built around an actual agent loop, meaning it observes, "
                "decides, acts, and repeats up to a limit, and it runs entirely on local, open-weight models "
                "through Ollama. No API key, no cloud dependency, and every model fits comfortably under 8GB "
                "of RAM.",
                BODY,
            ),
            (
                "Four small models (qwen2.5:3b, llama3.2:3b, gemma2:2b, phi3.5:3.8b, chosen to mirror the "
                "cross-company diversity from Beyond HyDE) each guess at the answer for the first search. "
                "Nomic-embed-text turns text into vectors, and one model decides sufficiency, writes the "
                "follow-up query, and produces the final answer. Everything else is plain Python and numpy, "
                "no PyTorch, no GPU.",
                BODY,
            ),
            (
                "Concretely, the four models each guess an answer, and averaging those guesses ranks the "
                "candidate paragraphs, giving round-one evidence identical to the cmha ablation below. The "
                "agent judges whether that's enough. If not, it names what's missing, retrieves a targeted "
                "second round excluding paragraphs already seen, and checks again, capped at two rounds since "
                "HotpotQA's questions need exactly two facts. It then answers and reports a confidence value.",
                BODY,
            ),
            (
                "I think this is reasonable because it builds on retrieval work I already validated in a "
                "peer-reviewed paper, reuses a confidence measure I already checked correlates with "
                "correctness, and the agent loop itself is already measurably working: twenty points of exact "
                "match gained purely by letting the model decide it needs more evidence, from the identical "
                "first search.",
                BODY,
            ),
            (
                "Everything lives in run_baseline.py and src/cmha_agent.py (agent loop plus the simpler "
                "fixed-depth versions), with small helpers for Ollama and scoring. Full code: "
                "https://github.com/sidesshmore/cmha-agent-hotpotqa",
                BODY,
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
                "Test case A, a real run: “Which American film director hosted the 18th Independent Spirit "
                "Awards in 2002?” (correct answer: John Waters). The agent answered correctly, with both "
                "supporting paragraphs found after one search. Interestingly, all four models individually "
                "named a different wrong person (Kevin Smith, Jon Favreau, Quentin Tarantino, Spike Jonze), "
                "yet averaging their guesses still pointed retrieval at the right evidence.",
                BODY,
            ),
            (
                "A second real run shows the actual multi-hop behavior firing: “What movie did Pitof direct "
                "which had an action-adventure tie-in video game based off of it in 2004?” (correct answer: "
                "Catwoman). The first search wasn't enough, so the agent said what it still needed to know, "
                "searched again specifically for that, found the two paragraphs the first search had missed, "
                "and answered correctly.",
                BODY,
            ),
        ],
    )
    insert_lines(
        paras[47],
        [
            (
                "Terminal output from reproducing test case A (`python run_baseline.py --data /tmp/single_case.json "
                "--strategy cmha --out examples/test_case_rerun.jsonl`, exact steps in examples/test_case.md): "
                "exact match 1.000, retrieval recall 1.000, matching the run described above. Full screenshot in "
                "the repo README.",
                BODY,
            ),
            (
                "What worked: both of these are real, unedited output, not constructed to look good. That's "
                "solid evidence that averaging several models' guesses can survive every one of them being "
                "wrong individually, and that the follow-up search genuinely recovers evidence the first pass "
                "missed.",
                BODY,
            ),
            (
                "What didn't work as cleanly: the model doesn't always describe what's missing in a usable "
                "way, and in one run it just echoed the instructions back instead of naming something real. "
                "With only ten questions tested so far, the simpler retrieval methods alone, without the agent "
                "loop, also don't show a clean winner yet, which I discuss more in Section 6.",
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
                "You'll need Python 3.10+ and Ollama installed locally, free, no API key required. Python "
                "needs just requests and numpy (requirements.txt); Ollama needs five small models pulled "
                "once, about 8GB total on disk.",
                BODY,
            ),
            (
                "To run it: python run_baseline.py --strategy agent --limit 10 (agent is the default, so you "
                "can drop that flag; drop --limit to run all 30 questions). There's also a --mock flag that "
                "skips Ollama entirely, just to confirm the code itself runs before installing anything.",
                BODY,
            ),
            (
                "The input questions live in data/hotpotqa_sample.json, already included in the repo. Output "
                "goes to results/, one JSON file per run with one line per question.",
                BODY,
            ),
            (
                "One real setup cost: the first run downloads about 8GB of models, and since the agent "
                "decides per-question whether to search again, a full 30-question run takes roughly 7-8 "
                "minutes. Full setup and troubleshooting steps are in README.md. There's also an optional "
                "Streamlit app (app.py) that runs the same agent loop on one question at a time, live.",
                BODY,
            ),
        ],
    )

    # ---------- Section 6: Initial Evaluation Plan (paras 60..73, keep 73 spacer) ----------
    for i in range(61, 73):
        delete_paragraph(paras[i])
    insert_lines(
        paras[73],
        [
            (
                "The code already runs four versions with one flag change: three that always retrieve a "
                "fixed amount and never decide anything, and the actual agent. That comparison tells me "
                "whether the gain comes from letting the model decide, not from a better retrieval trick. On a "
                "real 10-question run: direct scored 0.400 exact match, single_hyde 0.500, cmha 0.400, and "
                "agent 0.600, with 60% of questions triggering a second search. That jump reflects the "
                "decision-making alone, since agent's first search is identical to cmha's.",
                BODY,
            ),
            (
                "Going forward I'll track EM/F1, retrieval recall, whether a second look correlates with "
                "correctness, and the extra cost in model calls and time. Most importantly, I want a real "
                "statistical test rather than trusting small numbers, since ten questions already produced "
                "one surprising result (single_hyde matching the full method); the full evaluation will use "
                "a paired significance test over a larger question set, as in my earlier work.",
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
                "The biggest weakness: the model doesn't always describe what's missing in a usable way, "
                "and I've already seen it fail. The two-hop limit is also dataset-specific, since it works "
                "because HotpotQA needs exactly two facts, but a general document collection wouldn't "
                "guarantee that. I also expect failures where the model is confidently wrong about having "
                "enough evidence, or its second search is built on a mistaken guess about what's missing. "
                "Next: the larger statistical evaluation from Section 6, a more reliable sufficiency check, "
                "and possibly a bigger document collection or live web search, using ASU's Sol HPC cluster "
                "if more compute is needed.",
                BODY,
            ),
        ],
    )

    d.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    template = sys.argv[1] if len(sys.argv) > 1 else "/Users/sidessh/Agentic-AI/CSE598-capstone-proposal-template.docx"
    out = sys.argv[2] if len(sys.argv) > 2 else "/Users/sidessh/Agentic-AI/CSE598-capstone-proposal-Siddhesh-More.docx"
    main(template, out)
