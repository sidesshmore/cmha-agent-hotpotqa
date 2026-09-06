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
                "I'm building an agent that answers questions requiring two connected facts — the kind "
                "where one search isn't enough. It retrieves some evidence, decides for itself whether that's "
                "actually enough to answer confidently, and if not, figures out what's missing and retrieves "
                "again before answering. That decision — continue or answer — is the agentic part.",
                BODY,
            ),
            (
                "The intended user is anyone building a QA system over a document collection who needs "
                "multi-fact questions handled correctly, and who'd rather the system say \"I need more "
                "information\" than guess. The input is a question plus a pool of candidate paragraphs "
                "(HotpotQA supplies 10 per question in this baseline). The output is a short answer, the "
                "evidence actually used, an account of why it stopped searching when it did, and a "
                "confidence score.",
                BODY,
            ),
            (
                "It succeeds when the answer is correct and its stopping decision was sound — taking a "
                "second look only when the first one genuinely wasn't enough. It fails on a wrong answer, or "
                "worse, a confidently wrong one.",
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
                "earlier work (“Beyond HyDE”), I found that every single-model retrieval variant I tested "
                "actually did worse than not retrieving at all on multi-hop questions. A retriever that "
                "always grabs the same fixed number of paragraphs can't adapt to that — some questions really "
                "do need just one paragraph, and others need a second fact the first search will simply miss.",
                BODY,
            ),
            (
                "That's why this needs to be agentic rather than a fixed pipeline: the system has to look at "
                "its own evidence, judge whether it's enough, and act on that judgment. And this isn't just a "
                "plan — I already built and measured it. Using the identical first-round retrieval, letting "
                "the model decide whether to look again took exact match from 0.400 to 0.600 on a real "
                "10-question test run (details in Section 6).",
                BODY,
            ),
            (
                "This semester I'm scoping to the adaptive stopping behavior (built and measured), a properly "
                "sized statistical evaluation, and making the model's \"what's missing\" reports more reliable "
                "(a real failure case is in Section 7). I'm leaving out a new document collection, fine-tuning, "
                "production deployment, and live web search — the last one only as a stretch goal.",
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
                "This is a tool-use baseline built around an actual agent loop — observe, decide, act, "
                "repeat, with a limit — and it runs entirely on local, open-weight models through Ollama. No "
                "API key, no cloud dependency, and every model fits comfortably under 8GB of RAM.",
                BODY,
            ),
            (
                "Four small models (qwen2.5:3b, llama3.2:3b, gemma2:2b, phi3.5:3.8b — chosen to mirror the "
                "cross-company diversity from Beyond HyDE) each generate a guess at the answer for the first "
                "search; nomic-embed-text turns text into vectors; and one model handles deciding whether "
                "evidence is sufficient, writing a follow-up query, and producing the final answer. Everything "
                "else is plain Python and numpy — no PyTorch, no GPU needed.",
                BODY,
            ),
            (
                "Concretely: the four models each guess an answer; averaging those guesses ranks the "
                "candidate paragraphs, giving round-one evidence (identical to the cmha ablation below). The "
                "agent judges whether that's enough — if not, it names what's missing, retrieves a targeted "
                "second round excluding paragraphs it's already seen, and checks again, capped at two rounds "
                "since HotpotQA's questions need exactly two facts. It then answers from whatever evidence it "
                "has and reports a confidence value.",
                BODY,
            ),
            (
                "I think this is reasonable because it builds on retrieval work I already validated in a "
                "peer-reviewed paper, reuses a confidence measure I already checked correlates with "
                "correctness, and the agent loop itself is already measurably working — twenty points of exact "
                "match gained purely by letting the model decide it needs more evidence, from the identical first search.",
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
                "Awards in 2002?” (correct answer: John Waters). The agent answered correctly — exact match, "
                "F1 of 1.0, both supporting paragraphs found after one search. What I find genuinely "
                "interesting: all four models generating first-round guesses individually named a different "
                "wrong person (Kevin Smith, Jon Favreau, Quentin Tarantino, Spike Jonze), yet averaging their "
                "guesses still pointed retrieval at the right evidence, and the agent correctly judged one "
                "search was enough.",
                BODY,
            ),
            (
                "A second real run shows the actual multi-hop behavior firing: “What movie did Pitof direct "
                "which had an action-adventure tie-in video game based off of it in 2004?” (correct answer: "
                "Catwoman). The first search wasn't enough; the agent said what it still needed to know, "
                "searched again specifically for that, found the two paragraphs the first search had missed, "
                "and answered correctly.",
                BODY,
            ),
            (
                "[SCREENSHOT PLACEHOLDER — a terminal screenshot of test case A, from running "
                "`python run_baseline.py --strategy agent --limit 1`. Exact steps in examples/test_case.md.]",
                BODY,
            ),
            (
                "What worked: both of these are real, unedited output, not constructed to look good — solid "
                "evidence that averaging several models' guesses can survive every one of them being wrong "
                "individually, and that the follow-up search genuinely recovers evidence the first pass missed.",
                BODY,
            ),
            (
                "What didn't work as cleanly: the model doesn't always describe what's missing in a usable "
                "way — in one run it just echoed the instructions back instead of naming something real. And "
                "with only ten questions tested, the simpler retrieval methods alone (without the agent loop) "
                "don't show a clean winner yet — more on that in Section 6.",
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
                "You'll need Python 3.10 or newer and Ollama installed (free, runs locally). The Python side "
                "only needs two small packages — requests and numpy, listed in requirements.txt — and Ollama "
                "needs five small models pulled once, about 8GB total on disk. No API key or account is "
                "needed anywhere; everything runs on your own machine.",
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
                "One real setup cost worth knowing about: the first run downloads about 8GB of models, and "
                "because the agent has to decide per-question whether it needs a second look, a full 30-question "
                "run takes roughly 7-8 minutes rather than being instant. Full setup steps and troubleshooting "
                "are in README.md.",
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
                "fixed amount and never decide anything, and the actual agent. That comparison is the whole "
                "point — it tells me whether the gain comes from letting the model decide, not from a better "
                "retrieval trick. On a real 10-question run: direct 0.400 exact match, single_hyde 0.500, cmha "
                "0.400, agent 0.600, with 60% of questions triggering a second search. Since the agent's first "
                "search is identical to cmha's, that 0.400-to-0.600 jump is specifically what the "
                "decision-making is worth.",
                BODY,
            ),
            (
                "Going forward I'll track EM/F1 and retrieval recall across all four versions; for the agent, "
                "whether a second look actually correlates with getting the answer right rather than just "
                "adding noise; how well its confidence tracks correctness; and the extra cost in model calls "
                "and time.",
                BODY,
            ),
            (
                "Most importantly, I want a real statistical test rather than trusting small numbers — ten "
                "questions already produced one surprising result (single_hyde matching the full method), and "
                "I don't want to draw conclusions from a sample that small. The full evaluation will use a "
                "paired significance test over a much larger question set, the same approach from my earlier work.",
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
                "The biggest honest weakness: the model doesn't always describe what's missing in a usable "
                "way — small local models aren't perfectly reliable at following that instruction, and I've "
                "seen it fail already. The two-hop limit is also dataset-specific — it works because "
                "HotpotQA's questions need exactly two facts, but a general document collection wouldn't "
                "guarantee that. And the follow-up search uses one model instead of all four, purely to stay "
                "fast — untested whether that costs accuracy.",
                BODY,
            ),
            (
                "I expect failures where the model is confidently wrong about having enough evidence, or its "
                "second search is built on a mistaken guess about what's missing.",
                BODY,
            ),
            (
                "Next: the larger statistical evaluation from Section 6, a more reliable sufficiency check, "
                "and possibly a bigger document collection or live web search if time allows. The main risk is "
                "that small models may just cap how good this gets, and the larger evaluation needs careful "
                "time budgeting since the agent reasons about each question individually. I have access to "
                "ASU's Sol HPC cluster if more compute becomes necessary.",
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
