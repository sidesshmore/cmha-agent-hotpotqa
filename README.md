# CMHA-Agent: Cross-Model Retrieve-and-Answer on HotpotQA

**CSE598 Capstone Proposal — Runnable Baseline (Option A)**

A multi-hop open-domain QA agent that retrieves evidence using Cross-Model
Hypothesis Aggregation (CMHA) — the retrieval method from my "Beyond HyDE"
paper — and then answers the question from that evidence, scored on exact
match / F1 rather than retrieval recall alone. This closes the loop that
"Beyond HyDE" left open: that paper showed CMHA improves *what gets
retrieved*, never whether it improves *the final answer*.

Runs **entirely locally via [Ollama](https://ollama.com)** — no API key, no
cloud account, no network access needed after the one-time model downloads.
Every model was chosen so a single one comfortably fits under 8GB RAM.

This baseline reuses, rather than re-derives, three things I already built
and validated in prior work:

| Piece | Reused from |
|---|---|
| Cross-model hypothesis retrieval (CMHA) | *Beyond HyDE: Cross-Model Hypothesis Diversity for Robust Dense Retrieval* (VLDB Workshop 2026) |
| Single-pass generation over retrieved evidence | *Retrieve, Locate, Generate* (ACL submission), §3.3 |
| Diversity score as a confidence/difficulty proxy | *Beyond HyDE* §3.3 (`div(q)`), validated there at Pearson r=−0.53 with per-query correctness on HotpotQA |

---

## What's actually in this repo

```
CapstoneProposal-CMHA/
├── README.md                  ← you are here
├── requirements.txt
├── .env.example                ← only needed if you override the default local setup
├── run_baseline.py             ← CLI entry point
├── src/
│   ├── cmha_agent.py            ← retrieval + generation pipeline (model-major batching)
│   ├── llm_client.py            ← OpenAI-compatible chat client (+ --mock mode)
│   ├── embedder.py              ← local embeddings via Ollama (+ --mock mode)
│   └── hotpot_metrics.py        ← official-style EM/F1 scoring
├── data/
│   └── hotpotqa_sample.json     ← 30 frozen real HotpotQA dev questions (see below)
├── examples/
│   ├── test_case.md             ← the one required concrete test case (proposal §4)
│   ├── test_case_real_output.jsonl
│   ├── real_run_n10_{cmha,direct,single_hyde}.jsonl   ← real 3-way comparison evidence
│   └── mock_pipeline_smoketest.jsonl
└── results/                    ← your run outputs land here (gitignored)
```

---

## Local models and RAM budget

Everything runs through one local Ollama server. Five small models, chosen
for cross-organization diversity (mirroring "Beyond HyDE"'s Alibaba/Meta/
Google/Mistral spread — Mistral's smallest official model is 7B, so it's
swapped for Microsoft/Phi here to stay well under budget):

| Role | Model | Org | Disk size |
|---|---|---|---|
| Hypothesis | `qwen2.5:3b` | Alibaba | ~1.9GB |
| Hypothesis | `llama3.2:3b` | Meta | ~2.0GB |
| Hypothesis | `gemma2:2b` | Google | ~1.6GB |
| Hypothesis | `phi3.5:3.8b` | Microsoft | ~2.2GB |
| Answer | `qwen2.5:3b` | (reused) | — |
| Embedding | `nomic-embed-text` | Nomic | ~274MB |

**Peak RAM, not total.** Ollama loads exactly one model into memory at a
time. Total disk footprint across all five models is ~8GB, but **peak RAM
during a run is only the size of whichever single model is currently
loaded** (largest here: phi3.5 at 2.2GB) — comfortably inside an 8GB-RAM
machine alongside the OS and Python process.

**Why the pipeline is staged model-major, not question-major.** Measured on
this machine: a repeat call to an already-loaded model takes ~0.3s; calling
a *different* model costs ~3–5s (Ollama has to swap it into RAM from disk).
A naive per-question loop (load hypothesis-model-A, B, C, D, then the
embedder, then the answer model, repeat per question) pays that swap cost
6–7 times *per question*. `cmha_agent.run_batch()` instead runs each model
across *every* question before moving to the next model, paying the swap
cost only 6–7 times for the *entire run*. Measured effect: this cut
per-question wall-clock time from ~31s to ~12s for `--strategy cmha` on this
machine. The tradeoff is documented in `cmha_agent.py`'s module docstring:
results are written to disk once per full run rather than after each
question, since staging by model doesn't produce per-question results until
the last stage completes.

---

## Why the dataset is a frozen 30-question file, not "download HotpotQA"

HotpotQA's distractor-config dev set already ships each question with its
own 10-paragraph pool (2 gold-supporting paragraphs + 8 distractors) — you
don't need a full-corpus Wikipedia index to reproduce this baseline; that
index only matters for "Beyond HyDE"'s original full-corpus retrieval
numbers. Here, retrieval is: given *this question's own* 10 paragraphs,
which ones does the agent pick?

`data/hotpotqa_sample.json` is a real, unmodified slice of HotpotQA's
official validation split (`hotpotqa/hotpot_qa`, `distractor` config,
`bridge`-type questions — the multi-hop kind CMHA was built for), fetched
via HuggingFace's `datasets-server` API and frozen into this repo so
**grading does not depend on any dataset download succeeding at grading
time.** The selection was deliberately spread across a wide range of
context lengths (1,951–8,624 characters) and de-duplicated by gold answer,
not cherry-picked for easy cases.

---

## Setup

### 1. Install Ollama

```bash
# macOS
brew install ollama
# or download from https://ollama.com/download for your OS
```

Start the server (some installers register it as a background service
automatically — check with `curl -s http://localhost:11434/api/version`
before manually starting it):

```bash
ollama serve &
```

### 2. Pull the five models (~8GB total download, one-time)

```bash
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull phi3.5:3.8b
ollama pull nomic-embed-text
```

### 3. Python environment

```bash
cd CapstoneProposal-CMHA
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No API key, no `.env` file, no signup needed for the default setup —
`.env.example` documents the (optional) override path if you'd rather point
the chat calls at a remote OpenAI-compatible endpoint instead.

---

## Running it

### 1. Smoke test — no Ollama, no server, no models needed, ~instant

Confirms the whole pipeline (data loading → embedding → ranking → scoring →
logging) runs end-to-end using canned text instead of any real model call —
useful as a first check even before installing Ollama at all:

```bash
python run_baseline.py --strategy cmha --limit 3 --mock
```

Expected tail of output:

```
questions scored : 3 / 3  (0 errors)
exact match      : 0.000
token F1         : 0.000
retrieval recall : ~0.2–0.3 (varies — see examples/test_case.md for why)
mean confidence  : 1.000
elapsed          : <1s
```

The 0.000 EM/F1 here is expected and meaningless — `--mock` never calls a
real model, so treat this only as a "does it run" check.

### 2. The graded baseline run — real models, real scores

```bash
python run_baseline.py --strategy cmha --limit 10
```

Measured on this machine: **~12s/question for `cmha`** (5 model-swaps
total, not per-question — see "Local models and RAM budget" above), so a
10-question run takes ~2 minutes and the full 30-question set takes ~6
minutes. `direct` and `single_hyde` are faster (fewer/no hypothesis calls —
measured at ~1.8s and ~3.6s per question respectively). Drop `--limit 10`
to run the full 30-question set.

### 3. The three-way comparison this proposal's eval plan needs

```bash
python run_baseline.py --strategy direct      --out results/direct.jsonl
python run_baseline.py --strategy single_hyde --out results/single_hyde.jsonl
python run_baseline.py --strategy cmha        --out results/cmha.jsonl
```

`direct` embeds the raw question (no LLM call at retrieval time). `single_hyde`
uses one model's hypothesis (the original HyDE). `cmha` is the full N=4
cross-model method. Comparing the three summary blocks these print is the
whole evaluation story for Section 6.

### Where output goes

Every run writes one JSON record per question to a `results/*.jsonl` file
(default name includes strategy + UTC timestamp; override with `--out`).
Each record has the question, gold answer, predicted answer, EM, F1,
retrieval recall against the gold paragraph titles, the diversity/confidence
score, and (for `cmha`/`single_hyde`) the raw generated hypotheses — enough
to fully audit any individual question after the fact.

Runs are **resumable at the whole-file level**: if `results/cmha.jsonl`
already has some question IDs logged, rerunning the same command skips
those and only recomputes the rest.

---

## Real results (honest n=10 comparison)

Run on this machine, 2026-09-05, first 10 questions of the frozen set, real
Ollama calls (no mock):

| Strategy | EM | F1 | Retrieval recall | s/question |
|---|---|---|---|---|
| `direct` | 0.400 | 0.450 | 0.500 | 1.8s |
| `single_hyde` | **0.500** | **0.550** | 0.650 | 3.6s |
| `cmha` (N=4) | 0.400 | 0.586 | 0.650 | 11.7s |

**This is not the clean "CMHA wins" story "Beyond HyDE" told on the full
BEIR benchmark with much larger models** (Qwen3-235B, Llama4-Maverick-400B,
etc.) — at n=10 with these small 2–4B local models, `single_hyde` actually
edges out full `cmha` on exact match. Two honest readings, both worth
carrying into the full proposal: (1) n=10 is nowhere near enough to
distinguish these — "Beyond HyDE" itself needed a paired bootstrap over
hundreds of queries before crossing zero, and even "Transfer or Noise?"
found effects that flip sign or lose significance once you actually test
them rather than eyeball a point estimate; (2) it's plausible that
cross-model averaging helps *less* when every model in the ensemble is
small and noisy rather than large and individually strong — the qualitative
story from Beyond-HyDE §6.4's failure analysis (bridging-entity confusion:
models agree on the right entity but disagree on the follow-up fact) may
behave differently when every model is more likely to *also* get the
first hop wrong. This is exactly the kind of question a larger-scale,
significance-tested run (the real capstone's Section 6 evaluation plan) is
for — this baseline's job is only to prove the comparison is measurable at
all, and it is.

The one qualitative result that *does* hold up (see
[`examples/test_case.md`](examples/test_case.md)): on the specific test
case required for Section 4, all four individual hypothesis models
hallucinated a different wrong answer, yet CMHA's centroid retrieval still
pulled both gold paragraphs (`retrieval_recall: 1.0`) and the answer model
correctly read off "John Waters" — a concrete instance of retrieval
succeeding despite every individual generator failing, which is the
mechanism "Beyond HyDE" argues for.

---

## Command reference

| Flag | Default | Meaning |
|---|---|---|
| `--strategy` | `cmha` | `direct` / `single_hyde` / `cmha` |
| `--k` | `4` | paragraphs retrieved per question |
| `--limit N` | all 30 | only run the first N questions |
| `--hypothesis-models` | `qwen2.5:3b,llama3.2:3b,gemma2:2b,phi3.5:3.8b` | comma-separated Ollama model names |
| `--answer-model` | `qwen2.5:3b` | model used for the final answer call |
| `--embed-model` | `nomic-embed-text` | Ollama embedding model name |
| `--mock` | off | skip Ollama entirely (pipeline smoke test only) |
| `--out` | auto-named | output JSONL path |

---

## Known limitations (proposal §7)

- **Small local models, not the paper's original models.** "Beyond HyDE"
  used models up to 235B/400B parameters; this baseline deliberately uses
  2–4B models to fit an 8GB-RAM budget. Absolute scores are not comparable
  between the two — only the *qualitative pipeline structure*
  (direct/single-hyde/cmha) is preserved. See the honest n=10 table above.
- **Single-hop only.** The agent retrieves once, then answers — it does not
  yet decide "I don't have enough evidence, let me issue a second query,"
  which is what would make this a genuine multi-hop *agent* rather than a
  RAG pipeline. That's the explicit Phase 2 scope for the full-semester
  project (see the proposal document, Section 2).
- **Confidence is a proxy, not calibrated.** `confidence = 1/(1+diversity)`
  reuses a validated *correlate* of correctness (Beyond HyDE's `div(q)`,
  r=−0.53 on HotpotQA) but has not itself been calibrated on this exact
  pipeline/dataset slice — that calibration is part of the semester
  evaluation plan, not this baseline.
- **Batched-by-model logging is not per-question-incremental.** Results for
  a whole run are written to disk together at the end (see "Local models
  and RAM budget" above for why), not flushed after each question. Fine for
  a few dozen local questions; would need reworking for a much larger run
  where losing an entire in-progress batch to a crash is costly.
- **`--mock` mode's retrieval is not representative.** Every hypothesis is
  an identical placeholder string, so retrieval in mock mode reduces to
  "what's lexically closest to one generic sentence," not a real test of
  CMHA. Mock mode only certifies that the code runs, never that the method
  works.
- **10–30 questions is a proposal-stage baseline, not a claim.** Section 6
  of the proposal specifies the larger-scale, paired-significance-tested
  evaluation planned for the full capstone.

---

## Citations

- Cross-Model Hypothesis Aggregation, the diversity score, and the
  thinking-model-breaks-HyDE finding: *Beyond HyDE: Cross-Model Hypothesis
  Diversity for Robust Dense Retrieval*, VLDB 2026 Workshop on Vector
  Databases.
- The retrieve→generate scoring structure and oracle-substitution framing
  this evaluation plan extends: *Retrieve, Locate, Generate: An
  Oracle-Substitution Diagnostic for Literature-Grounded QA*.
- The "point estimate vs. paired significance test" caution behind the
  honest n=10 discussion above: *Transfer or Noise? A Native-Scaffold
  Control for Meta-Optimized Agent Harnesses*.
- Dataset: Yang et al., *HotpotQA: A Dataset for Diverse, Explainable
  Multi-hop Question Answering*, EMNLP 2018 (`hotpotqa/hotpot_qa`,
  `distractor` config, via HuggingFace).
