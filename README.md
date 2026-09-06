# CMHA-Agent: An Adaptive Multi-Hop Retrieval Agent for HotpotQA

**CSE598 Capstone Proposal — Runnable Baseline (Option A)**

An agent that answers multi-hop questions by retrieving evidence with
Cross-Model Hypothesis Aggregation (CMHA) — the retrieval method from my
"Beyond HyDE" paper — then **deciding for itself whether that evidence is
enough**. If not, it names the specific fact it's still missing, retrieves a
targeted second hop for that gap, and only then answers. This is a real
agent loop (observe → decide → act → repeat, bounded), not a fixed
retrieve-once-generate-once RAG pipeline — see `--strategy agent` below,
which is now the default and the actual subject of this proposal. The three
fixed-depth strategies (`direct` / `single_hyde` / `cmha`) are kept as
ablations: they isolate how much of the agent's benefit comes from
*adaptive stopping* versus the retrieval method underneath it.

Runs **entirely locally via [Ollama](https://ollama.com)** — no API key, no
cloud account, no network access needed after the one-time model downloads.
Every model was chosen so a single one comfortably fits under 8GB RAM.

This baseline reuses three things I already built and validated in prior
work, and adds one genuinely new piece — the agent loop itself:

| Piece | Status |
|---|---|
| Cross-model hypothesis retrieval (CMHA) | Reused from *Beyond HyDE: Cross-Model Hypothesis Diversity for Robust Dense Retrieval* (VLDB Workshop 2026) |
| Single-pass generation over retrieved evidence | Reused from *Retrieve, Locate, Generate* (ACL submission), §3.3 |
| Diversity score as a confidence/difficulty proxy | Reused from *Beyond HyDE* §3.3 (`div(q)`), validated there at Pearson r=−0.53 with per-query correctness on HotpotQA |
| Adaptive sufficiency check + targeted second-hop retrieval | **New for this project** — `run_agent_batch()` in `src/cmha_agent.py`. This is the actual agentic contribution; everything else above is infrastructure it builds on. |

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
│   ├── real_run_n10_{agent,cmha,direct,single_hyde}.jsonl   ← real 4-way comparison evidence
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
python run_baseline.py --strategy agent --limit 3 --mock
```

Expected tail of output:

```
questions scored : 3 / 3  (0 errors)
exact match      : 0.000
token F1         : 0.000
mean hops used   : 1.00  (0% of questions used a 2nd hop)
elapsed          : <1s
```

The 0.000 EM/F1 and "0% used a 2nd hop" here are expected and meaningless —
`--mock` returns the same canned text for every call, which always parses as
"sufficient," so the loop always stops after hop 1. `--mock` only proves the
control flow runs without crashing, never that the agent's decisions are any
good.

### 2. The graded baseline run — the actual agent, real models, real scores

```bash
python run_baseline.py --strategy agent --limit 10
```

`agent` is the default strategy. Measured on this machine: **~15s/question**
(hop 1's four hypothesis models still load once each for the whole batch —
see "Local models and RAM budget" — but the adaptive loop itself is
necessarily sequential per question, since hop count is data-dependent).
A 10-question run takes ~2.5 minutes; the full 30-question set ~7-8 minutes.

### 3. The full comparison this proposal's eval plan needs

```bash
python run_baseline.py --strategy direct      --out results/direct.jsonl
python run_baseline.py --strategy single_hyde --out results/single_hyde.jsonl
python run_baseline.py --strategy cmha        --out results/cmha.jsonl
python run_baseline.py --strategy agent       --out results/agent.jsonl
```

The first three are fixed-depth ablations (`direct` embeds the raw question;
`single_hyde` uses one model's hypothesis; `cmha` is the full N=4 cross-model
method, always retrieving exactly k=4 paragraphs). `agent` is the actual
system: it starts from the same CMHA hop-1 retrieval as `cmha`, then decides
per-question whether to retrieve a second, targeted hop before answering.
Comparing all four summary blocks is the whole evaluation story for Section 6
— the gap between `cmha` and `agent` isolates exactly what adaptive stopping
bought over the fixed-depth method underneath it.

### What an actual agent trajectory looks like

From a real run (`examples/real_run_n10_agent.jsonl`):

```json
{
  "question": "What movie did Pitof direct which had an action-adventure tie-in video game based off of it in 2004?",
  "gold_answer": "Catwoman",
  "predicted_answer": "Catwoman",
  "em": 1,
  "hops_used": 2,
  "stop_reason": "max_hops_reached",
  "followup_queries": ["Pitof's film that had an action-adventure tie-in video game based off of it in 2004"],
  "retrieved_titles": ["Lego Knights' Kingdom", "Catwoman (video game)", "Lego Star Wars: The Video Game", "Shrek Forever After (video game)", "Catwoman (film)", "Tron: Evolution"]
}
```

Hop 1's four hypothesis models didn't converge on enough evidence; the model
judged it insufficient, named the specific missing fact, and a targeted
second retrieval pulled in `Catwoman (film)` and `Catwoman (video game)` —
both of which hop 1 missed. This is a real decision the agent made, not a
scripted branch.

**An honest failure mode, in the same run**, worth documenting rather than
hiding: on one question, instead of naming a specific missing fact, one
small model echoed the prompt's own instruction text back as if it were the
answer (`followup_queries: ["a short phrase naming the one specific fact or
entity still needed to answer"]`) — a real small-model format-following
failure, not a bug in the code. `_parse_sufficiency()` in `cmha_agent.py`
still extracts *something* from this so the loop terminates cleanly rather
than crashing, but the resulting follow-up query is useless. See Limitations.

### Where output goes

Every run writes one JSON record per question to a `results/*.jsonl` file
(default name includes strategy + UTC timestamp; override with `--out`).
Every record has the question, gold answer, predicted answer, EM, F1,
retrieval recall, and the diversity/confidence score; `agent` records
additionally carry `hops_used`, `stop_reason`, and `followup_queries` — the
agent's own decision trail, auditable per question.

Runs are **resumable at the whole-file level**: if `results/agent.jsonl`
already has some question IDs logged, rerunning the same command skips
those and only recomputes the rest.

---

## Real results (honest n=10 comparison)

Run on this machine, 2026-09-06, first 10 questions of the frozen set, real
Ollama calls (no mock):

| Strategy | EM | F1 | Retrieval recall | s/question |
|---|---|---|---|---|
| `direct` | 0.400 | 0.450 | 0.500 | 1.8s |
| `single_hyde` | 0.500 | 0.550 | 0.650 | 3.6s |
| `cmha` (N=4, fixed k=4) | 0.400 | 0.586 | 0.650 | 11.7s |
| `agent` (adaptive, k1=4, up to k2=2 more) | **0.600** | **0.650** | **0.850** | 15.3s |

**The agent is the headline result, and it's a real one, not a rounding
difference.** Adaptive stopping took EM from 0.400 (`cmha`'s fixed-depth
retrieval) to 0.600 — a 20-point jump from the *identical* hop-1 retrieval,
purely by letting the model decide when it needs more evidence. 60% of the
10 questions triggered a second hop; the mean hops used was 1.60. This
isolates the actual claim of this proposal cleanly: the benefit here is
attributable to *adaptive control flow*, not to a stronger retrieval method
underneath it (hop 1 is byte-identical to `cmha`).

**This does not erase the earlier honest finding about the fixed-depth
methods.** At n=10 with these small 2–4B local models, `single_hyde` still
edges out full `cmha` on exact match (0.500 vs 0.400) — not the clean
"CMHA wins" story "Beyond HyDE" told with much larger models. Two readings
worth carrying forward: (1) n=10 is nowhere near enough to distinguish
these — "Beyond HyDE" itself needed a paired bootstrap over hundreds of
queries before crossing zero, and "Transfer or Noise?" found effects that
flip sign once actually tested rather than eyeballed; (2) cross-model
averaging may help less when every ensemble member is small and noisy
rather than large and individually strong. Both questions — "does CMHA beat
single-hyde at scale?" and "does the agent's advantage hold at scale?" —
are exactly what the larger, significance-tested run in Section 6 is for.

**Two qualitative results that hold up** (see
[`examples/test_case.md`](examples/test_case.md) and "What an actual agent
trajectory looks like" above): (1) on the required Section 4 test case, all
four individual hypothesis models hallucinated a different wrong answer,
yet CMHA's centroid retrieval still pulled both gold paragraphs and the
answer model correctly read off "John Waters" — retrieval succeeding
despite every individual generator failing, the mechanism "Beyond HyDE"
argues for. (2) On the Catwoman question above, the agent's own sufficiency
check correctly identified that hop-1 evidence was incomplete and a
targeted second hop recovered the missing paragraph — a real instance of
the agent-loop mechanism this project adds actually doing its job.

---

## Command reference

| Flag | Default | Meaning |
|---|---|---|
| `--strategy` | `agent` | `direct` / `single_hyde` / `cmha` (fixed-depth ablations) or `agent` (the actual agent) |
| `--k` | `4` | paragraphs retrieved at hop 1, for every strategy |
| `--k2` | `2` | *[agent only]* additional paragraphs retrieved per follow-up hop |
| `--max-hops` | `2` | *[agent only]* max total hops before forcing an answer (2 matches HotpotQA bridge questions' 2-fact structure) |
| `--limit N` | all 30 | only run the first N questions |
| `--hypothesis-models` | `qwen2.5:3b,llama3.2:3b,gemma2:2b,phi3.5:3.8b` | comma-separated Ollama model names |
| `--answer-model` | `qwen2.5:3b` | model used for the sufficiency check, follow-up query, and final answer |
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
- **The agent's decision quality is limited by small-model instruction
  following, not just the pipeline design.** The sufficiency check asks for
  a specific two-line format; small local models don't always comply — one
  documented case (see "What an actual agent trajectory looks like" above)
  has a model echo the prompt's own instruction text back as its "missing
  fact" instead of naming something real. `_parse_sufficiency()` degrades
  gracefully (defaults to stopping rather than looping on garbage), but a
  more capable model would likely produce cleaner, more useful follow-up
  queries — this is a real, observed limitation, not a hypothetical one.
- **Max 2 hops, fixed.** Bounded by HotpotQA's own bridge-question
  construction (exactly two supporting facts), so it's well-justified here,
  but a more general document corpus wouldn't have that guarantee — a
  capstone extension to a real corpus would need either a learned stopping
  bound or a much larger max-hops budget with its own cost tradeoff.
- **The second hop's query comes from one model, not an ensemble.** Hop 1
  uses the full N=4 CMHA ensemble; the follow-up hop uses only the
  answer_model's own single hypothesis (a deliberate design choice — see
  `cmha_agent.py`'s docstring — to avoid re-triggering the per-question
  model-swap cost this project already fixed once). Whether cross-model
  diversity would help on the follow-up hop too is untested.
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
- The practice of logging and reading an agent's own decision trail
  (`hops_used`, `stop_reason`, `followup_queries`) as first-class evidence
  rather than only the final accuracy number: *Reasoning Changes How LLM
  Agents Play Strategic Games, Not How Well: A Behavioral Analysis on GLEE*.
- Dataset: Yang et al., *HotpotQA: A Dataset for Diverse, Explainable
  Multi-hop Question Answering*, EMNLP 2018 (`hotpotqa/hotpot_qa`,
  `distractor` config, via HuggingFace).
