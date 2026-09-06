# CMHA-Agent: An Adaptive Multi-Hop Retrieval Agent for HotpotQA

**CSE598 Capstone Proposal — Runnable Baseline (Option A)**

This is a small agent that answers questions which need two connected facts to solve — the "who directed the movie that had a video game based on it" kind of question, where you can't just search once and be done. It retrieves some evidence, and then — this is the important part — it actually stops and asks itself whether that evidence is enough. If it isn't, it figures out what's missing, goes and looks for that specific thing, and only then answers.

That "check yourself, go back if you need to" step is the whole point of this project. Everything else here (the retrieval method, the confidence score) is infrastructure I already built and tested in earlier work of mine; the agent loop on top of it is new, and it's the actual thing being proposed. The three simpler strategies in the code (`direct`, `single_hyde`, `cmha`) exist so I can show exactly how much of the improvement comes from the agent's decision-making versus the retrieval method underneath it.

It all runs on your own laptop through [Ollama](https://ollama.com) — no API key, no account, nothing to pay for, and every model is small enough to run comfortably even with 8GB of RAM.

Three things I'm reusing from earlier work, and one new piece:

| Piece | Where it's from |
|---|---|
| Cross-model hypothesis retrieval (CMHA) | My paper *Beyond HyDE: Cross-Model Hypothesis Diversity for Robust Dense Retrieval* (VLDB Workshop 2026) |
| The generation step that reads evidence and answers | My *Retrieve, Locate, Generate* paper, §3.3 |
| The confidence score | Also from *Beyond HyDE* — it's the same "how much did the models agree" statistic that predicted question difficulty there |
| The agent loop itself (check sufficiency → retrieve again if needed) | **New for this project** — this is the actual contribution, in `run_agent_batch()` inside `src/cmha_agent.py` |

---

## What's in this folder

```
CapstoneProposal-CMHA/
├── README.md                  ← you're reading it
├── requirements.txt
├── .env.example                ← only matters if you don't want the local-only setup
├── run_baseline.py             ← run this
├── src/
│   ├── cmha_agent.py            ← the actual pipeline and the agent loop
│   ├── llm_client.py            ← talks to Ollama
│   ├── embedder.py              ← also talks to Ollama, for embeddings
│   └── hotpot_metrics.py        ← scores answers (exact match / F1)
├── data/
│   └── hotpotqa_sample.json     ← 30 real questions, frozen so you don't need to download anything
├── examples/
│   ├── test_case.md             ← the required test case, written up
│   ├── test_case_real_output.jsonl
│   ├── real_run_n10_{agent,cmha,direct,single_hyde}.jsonl   ← real numbers from real runs
│   └── mock_pipeline_smoketest.jsonl
└── results/                    ← your own runs land here
```

---

## Why five small models instead of one big one

I picked five models that each stay comfortably under 8GB of RAM, spread across different companies (Alibaba, Meta, Google, Microsoft) the same way "Beyond HyDE" used Alibaba/Meta/Google/Mistral — I swapped Mistral out only because their smallest model is 7B, a bit tight for the RAM budget I wanted.

| Role | Model | From | Size on disk |
|---|---|---|---|
| Hypothesis generator | `qwen2.5:3b` | Alibaba | ~1.9GB |
| Hypothesis generator | `llama3.2:3b` | Meta | ~2.0GB |
| Hypothesis generator | `gemma2:2b` | Google | ~1.6GB |
| Hypothesis generator | `phi3.5:3.8b` | Microsoft | ~2.2GB |
| Answering / deciding | `qwen2.5:3b` | (same as above) | — |
| Embeddings | `nomic-embed-text` | Nomic | ~274MB |

Here's the thing people usually get wrong about this: Ollama only ever holds **one** model in memory at a time. So even though the five models add up to about 8GB on your hard drive, the actual RAM used at any moment is just whichever single model is currently loaded — at most about 2.2GB. Plenty of room on an 8GB machine.

The catch is that switching between models isn't free — I measured it, and calling a model that's already loaded takes about 0.3 seconds, but switching to a *different* model costs 3-5 seconds while Ollama swaps it in from disk. If I'd written the code the obvious way (handle one question completely, then move to the next), every single question would pay that switching cost five, six, seven times over. Instead, the code finishes everything it needs from model A across *all* the questions before it ever touches model B. That one change took a 30-question run from about 15 minutes down to 2-3 minutes for the simpler strategies. The tradeoff is that results get written to disk once the whole batch finishes rather than after each question — fine for a run this size, but worth knowing about.

---

## Why the data is a small frozen file instead of "just download HotpotQA"

HotpotQA already hands you, for every question, a neat little pool of 10 paragraphs — 2 that actually contain the answer, and 8 distractors that don't. So you don't need the giant 5-million-paragraph search index the original "Beyond HyDE" paper needed; the question here is simpler: given *these* 10 paragraphs, can the agent pick the right ones?

`data/hotpotqa_sample.json` holds 30 real questions pulled straight from HotpotQA's official validation set (the "bridge" type — the ones that genuinely need two connected facts). I fetched these through HuggingFace and saved them into the repo so that grading this doesn't depend on any download working on the day it's graded. I picked a spread of easy-to-hard questions rather than cherry-picking ones I knew would work.

---

## Setting it up

**1. Install Ollama**

```bash
brew install ollama
# or grab an installer from https://ollama.com/download
```

Check if it's already running before starting it yourself:
```bash
curl -s http://localhost:11434/api/version
```
If that doesn't respond, start it:
```bash
ollama serve &
```

**2. Pull the models** (about 8GB total, one-time)

```bash
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull phi3.5:3.8b
ollama pull nomic-embed-text
```

**3. Set up Python**

```bash
cd CapstoneProposal-CMHA
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That's genuinely it — no API key to hunt down, no `.env` file to fill in. `.env.example` is only there in case you'd rather point this at a hosted model instead of running everything locally.

---

## Actually running it

**First, a quick check that doesn't need Ollama at all.** This just proves the code runs — it uses fake canned text instead of calling any model, so don't read anything into the numbers:

```bash
python run_baseline.py --strategy agent --limit 3 --mock
```

You should see it finish almost instantly, scoring 3/3 with 0 errors. The exact-match score will be 0 and "hops used" will always be 1 — that's expected, since the fake text always looks "sufficient" to the parser. This step is just a sanity check that nothing is broken before you install anything else.

**Now the real thing** — this is what actually matters, and what the assignment is asking to see:

```bash
python run_baseline.py --strategy agent --limit 10
```

`agent` is the default, so you could also just run `python run_baseline.py --limit 10`. On my machine this takes roughly 15 seconds per question — mostly because, unlike the simpler strategies, the agent has to decide per-question whether it needs a second look, so it can't be batched quite as aggressively. Ten questions takes about 2.5 minutes; the full 30 takes 7-8.

**To reproduce the full comparison** the proposal talks about:

```bash
python run_baseline.py --strategy direct      --out results/direct.jsonl
python run_baseline.py --strategy single_hyde --out results/single_hyde.jsonl
python run_baseline.py --strategy cmha        --out results/cmha.jsonl
python run_baseline.py --strategy agent       --out results/agent.jsonl
```

The first three never make a decision — they just retrieve a fixed number of paragraphs and answer. `agent` starts from the exact same first search as `cmha`, but then it's allowed to decide it needs more. Comparing `cmha` against `agent` is really the whole point: it isolates what the decision-making itself is worth, separate from the retrieval method underneath it.

### What it actually looks like when the agent decides to dig deeper

Here's a real question from one of my test runs, not something I made up to look good:

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

The first search didn't turn up enough — the model recognized that, said out loud what it still needed to know, went and searched again specifically for that, and this time found both "Catwoman (film)" and "Catwoman (video game)," which the first search had completely missed. That's a real decision it made mid-run, not something scripted to happen.

In that same batch of runs, one question went the other way: instead of naming something specific it was missing, one of the smaller models just parroted back the instructions from the prompt itself, word for word, as if that were the missing fact. The code handles this gracefully — it notices the response doesn't make sense and just stops rather than looping forever on garbage — but it's a real reminder that small local models don't always follow instructions the way you'd hope. More in the Limitations section below.

### Where everything ends up

Each run writes one line of JSON per question into `results/*.jsonl`. Every record has the question, the correct answer, what the agent guessed, whether it was right, and how confident it was. The `agent` runs additionally log how many hops it took and why it stopped, so you can look at exactly what it decided and when.

If you run the same command twice, it picks up where it left off — it checks what's already in the output file and only computes what's missing.

---

## The actual numbers, from a real run (not made up)

Ran on my machine, first 10 questions, real Ollama calls, no faking anything:

| Strategy | Exact match | F1 | Retrieval recall | Time per question |
|---|---|---|---|---|
| `direct` | 0.400 | 0.450 | 0.500 | 1.8s |
| `single_hyde` | 0.500 | 0.550 | 0.650 | 3.6s |
| `cmha` (always retrieves 4 paragraphs) | 0.400 | 0.586 | 0.650 | 11.7s |
| `agent` (decides for itself) | **0.600** | **0.650** | **0.850** | 15.3s |

That jump from 0.400 to 0.600 is the number I actually care about here. `cmha` and `agent` start from the *identical* first search — the only difference is that `agent` is allowed to notice when that search wasn't enough and go back for more. Six of the ten questions triggered a second look. So this isn't "a better retrieval trick made things better" — it's specifically "letting it decide for itself made things better," which is the actual claim of this whole proposal.

I want to be upfront about something that doesn't fit as neatly, though: `single_hyde` (the simplest possible version, using just one model instead of four) actually did about as well as, or slightly better than, `cmha` on this small sample. That's the opposite of what my earlier "Beyond HyDE" paper found with much bigger models. Ten questions really isn't enough to draw conclusions from — that paper needed hundreds of examples before its numbers held up under a proper statistical test — so I'm reporting this honestly rather than pretending it isn't there. Sorting out whether it's just noise, or whether cross-model averaging genuinely helps less when the models involved are small and error-prone, is exactly what the larger evaluation in Section 6 of the proposal is for.

Two things from these runs that I think hold up regardless of sample size: first, on the required test case (see [`examples/test_case.md`](examples/test_case.md)), all four models guessed a different wrong answer individually, and yet averaging their guesses still pointed retrieval at the right paragraphs — the method survives every individual model being wrong. Second, the Catwoman example above is a genuine, observed instance of the agent catching its own incomplete evidence and fixing it before answering — which is the entire mechanism this project is trying to add.

---

## Every flag, if you want to tweak something

| Flag | Default | What it does |
|---|---|---|
| `--strategy` | `agent` | `direct` / `single_hyde` / `cmha` (no decisions made) or `agent` (the real thing) |
| `--k` | `4` | how many paragraphs to grab on the first search |
| `--k2` | `2` | *(agent only)* how many more to grab if it decides it needs a second look |
| `--max-hops` | `2` | *(agent only)* how many tries it gets before being forced to answer — capped at 2 because HotpotQA's questions are built around exactly two facts |
| `--limit N` | all 30 | just run the first N questions |
| `--hypothesis-models` | the four listed above | which models generate first-search guesses |
| `--answer-model` | `qwen2.5:3b` | which model decides sufficiency and writes the final answer |
| `--embed-model` | `nomic-embed-text` | which model turns text into vectors |
| `--mock` | off | skip Ollama entirely, just check the code runs |
| `--out` | auto-named | where to save results |

---

## What I know is still rough (being upfront, not hiding it)

- **These are much smaller models than my earlier paper used.** "Beyond HyDE" tested models up to 235B and 400B parameters; this uses 2-4B models so it fits on a regular laptop. The numbers here aren't directly comparable to that paper — only the overall shape of the comparison is.
- **The model doesn't always follow the "what's missing" format properly.** I found a real case of this (described above) where a small model just echoed the instructions back instead of naming something real. The code handles it without crashing, but it means the follow-up search in that case was useless.
- **The 2-hop limit is specific to this dataset.** It's well justified for HotpotQA, since these questions are built around exactly two facts — but a real document collection wouldn't come with that guarantee, and a future version would need a smarter way to decide when to stop.
- **The second search only uses one model, not all four.** The first search uses all four models together for better coverage; the follow-up search, to keep things fast, only uses one. Whether using all four again would help more is something I haven't tested yet.
- **The confidence score is borrowed, not freshly calibrated.** It's the same statistic that predicted question difficulty in my earlier paper, but I haven't specifically verified it's well-calibrated on this exact setup — that's planned for the fuller evaluation.
- **Results save once per run, not question-by-question.** Fine for a batch this size; would need reworking for something much bigger where losing an in-progress run would actually hurt.
- **Ten to thirty questions is enough to show this works, not enough to prove how well.** The bigger, properly tested comparison is what Section 6 of the proposal lays out.

---

## Where the ideas came from

- The cross-model retrieval trick, the confidence score, and the discovery that "thinking" models can badly break this kind of retrieval: my paper *Beyond HyDE: Cross-Model Hypothesis Diversity for Robust Dense Retrieval*, VLDB 2026 Workshop on Vector Databases.
- The retrieve-then-generate scoring approach this builds on: my paper *Retrieve, Locate, Generate: An Oracle-Substitution Diagnostic for Literature-Grounded QA*.
- The reminder that a small sample can mislead you and a real significance test is needed before trusting a result: my paper *Transfer or Noise? A Native-Scaffold Control for Meta-Optimized Agent Harnesses*.
- The idea of logging and reading an agent's own decisions (how many hops, why it stopped) as real evidence, not just its final score: my paper *Reasoning Changes How LLM Agents Play Strategic Games, Not How Well: A Behavioral Analysis on GLEE*.
- The dataset: Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018.
