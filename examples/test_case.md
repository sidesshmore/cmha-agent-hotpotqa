# Test Case (proposal Section 4)

This is the one concrete test case referenced in the proposal, run through
the real pipeline (Ollama, no mock) on 2026-09-05.

## Sample input

- **Question:** *"Which American film director hosted the 18th Independent Spirit Awards in 2002?"*
- **Question ID:** `5ac3165c5542995ef918c10a` (HotpotQA validation, distractor config, bridge type)
- **Candidate paragraphs given to the retriever:** 10 (2 gold + 8 distractors, exactly as HotpotQA ships them — see `data/hotpotqa_sample.json`)
- **Gold supporting paragraphs:** `"18th Independent Spirit Awards"`, `"John Waters"`
- **Gold answer:** `"John Waters"`

## Expected behavior

1. Four small local models (qwen2.5:3b, llama3.2:3b, gemma2:2b, phi3.5:3.8b)
   each generate a short hypothetical passage answering the question.
2. Each hypothesis is embedded (nomic-embed-text); the centroid should land
   close to the two paragraphs that actually discuss the ceremony and its
   host, among the 10 candidates.
3. The top-4 retrieved paragraphs should include both gold paragraphs.
4. Given that evidence, the answer model should output `"John Waters"`.

## Actual output — real run, no mock, `--strategy cmha`

```json
{
  "id": "5ac3165c5542995ef918c10a",
  "question": "Which American film director hosted the 18th Independent Spirit Awards in 2002?",
  "gold_answer": "John Waters",
  "predicted_answer": "John Waters",
  "em": 1,
  "f1": 1.0,
  "retrieval_recall": 1.0,
  "retrieved_titles": [
    "18th Independent Spirit Awards",
    "John Waters",
    "Mihai Mălaimare Jr.",
    "Independent Spirit Awards"
  ],
  "gold_titles": ["18th Independent Spirit Awards", "John Waters"],
  "diversity_score": 0.1928,
  "confidence": 0.8383,
  "hypotheses": [
    "The 18th Independent Spirit Awards in 2002 were hosted by Kevin Smith.",
    "The American film director who hosted the 18th Independent Spirit Awards in 2002 was Jon Favreau.",
    "Quentin Tarantino hosted the 18th Independent Spirit Awards in 2002. He brought his signature style and humor to the ceremony, marking a significant moment for independent cinema.",
    "Spike Jonze was the American film director who hosted the 1... independent spirit awards ceremony in 2002."
  ],
  "strategy": "cmha"
}
```

Full record: [`test_case_real_output.jsonl`](test_case_real_output.jsonl).

## What worked and what didn't (proposal §4 requirement)

**Worked:** the final answer is correct and retrieval recall is a perfect
1.0 — both gold paragraphs were pulled into the top-4. This is a genuinely
interesting result: **every one of the four hypothesis models individually
guessed a wrong host** (Kevin Smith, Jon Favreau, Quentin Tarantino, Spike
Jonze — the real answer, John Waters, appears in *none* of them). Despite
that, the centroid of those four wrong-but-topically-on-target hypotheses
still pointed retrieval at the right two paragraphs, and the answer model
correctly read "John Waters" off the retrieved evidence rather than trusting
any of the hallucinated names from the hypothesis stage. This is direct,
concrete evidence for the CMHA thesis (cross-model *retrieval* can succeed
even when every individual model's *generation* fails) — not just a
citation of the original paper's claim.

**Didn't work as cleanly:** on the small 10-question slice tested so far,
this doesn't hold up as a clean win for the *full* method over its cheaper
ablations — see [`../README.md`'s comparison table](../README.md#real-results-honest-n10-comparison)
for the honest three-way numbers and why n=10 isn't enough to conclude
anything yet.

## Reproducing this exact test case

```bash
python3 -c "
import json
items = json.load(open('data/hotpotqa_sample.json'))
target = [i for i in items if i['id'] == '5ac3165c5542995ef918c10a']
json.dump(target, open('/tmp/single_case.json', 'w'))
"
python run_baseline.py --data /tmp/single_case.json --strategy cmha --out examples/test_case_rerun.jsonl
```

A `--mock` version of this same question (proves the code path runs with
zero Ollama/network dependency, produces no meaningful score) is in
[`mock_pipeline_smoketest.jsonl`](mock_pipeline_smoketest.jsonl).
