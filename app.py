#!/usr/bin/env python3
"""Interactive demo for the CMHA-Agent baseline.

Pick one of the 30 frozen HotpotQA questions and watch the agent loop run
live: hypothesis generation, hop-1 retrieval, the model's own sufficiency
check, an optional hop-2 retrieval, and the final answer. This is a demo
of the same pipeline in src/cmha_agent.py / run_baseline.py, not a
separate implementation.

Run:
    pip install -r requirements.txt -r requirements-app.txt
    streamlit run app.py

See README.md for Ollama setup. Use the "mock mode" checkbox in the
sidebar to try the UI without Ollama running (canned text, not a real
answer).
"""

import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cmha_agent import run_agent_single  # noqa: E402
from embedder import Embedder  # noqa: E402
from llm_client import LLMClient  # noqa: E402

DEFAULT_HYPOTHESIS_MODELS = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "phi3.5:3.8b"]
DEFAULT_ANSWER_MODEL = "qwen2.5:3b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="CMHA-Agent demo", layout="wide")


@st.cache_data
def load_questions():
    with open(os.path.join(os.path.dirname(__file__), "data", "hotpotqa_sample.json")) as f:
        return json.load(f)


def render_hypothesis(payload):
    model, text = payload["model"], payload["text"]
    if text is None:
        st.write(f"**{model}**: _call failed_")
    else:
        st.write(f"**{model}**: {text}")


def render_retrieval(payload):
    hop = payload["hop"]
    evidence = payload["evidence"]
    gold = set(evidence["gold_titles"])
    if hop > 1:
        st.write(f"Follow-up query (hop {hop}), targeting: *“{payload['missing']}”*")
        st.caption(f"Follow-up hypothesis: {payload.get('followup_hypothesis', '')}")
    st.write(f"**Hop {hop} retrieved paragraphs:**")
    for title in evidence["retrieved_titles"]:
        tag = "✅ gold" if title in gold else "distractor"
        st.write(f"- {title} ({tag})")
    st.caption(f"Retrieval recall so far: {evidence['retrieval_recall']:.2f} (gold paragraphs found / gold paragraphs needed)")


def render_sufficiency(payload):
    hop = payload["hop"]
    prior_hop = hop - 1
    if payload["is_sufficient"]:
        st.success(f"Checking evidence after hop {prior_hop}: judged **sufficient** — stopping here.")
    else:
        missing = payload["missing"] or "(no specific fact named)"
        st.warning(f"Checking evidence after hop {prior_hop}: **insufficient**, deciding to try hop {hop}. Model says it's still missing: *{missing}*")
    st.caption(f"Raw model output (sufficiency check after hop {prior_hop}):")
    st.code(payload["raw"] or "(call failed)")


def render_answer(payload, gold_answer):
    predicted = payload["predicted"]
    if predicted is None:
        st.error("Answer generation failed.")
        return
    correct = predicted.strip().lower() == gold_answer.strip().lower()
    if correct:
        st.success(f"**Predicted answer:** {predicted}  \n**Gold answer:** {gold_answer}  \n(exact match)")
    else:
        st.error(f"**Predicted answer:** {predicted}  \n**Gold answer:** {gold_answer}  \n(not an exact match — see F1 below for partial credit)")


st.title("CMHA-Agent: watch the agent think")
st.caption(
    "Cross-model hypothesis retrieval, plus a model-judged sufficiency check that can trigger a "
    "second, targeted retrieval hop before answering. Same code as run_baseline.py --strategy agent."
)

questions = load_questions()

with st.sidebar:
    st.header("Setup")
    mock = st.checkbox(
        "Mock mode (no Ollama needed)",
        value=False,
        help="Uses canned text instead of real model calls, just to try the UI. Answers are meaningless in this mode.",
    )
    labels = [f"{q['question'][:70]}{'...' if len(q['question']) > 70 else ''}" for q in questions]
    choice = st.selectbox("Pick a question", options=range(len(questions)), format_func=lambda i: labels[i])
    st.divider()
    k1 = st.number_input("k (hop 1 paragraphs)", min_value=1, max_value=10, value=4)
    k2 = st.number_input("k2 (extra paragraphs per follow-up hop)", min_value=1, max_value=10, value=2)
    max_hops = st.number_input("max hops", min_value=1, max_value=4, value=2)
    run_clicked = st.button("Run agent", type="primary")

item = questions[choice]
st.subheader(item["question"])
with st.expander("Gold answer (spoiler — the agent never sees this)"):
    st.write(item["answer"])

if run_clicked:
    embedder = Embedder(model=DEFAULT_EMBED_MODEL, mock=mock)
    llm = LLMClient(mock=mock)

    with st.status("Running agent loop...", expanded=True) as status:
        st.write("**Step 1 — generating hypotheses from 4 diverse local models:**")

        final_record = {}

        def on_event(kind, payload):
            if kind == "hypothesis":
                render_hypothesis(payload)
            elif kind == "retrieval":
                render_retrieval(payload)
            elif kind == "sufficiency":
                render_sufficiency(payload)
            elif kind == "followup_failed":
                st.error(f"Follow-up call failed for hop {payload['hop']}; stopping and answering with current evidence.")
            elif kind == "followup_empty":
                st.info(f"No more distinct paragraphs left to retrieve for hop {payload['hop']}; answering with current evidence.")
            elif kind == "answer":
                render_answer(payload, item["answer"])
            elif kind == "done":
                final_record.update(payload["record"])

        try:
            run_agent_single(
                item,
                embedder=embedder,
                llm=llm,
                hypothesis_models=DEFAULT_HYPOTHESIS_MODELS,
                answer_model=DEFAULT_ANSWER_MODEL,
                k1=k1,
                k2=k2,
                max_hops=max_hops,
                on_event=on_event,
            )
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.exception(e)
            st.info(
                "If this is a connection error, make sure Ollama is running (`ollama serve`) and the "
                "models are pulled — see README.md — or check 'Mock mode' in the sidebar to try the UI "
                "without Ollama."
            )
            final_record = {}

    if final_record and "error" not in final_record:
        st.divider()
        st.subheader("Summary")
        cols = st.columns(4)
        cols[0].metric("Exact match", final_record["em"])
        cols[1].metric("F1", final_record["f1"])
        cols[2].metric("Hops used", final_record["hops_used"])
        cols[3].metric("Confidence", final_record["confidence"])
        st.caption(f"Stop reason: {final_record['stop_reason']}")
        with st.expander("Full result record (JSON)"):
            st.json(final_record)
else:
    st.info("Pick a question in the sidebar and click **Run agent** to watch it work.")
