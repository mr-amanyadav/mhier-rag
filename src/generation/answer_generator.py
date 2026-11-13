import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.retrieval.page_retriever import retrieve_context
from src.utils.llm_client import call_llm


COT_PROMPT = """You are a retrieval-augmented question answering system.
Answer the question using ONLY the evidence provided below.

=== EVIDENCE: Page Content ===
{page_text}

=== EVIDENCE: Cross-Page Summaries ===
{summary_text}

=== QUESTION ===
{question}

Respond in this EXACT format:

STEP-BY-STEP ANALYSIS:
1. [First observation from evidence]
2. [Second observation]
3. [Synthesis / comparison]

REASONING SUMMARY:
[One concise sentence]

RELEVANT PAGES:
[List only pages that directly support the answer, e.g., 79, 80]

FINAL ANSWER:
[The direct answer — a number, name, or short phrase. If not answerable from evidence, write "Not answerable"]"""


def answer_question(doc_name: str, question: str, verbose: bool = True) -> dict:
    """Full RAG pipeline: retrieve → rerank → generate CoT answer."""
    if verbose:
        print(f"\n{'=' * 70}\nQUESTION: {question}\n{'=' * 70}", flush=True)

    ctx = retrieve_context(doc_name, question)

    prompt = COT_PROMPT.format(
        page_text=ctx["page_text"],
        summary_text=ctx["summary_text"],
        question=question,
    )

    if verbose:
        print(f"[generator] Sending CoT prompt ({len(prompt)} chars) to LLM", flush=True)

    answer = call_llm(prompt, max_tokens=1500)

    return {
        "question": question,
        "answer": answer,
        "top_pages": ctx["top_pages"],
        "candidates": ctx["num_candidates"],
    }