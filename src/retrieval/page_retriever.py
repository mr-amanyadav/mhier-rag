import sys
import re
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TOP_K_PAGES
from src.indexing.in_page_index import search_chunks
from src.indexing.cross_page_index import search_summaries
from src.utils.llm_client import call_llm, get_client


# Token-budget constants (keeps us under Groq's 8000 TPM free-tier limit)
RERANK_EXCERPT_CHARS = 400    # per page, for re-rank input
CONTEXT_EXCERPT_CHARS = 800   # per page, for final answer context
SUMMARY_CHARS = 300           # per summary
SUMMARY_TOP_K = 3             # fewer summaries = fewer tokens


def rerank_pages(question: str, page_chunks: dict) -> list:
    """
    Batched LLM re-ranking using a smaller, non-reasoning model.

    Why gpt-oss-20b (not 120b)?
      The 120B reasoning model returns prose instead of the strict
      "PAGENUM: SCORE" format we need for parsing. The 20B model follows
      format instructions reliably, is ~3x faster, and has a higher TPM
      ceiling on Groq's free tier. Ranking is a structured task — reasoning
      overhead is wasted here.

    Paper's original: 20 individual per-page LLM calls (~5 min).
    Our optimization: 1 batched call on gpt-oss-20b (~15-25 sec).
    """
    if not page_chunks:
        return []

    # Compact "PAGENUM: excerpt" format the small model parses reliably
    entries = []
    page_numbers = list(page_chunks.keys())
    for page_num in page_numbers:
        excerpt = "\n".join(page_chunks[page_num])[:RERANK_EXCERPT_CHARS].replace("\n", " ")
        entries.append(f"{page_num}: {excerpt}")

    pages_str = "\n".join(entries)

    prompt = f"""You are a ranking assistant. Score how relevant each page is to answering the question.

QUESTION: {question}

PAGES (format "PAGENUM: excerpt"):
{pages_str}

Task: For every page number above, output a score from 0.0 to 1.0.
Rules:
- One line per page
- Format exactly: PAGENUM: SCORE
- No prose, no explanation, no markdown, no thinking
- Every page number above must appear exactly once

Example output:
12: 0.9
36: 0.3
79: 0.85

Scores:"""

    try:
        client = get_client()
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.0,
        )
        response = (resp.choices[0].message.content or "").strip()

        # Accept "12: 0.9", "PAGE 12: 0.9", "[12]: 0.9"
        scores = {}
        for m in re.finditer(r"(?:PAGE\s+|\[)?(\d+)(?:\]|)\s*[:=]\s*([0-9]*\.?[0-9]+)", response):
            page_num = int(m.group(1))
            if page_num in page_chunks and page_num not in scores:
                val = float(m.group(2))
                if 0.0 <= val <= 1.0:
                    scores[page_num] = val

        print(f"[retriever] parsed {len(scores)}/{len(page_chunks)} scores", flush=True)

        if len(scores) < len(page_chunks) * 0.7:
            print(f"[retriever] ⚠ only {len(scores)} scores, using original order", flush=True)
            return list(page_chunks.keys())[:TOP_K_PAGES]

        sorted_pages = sorted(scores.items(), key=lambda x: -x[1])
        return [p for p, _ in sorted_pages[:TOP_K_PAGES]]

    except Exception as e:
        print(f"[retriever] ⚠ re-rank failed ({e}), using original order", flush=True)
        return list(page_chunks.keys())[:TOP_K_PAGES]


def retrieve_context(doc_name: str, question: str) -> dict:
    """Multi-granularity retrieval: page-level + document-level."""
    # --- Step A: chunk retrieval → candidate pages ---
    chunk_hits = search_chunks(doc_name, question, top_k=20)
    page_chunks = {}
    for chunk, meta in chunk_hits:
        page_chunks.setdefault(meta["page"], []).append(chunk)

    # --- Step B: batched LLM re-rank ---
    print(f"[retriever] {len(page_chunks)} candidate pages → batched re-rank", flush=True)
    top_pages = rerank_pages(question, page_chunks)
    print(f"[retriever] top pages: {top_pages}", flush=True)

    # --- Step C: summary retrieval (for long-distance reasoning) ---
    summary_hits = search_summaries(doc_name, question, top_k=SUMMARY_TOP_K)
    print(f"[retriever] {len(summary_hits)} summary clusters retrieved", flush=True)

    # --- Step D: assemble context (trimmed to fit 8K TPM) ---
    page_text = "\n\n".join(
        f"--- Page {p} ---\n" + "\n".join(page_chunks[p])[:CONTEXT_EXCERPT_CHARS]
        for p in top_pages
    )

    summary_text = "\n\n".join(
        f"[pages {s['pages'][0]}-{s['pages'][-1]}]: {s['summary'][:SUMMARY_CHARS]}"
        for s in summary_hits
    )

    return {
        "top_pages": top_pages,
        "page_text": page_text,
        "summary_text": summary_text,
        "num_candidates": len(page_chunks),
    }