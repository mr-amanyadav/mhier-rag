import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import TOP_K_PAGES, TOP_K_SUMMARIES
from src.indexing.in_page_index import search_chunks
from src.indexing.cross_page_index import search_summaries
from src.utils.llm_client import call_llm


def rerank_pages(question: str, page_chunks: dict) -> list:
    """LLM re-ranks candidate pages by relevance (paper's Section A)."""
    scored = []
    for page_num, chunks in page_chunks.items():
        excerpt = "\n".join(chunks)[:1500]
        prompt = f"""Question: {question}

Content from page {page_num}:
{excerpt}

Rate how relevant this page is to answering the question.
Reply with ONLY a number from 0 to 1 (e.g., 0.7)"""
        try:
            score_str = call_llm(prompt, max_tokens=50)
            # Extract first float from response
            import re
            m = re.search(r"0?\.\d+|[01](?:\.0)?", score_str)
            score = float(m.group()) if m else 0.5
        except Exception:
            score = 0.5
        scored.append((page_num, score))

    scored.sort(key=lambda x: -x[1])
    return [p for p, _ in scored[:TOP_K_PAGES]]


def retrieve_context(doc_name: str, question: str) -> dict:
    """Multi-granularity retrieval: page-level + document-level."""
    # --- Step A: chunk retrieval → candidate pages ---
    chunk_hits = search_chunks(doc_name, question, top_k=20)
    page_chunks = {}
    for chunk, meta in chunk_hits:
        page_chunks.setdefault(meta["page"], []).append(chunk)

    # --- Step B: LLM re-rank pages ---
    print(f"[retriever] {len(page_chunks)} candidate pages → re-ranking", flush=True)
    top_pages = rerank_pages(question, page_chunks)
    print(f"[retriever] top pages: {top_pages}", flush=True)

    # --- Step C: summary retrieval for long-distance reasoning ---
    summary_hits = search_summaries(doc_name, question, top_k=TOP_K_SUMMARIES)
    print(f"[retriever] {len(summary_hits)} summary clusters retrieved", flush=True)

    # --- Step D: assemble context ---
    page_text = "\n\n".join(
        f"--- Page {p} ---\n" + "\n".join(page_chunks[p])[:2000]
        for p in top_pages
    )

    summary_text = "\n\n".join(
        f"[Cluster covering pages {s['pages'][0]}-{s['pages'][-1]}, sim={s['similarity']:.2f}]:\n{s['summary']}"
        for s in summary_hits
    )

    return {
        "top_pages": top_pages,
        "page_text": page_text,
        "summary_text": summary_text,
        "num_candidates": len(page_chunks),
    }