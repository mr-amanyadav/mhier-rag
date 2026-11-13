import json
import pickle
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (DATA_PROCESSED, CROSS_PAGE_CHUNK_SIZE, TOP_K_SUMMARIES)
from src.indexing.in_page_index import get_model, chunk_text
from src.utils.llm_client import call_llm

N_CLUSTERS = 15           # paper uses similar; safe for 444 chunks
MAX_CHUNKS_PER_CLUSTER = 40  # cap prompt size to save tokens
SUMMARY_MAX_TOKENS = 200   # keep summaries short & focused

SUMMARY_PROMPT = """You are analyzing excerpts from a long financial document (NVIDIA 10-K).

Below are {n} text excerpts from pages {pages}:

{excerpts}

Write a 3-4 sentence summary covering:
1. Main topic of these excerpts
2. Key facts, numbers, or entities mentioned
3. What kinds of questions this section could answer

Be specific. Include any dollar amounts, percentages, or named entities you see.

SUMMARY:"""


def build_cross_page_tree(doc_name: str):
    print("=== STEP 1: Loading chunks ===", flush=True)
    json_path = Path(DATA_PROCESSED) / f"{doc_name}.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []
    for page in data["pages"]:
        for c in chunk_text(page["text"], CROSS_PAGE_CHUNK_SIZE, overlap=0):
            all_chunks.append({"text": c, "page": page["page"]})
    print(f"  {len(all_chunks)} chunks across {len(data['pages'])} pages", flush=True)

    if len(all_chunks) < N_CLUSTERS:
        raise ValueError(f"Too few chunks ({len(all_chunks)}) for {N_CLUSTERS} clusters")

    print("=== STEP 2: Embedding for clustering ===", flush=True)
    model = get_model()
    embs = model.encode(
        [c["text"] for c in all_chunks],
        batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )

    print("=== STEP 3: GMM clustering ===", flush=True)
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42, n_init=3)
    labels = gmm.fit_predict(embs)
    print(f"  Assigned chunks to {N_CLUSTERS} clusters", flush=True)

    print("=== STEP 4: LLM summarization ===", flush=True)
    summaries = []
    for k in range(N_CLUSTERS):
        cluster_chunks = [all_chunks[i] for i in range(len(all_chunks)) if labels[i] == k]
        if not cluster_chunks:
            continue

        # Cap chunk count to keep prompt small
        sampled = cluster_chunks[:MAX_CHUNKS_PER_CLUSTER]
        pages = sorted(set(c["page"] for c in cluster_chunks))

        excerpts = "\n\n".join(
            f"[p.{c['page']}] {c['text'][:400]}" for c in sampled
        )

        prompt = SUMMARY_PROMPT.format(
            n=len(sampled),
            pages=f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0]),
            excerpts=excerpts[:8000]  # hard cap on prompt size
        )

        try:
            summary = call_llm(prompt, max_tokens=SUMMARY_MAX_TOKENS)
            print(f"  Cluster {k+1}/{N_CLUSTERS}: {len(cluster_chunks)} chunks, "
                  f"{len(pages)} pages → summary OK", flush=True)
        except Exception as e:
            print(f"  Cluster {k+1}/{N_CLUSTERS}: LLM failed ({e}), using fallback", flush=True)
            summary = f"Section covering pages {pages} about: {sampled[0]['text'][:200]}"

        summaries.append({
            "cluster_id": k,
            "summary": summary,
            "pages": pages,
            "chunk_count": len(cluster_chunks),
        })

    print("=== STEP 5: Embedding summaries ===", flush=True)
    summary_embs = model.encode(
        [s["summary"] for s in summaries], convert_to_numpy=True
    )

    out_path = Path(DATA_PROCESSED) / f"{doc_name}_summaries.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "summaries": summaries,
            "embeddings": summary_embs,
        }, f)

    print(f"✅ DONE: {len(summaries)} summaries → {out_path}", flush=True)
    return summaries


def search_summaries(doc_name: str, query: str, top_k: int = TOP_K_SUMMARIES):
    model = get_model()
    q_emb = model.encode(query, convert_to_numpy=True)

    path = Path(DATA_PROCESSED) / f"{doc_name}_summaries.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Build summaries first: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    sims = np.dot(data["embeddings"], q_emb) / (
        np.linalg.norm(data["embeddings"], axis=1) * np.linalg.norm(q_emb) + 1e-9
    )
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [
        {**data["summaries"][i], "similarity": float(sims[i])}
        for i in top_idx
    ]


if __name__ == "__main__":
    print(">>> Cross-page index build starting <<<", flush=True)
    build_cross_page_tree("sample")
    print(">>> Cross-page index complete <<<", flush=True)