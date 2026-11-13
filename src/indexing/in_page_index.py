import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (EMBED_MODEL, CHROMA_DIR, IN_PAGE_CHUNK_SIZE,
                    IN_PAGE_OVERLAP, TOP_K_CHUNKS, DATA_PROCESSED)

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {EMBED_MODEL}", flush=True)
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def chunk_text(text, size, overlap):
    if not text.strip():
        return []
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
        if i + size >= len(words):
            break
        i += size - overlap
    return chunks


def build_in_page_index(doc_name):
    print("STEP 1: Loading JSON", flush=True)
    json_path = Path(DATA_PROCESSED) / f"{doc_name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Not found: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data['pages'])} pages", flush=True)

    print("STEP 2: Chunking", flush=True)
    ids, docs, metadatas = [], [], []
    for page in data["pages"]:
        for chunk in chunk_text(page["text"], IN_PAGE_CHUNK_SIZE, IN_PAGE_OVERLAP):
            ids.append(f"c{len(ids)}")
            docs.append(chunk)
            metadatas.append({"page": page["page"]})
    print(f"  Created {len(docs)} chunks", flush=True)

    print("STEP 3: Loading model", flush=True)
    model = get_model()

    print("STEP 4: Embedding (slow)", flush=True)
    embeddings = model.encode(docs, batch_size=32,
                              show_progress_bar=True,
                              convert_to_numpy=True).tolist()
    print(f"  Embedded {len(embeddings)} chunks", flush=True)

    print("STEP 5: Writing to ChromaDB", flush=True)
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    name = f"{doc_name}_in_page"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(docs), 500):
        col.add(ids=ids[i:i+500], documents=docs[i:i+500],
                metadatas=metadatas[i:i+500], embeddings=embeddings[i:i+500])
    print(f"DONE: {len(docs)} chunks saved", flush=True)


def search_chunks(doc_name, query, top_k=TOP_K_CHUNKS):
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(f"{doc_name}_in_page")
    model = get_model()
    q_emb = model.encode(query, convert_to_numpy=True).tolist()
    r = col.query(query_embeddings=[q_emb], n_results=top_k)
    return list(zip(r["documents"][0], r["metadatas"][0]))


if __name__ == "__main__":
    print(">>> Script starting <<<", flush=True)
    build_in_page_index("sample")
    print(">>> Script complete <<<", flush=True)