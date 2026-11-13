import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.indexing.in_page_index import search_chunks

QUERIES = [
    "What was the total amortization expense for intangible assets in fiscal year 2023?",
    "What is the total estimated fair value of cash equivalents and marketable securities as of January 29, 2023?",
    "What is the total of NVIDIA's future purchase commitments for fiscal year 2024?",
]

for q in QUERIES:
    print("=" * 70)
    print(f"QUERY: {q}\n")
    try:
        hits = search_chunks("sample", q, top_k=3)
        for chunk, meta in hits:
            page = meta["page"]
            preview = chunk[:200].replace("\n", " ")
            print(f"  [Page {page}] {preview}...")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
    print()

print("=" * 70)
print("✅ Retrieval test complete")