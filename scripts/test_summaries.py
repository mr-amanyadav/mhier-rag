import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.indexing.cross_page_index import search_summaries

for q in [
    "What is the total of NVIDIA's future purchase commitments for fiscal year 2024?",
    "What was the total amortization expense for intangible assets in fiscal year 2023?",
    "How many full-time employees does NVIDIA have?",
]:
    print("=" * 70)
    print(f"QUERY: {q}\n")
    hits = search_summaries("sample", q, top_k=2)
    for h in hits:
        pages = h['pages']
        pages_str = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0])
        print(f"  [sim={h['similarity']:.3f}] pages: {pages_str}")
        print(f"  {h['summary'][:400]}\n")