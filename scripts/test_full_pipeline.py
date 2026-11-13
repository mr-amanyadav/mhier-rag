import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generation.answer_generator import answer_question

QUESTIONS = [
    "How many full-time employees does NVIDIA have?",                          # should work (page 11-61)
    "What was the total amortization expense for intangible assets in fiscal 2023?",  # hard case
    "What is the total of NVIDIA's future purchase commitments for fiscal year 2024?", # hard case
]

for q in QUESTIONS:
    result = answer_question("sample", q)
    print("\n" + result["answer"])
    print(f"\n[Retrieved from {result['candidates']} candidate pages, top 5: {result['top_pages']}]")
    print("\n" + "=" * 70)