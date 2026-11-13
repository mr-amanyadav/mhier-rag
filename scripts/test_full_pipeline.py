import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generation.answer_generator import answer_question

QUESTIONS = [
    "How many full-time employees does NVIDIA have?",
    "What was the total amortization expense for intangible assets in fiscal 2023?",
    "What is the total of NVIDIA's future purchase commitments for fiscal year 2024?",
]

print(f"\nRunning {len(QUESTIONS)} questions...\n")

for i, q in enumerate(QUESTIONS, 1):
    print(f"\n{'=' * 70}\n[QUESTION {i}/{len(QUESTIONS)}] {q}\n{'=' * 70}")
    result = answer_question("sample", q)
    print("\n" + result["answer"])
    print(f"\n[Candidates: {result['candidates']}, top pages: {result['top_pages']}]")

    if i < len(QUESTIONS):
        print("(sleeping 15s for rate limit)")
        time.sleep(15)

print("\n✅ All 3 questions complete.")