# src/ingestion/pdf_parser.py
import pymupdf as fitz  # PyMuPDF
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import DATA_PROCESSED


def parse_pdf(pdf_path: str, output_name: str):
    """Parse a PDF into per-page text using PyMuPDF (fast + stable)."""
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        page_num = i + 1
        if page_num % 10 == 0:
            print(f"  ...parsed {page_num}/{len(doc)} pages", flush=True)

        # 1. Plain text
        text_parts = [page.get_text("text")]

        # 2. Try tables (best-effort — PyMuPDF's table API is solid)
        try:
            tabs = page.find_tables()
            for t_idx, table in enumerate(tabs.tables):
                rows = table.extract()
                table_str = "\n".join(
                    " | ".join((cell or "").strip() for cell in row)
                    for row in rows
                )
                if table_str.strip():
                    text_parts.append(f"\n[TABLE {t_idx + 1}]\n{table_str}")
        except Exception as e:
            # Some PDFs don't allow table extraction — that's fine
            pass

        pages.append({
            "page": page_num,
            "text": "\n".join(text_parts).strip(),
            "tables": [],
            "images": [],
        })

    out_path = Path(DATA_PROCESSED) / f"{output_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"pages": pages}, f, indent=2, ensure_ascii=False)

    total_chars = sum(len(p["text"]) for p in pages)
    print(f"✅ Parsed {len(pages)} pages, {total_chars:,} chars → {out_path}")
    return pages


if __name__ == "__main__":
    parse_pdf("data/raw/sample.pdf", "sample")