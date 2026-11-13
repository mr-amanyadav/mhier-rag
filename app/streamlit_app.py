import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from src.generation.answer_generator import answer_question

st.set_page_config(page_title="MHier-RAG", page_icon="📄", layout="wide")
st.title("📄 MHier-RAG — Document QA Demo")

if "doc_name" not in st.session_state:
    st.session_state["doc_name"] = "sample"

st.markdown(f"**Document**: `{st.session_state['doc_name']}` (pre-indexed)")

with st.form("ask_form", clear_on_submit=False):
    question = st.text_input(
        "Ask a question:",
        placeholder="How many full-time employees does NVIDIA have?"
    )
    submitted = st.form_submit_button("🔍 Ask", type="primary")

if submitted:
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner(f"Working... (~90 sec)"):
            try:
                result = answer_question(st.session_state["doc_name"], question, verbose=False)
                st.success("Done!")
                st.markdown("### 💡 Answer")
                st.markdown(result["answer"])
                st.caption(f"Pages: {result['top_pages'][:3]} · Candidates: {result['candidates']}")
            except Exception as e:
                st.error(f"❌ Error: {type(e).__name__}: {e}")

st.divider()
st.markdown("### 💡 Suggested questions")
for q in [
    "How many full-time employees does NVIDIA have?",
    "What was the total amortization expense for intangible assets in fiscal 2023?",
    "What is the total of NVIDIA's future purchase commitments for fiscal year 2024?",
]:
    if st.button(q, key=f"btn_{q}"):
        with st.spinner("Working... (~90 sec)"):
            try:
                result = answer_question(st.session_state["doc_name"], q, verbose=False)
                st.success("Done!")
                st.markdown("### 💡 Answer")
                st.markdown(result["answer"])
                st.caption(f"Pages: {result['top_pages'][:3]}")
            except Exception as e:
                st.error(f"❌ Error: {type(e).__name__}: {e}")