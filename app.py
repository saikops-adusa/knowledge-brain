import streamlit as st

from knowledge_brain.qa import answer_question, chunks_from_uploaded_files, upload_signature

st.set_page_config(page_title="Knowledge Brain", page_icon="🧠")
st.title("🧠 Knowledge Brain")
st.write("Upload PDF presentations and ask questions. Answers are generated from uploaded document content.")

uploaded_files = st.file_uploader(
    "Upload PDF presentations",
    type=["pdf"],
    accept_multiple_files=True,
)

current_signature = upload_signature(uploaded_files)
previous_signature = st.session_state.get("upload_signature")
if not uploaded_files:
    st.session_state["chunks"] = []
    st.session_state["upload_signature"] = tuple()
elif current_signature != previous_signature:
    st.session_state["chunks"] = chunks_from_uploaded_files(uploaded_files)
    st.session_state["upload_signature"] = current_signature
    st.success(f"Loaded {len(uploaded_files)} PDF(s).")

question = st.text_input("Ask a question about your presentations")
if st.button("Ask"):
    chunks = st.session_state.get("chunks", [])
    if not chunks:
        st.warning("Please upload at least one PDF first.")
    elif not question.strip():
        st.warning("Please type a question.")
    else:
        st.subheader("Answer")
        st.write(answer_question(question, chunks))
