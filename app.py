import os
import streamlit as st
from src.indexer import build_vector_store_from_file
from src.retriever import ask, reload_vector_store
import tempfile

st.set_page_config(page_title="AskMyDocs", page_icon="📚")
st.title("📚 AskMyDocs")
st.caption("Upload any PDF and ask questions about it")

if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_file" not in st.session_state:
    st.session_state.last_file = None

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and uploaded_file.name != st.session_state.last_file:
    with st.spinner("Reading and indexing your document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        build_vector_store_from_file(tmp_path)
        reload_vector_store()  #loads the new index into retriever
        st.session_state.indexed = True
        st.session_state.last_file = uploaded_file.name
        st.session_state.messages = []

if st.session_state.indexed:
    st.success(f"✅ {st.session_state.last_file} is ready!")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask(question)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})