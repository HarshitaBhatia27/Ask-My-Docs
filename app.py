import streamlit as st
import requests

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

if uploaded_file:
    # only index if this is a NEW file we haven't indexed yet
    if uploaded_file.name != st.session_state.last_file:
        with st.spinner("Reading and indexing your document..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            index_response = requests.post("http://fastapi:8000/upload", files=files)

        if index_response.status_code == 200:
            st.session_state.indexed = True
            st.session_state.last_file = uploaded_file.name
            st.session_state.messages = []  # clear old chat
        else:
            st.error("Something went wrong while indexing the document.")

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
                response = requests.post(
                    "http://fastapi:8000/ask",
                    json={"question": question},
                    timeout=120
                )
                answer = response.json()["answer"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})