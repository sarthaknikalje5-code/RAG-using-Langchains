import streamlit as st

from rag_pipeline import setup_rag, ask_question


st.title("📚 Research Assistant")
st.write("Ask questions about your research paper.")


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    file_path = "uploaded.pdf"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    retriever, prompt, llm = setup_rag(file_path)

    query = st.chat_input("Ask a question about the paper")

    if query:
        response = ask_question(
            query,
            retriever,
            prompt,
            llm
        )

        st.write(response)