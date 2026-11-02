# RAG Research Assistant using LangChain

A Retrieval-Augmented Generation (RAG) research assistant that helps users explore research papers and ask questions about an uploaded PDF using context-grounded answers.

## Features

- Upload a PDF document for question answering.
- Extract and chunk PDF text for retrieval.
- Generate embeddings using Sentence Transformers.
- Retrieve relevant document chunks using a vector store.
- Generate answers using Google's Gemini API.
- Answer questions based on the uploaded document's retrieved context.

## Tech Stack

- Python
- LangChain
- LangGraph (if used in the current implementation)
- Streamlit
- Sentence Transformers
- ChromaDB
- PyMuPDF / PDF parsing tools
-  Python package
- Google Gemini API

## Project Workflow

1. Search for research papers on arXiv.
2. Upload a PDF document.
3. Extract and split the document into chunks.
4. Generate embeddings and index the chunks.
5. Retrieve relevant context for a user's question.
6. Generate an answer using the retrieved context and the language model.

## Project Structure

```text
RAG-using-Langchains/
├── app.py
├── rag_pipeline.py
├── requirements.txt
└── README.md
```

## Setup and Installation

Clone the repository:

```bash
git clone https://github.com/sarthaknikalje5-code/RAG-using-Langchains.git
cd RAG-using-Langchains
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the required API credentials using environment variables. Never commit API keys.

Run the application:

```bash
streamlit run app.py
```

## Learning Outcomes

- Understanding Retrieval-Augmented Generation
- Document preprocessing and chunking
- Text embeddings and semantic retrieval
- Vector databases
- LLM integration using LangChain
- Building AI applications with Python

## Future Improvements

- Advanced RAG techniques
- Improved retrieval and reranking
- Evaluation of retrieval quality and answer faithfulness
- GraphRAG exploration

---

**Note:** This project is a learning project. Verify the listed features, dependencies, and setup instructions against the current implementation.
