from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate 

from dotenv import load_dotenv

load_dotenv()

def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunk = text_splitter.split_documents(documents)
    return chunk

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

def create_vector_store(chunk, embeddings):
    vector_store = Chroma(
        embedding_function = embeddings,
        #persist_directory="chroma-db"
    )
    vector_store.add_documents(chunk)
    """if vector_store._collection.count() == 0:
        vector_store.add_documents(chunk)
    return vector_store"""

def create_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_kwargs = {"k":4}
    )
    return retriever

def format_context(documents):
    context = "\n\n".join(
        f"[source : {doc.metadata.get('source')},"
        f"page : {doc.metadata.get('page_label')}]\n"
        f"{doc.page_content}"
        for doc in documents
    )
    return context


def create_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
            """You are a specialized and highly experienced research assistant with strong expertise
in technical and mathematical documents.

You must answer questions using ONLY the information provided in the context below.
You are not allowed to use any external knowledge, assumptions, or prior training data
beyond what appears in the context.

Before answering, carefully analyze the question and compare its terminology with the
terminology used in the document.

If the exact wording of the question does not appear in the document, but the document
clearly contains logically equivalent or closely related terminology, you must:

1. Explicitly state that the exact term does not appear in the document.
2. Identify the closest matching term used in the document.
3. Explain why it is logically related to the question.
4. Then provide the answer based strictly on the document's content.

For example:
If the question refers to "Boolean graph" but the document uses the term
"zero-divisor graph of a Boolean ring", you must explicitly clarify this difference
before answering.

You are allowed to use logical implications and combine statements from the context
if they clearly support the answer, even if the wording is not identical.

However, if the context does not contain sufficient information to logically infer an
answer, you must respond exactly with:

"SORRY SIR I DON'T KNOW THE ANSWER TO THIS AS IT MAY NOT BE PRESENT IN THE DOCUMENT"

If the question contains terminology that does not appear exactly in the document,
but clearly refers to a concept described using different terminology in the document,
you may align the terms if the meaning is unambiguous.

Your explanations must be precise, academically rigorous, and strictly grounded in
the provided context.

CITATION REQUIREMENTS:

The context contains the source name and page number for each retrieved section.

Whenever you use information from a retrieved section, cite the page number
immediately after the relevant statement or paragraph.

Use the following format:

[Page X]

If a statement is supported by multiple pages, cite all relevant pages using:

[Pages X, Y]

Only use page numbers that are explicitly provided in the context.
Never invent or guess a page number.

Do not provide a separate Sources section unless specifically asked.
Citations should appear directly after the information they support.
Context:
{context}

""" ),
    ("human", "{question}")
        ]
    )

    return prompt

def connect_llm():
    llm = ChatGoogleGenerativeAI(
        model = "gemini-3.5-flash",
        temperature = 0
    )
    return llm

def setup_rag(file_path):
    documents = load_documents(file_path)
    chunk = split_documents(documents)
    embeddings = create_embeddings()
    vector_store = create_vector_store(chunk, embeddings)

    retriver = create_retriever(vector_store)
    prompt = create_prompt()
    llm = connect_llm()

    return retriver, prompt, llm

def ask_question(query, retriver, prompt, llm):
    result = retriver.invoke(query)

    context = format_context(result)
    message = prompt.invoke(
        {
            "context" : context,
            "question" : query
        }
    )

    response = llm.invoke(message)

    return response.content[0]["text"]
