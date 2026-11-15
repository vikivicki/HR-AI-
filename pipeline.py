import os
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

DATA_DIR = Path(os.getenv("HR_PDF_DIR", "data/hr_policies"))
INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "."))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# This is a starting value, not a universal truth. Calibrate it on in-scope
# and out-of-scope validation questions before submission.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.25"))

SYSTEM_PROMPT = """You are Zyro Dynamics' HR Help Desk assistant.
Answer ONLY questions about the HR policies contained in the supplied context.
Use only the context. Do not invent policy details or fill gaps with assumptions.
If the context does not contain the answer, say that the available HR policy
documents do not provide enough information and suggest contacting HR.
Be concise and helpful. Do not answer unrelated questions.
Do not claim that a policy says something unless the context supports it.

Context:
{context}

Employee question:
{question}

Answer:"""

def load_and_split_pdfs(data_dir: Path = DATA_DIR,
                        chunk_size: int = 800,
                        chunk_overlap: int = 120):
    """Load all PDFs in data_dir and split them into overlapping chunks."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {data_dir}. Add the 11 HR policy PDFs there."
        )
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. Add the HR policy PDFs first."
        )

    loader = PyPDFDirectoryLoader(str(data_dir), glob="*.pdf")
    documents = loader.load()
    if not documents:
        raise ValueError("PDFs were found, but no text could be extracted.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Keep concise metadata for source display and citations.
    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "Unknown document")).name
        chunk.metadata["source_file"] = source
        chunk.metadata["page_number"] = int(chunk.metadata.get("page", 0)) + 1

    return documents, chunks

def create_vector_store(chunks, save_local: bool = True):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    if save_local:
        vector_store.save_local(str(INDEX_DIR))
    return vector_store

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    if not INDEX_DIR.exists():
        raise FileNotFoundError("FAISS index not found. Run ingestion first.")
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

def format_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "Unknown PDF")
        page = doc.metadata.get("page_number", "?")
        parts.append(f"[Source {i}: {source}, page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)

def answer_question(question: str, vector_store=None,
                    threshold: float = RELEVANCE_THRESHOLD):
    """Apply a relevance gate, retrieve with MMR, then answer using LCEL."""
    if not question or not question.strip():
        return {
            "answer": "Please enter an HR policy question.",
            "sources": [],
            "in_scope": False,
            "relevance_score": None
        }

    if vector_store is None:
        vector_store = load_vector_store()

    # Similarity score is used as a first-pass scope/relevance gate.
    scored_docs = vector_store.similarity_search_with_relevance_scores(
        question, k=3
    )
    best_score = float(scored_docs[0][1]) if scored_docs else 0.0

    if not scored_docs or best_score < threshold:
        return {
            "answer": (
                "I can help with Zyro Dynamics HR policies, but this question "
                "does not appear to be covered by the available HR documents. "
                "Please ask an HR-related question or contact the HR team."
            ),
            "sources": [],
            "in_scope": False,
            "relevance_score": best_score
        }

    # MMR selects relevant but less redundant chunks.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.5}
    )
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    llm = ChatGroq(model=LLM_MODEL, temperature=0, max_retries=2)
    chain = (
        {
            "context": RunnableLambda(lambda x: format_context(x["docs"])),
            "question": RunnableLambda(lambda x: x["question"])
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    docs = retriever.invoke(question)
    answer = chain.invoke({"question": question, "docs": docs})

    # Sources are returned separately from the LLM answer so the UI can
    # display verifiable document/page references.
    unique_sources = []
    seen = set()
    for doc in docs:
        item = (
            doc.metadata.get("source_file", "Unknown PDF"),
            doc.metadata.get("page_number", "?")
        )
        if item not in seen:
            seen.add(item)
            unique_sources.append({
                "file": item[0],
                "page": item[1],
                "excerpt": doc.page_content[:350]
            })

    return {
        "answer": answer,
        "sources": unique_sources,
        "in_scope": True,
        "relevance_score": best_score
    }
