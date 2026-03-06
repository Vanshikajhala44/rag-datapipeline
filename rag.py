import os
import uuid
from dotenv import load_dotenv
from groq import Groq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pdfplumber
import chromadb
from chromadb.config import Settings

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ"))

import streamlit as st

api_key = os.getenv("GROQ") or st.secrets.get("GROQ")
groq_client = Groq(api_key=api_key)



# ---------- LOAD PDF PAGE BY PAGE ----------
def load_pdf_by_page(file_path: str) -> dict:
    pages = {}
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            for table in page.extract_tables():
                for row in table:
                    text += " | ".join(cell.strip() if cell else "" for cell in row) + "\n"
            pages[i + 1] = text
    return pages


# ---------- CLEAN TEXT ----------
def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


# ---------- CHUNK TEXT WITH METADATA ----------
def chunk_text_with_metadata(text_by_page: dict) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    documents = []
    for page_num, text in text_by_page.items():
        text = clean_text(text)
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={"page": page_num}
            ))
    return documents


# ---------- EMBEDDING MODEL ----------
def load_embedding_model(model_name="BAAI/bge-small-en-v1.5"):
    return SentenceTransformer(model_name)


# ---------- CHROMADB EMBEDDING FUNCTION WRAPPER ----------
class SentenceTransformerEmbeddingFunction:
    """Wraps SentenceTransformer to be compatible with ChromaDB's embedding interface."""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input).tolist()

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input).tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input).tolist()


# ---------- CREATE CHROMA COLLECTION ----------
def create_chroma_collection(
    documents: list,
    model: SentenceTransformer,
    collection_name: str = "rag_collection",
    persist_directory: str = "./chroma_db",
):
    chroma_client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    embedding_fn = SentenceTransformerEmbeddingFunction(model)

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(uuid.uuid4()) for _ in documents]
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    batch_size = 500
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    return collection


# ---------- RAG PIPELINE ----------
class RAGPipeline:
    def __init__(self, pdf_path: str, persist_directory: str = "./chroma_db"):
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.model = load_embedding_model()
        self.collection = None
        self.documents = None

    def setup(self):
        pages = load_pdf_by_page(self.pdf_path)
        self.documents = chunk_text_with_metadata(pages)
        self.collection = create_chroma_collection(
            self.documents,
            self.model,
            persist_directory=self.persist_directory,
        )
        print(f"Indexed {len(self.documents)} chunks from {len(pages)} pages.")

    def ask(self, query: str) -> str:
        if not query.strip():
            return "No question provided."

        results = self.collection.query(
            query_texts=[query],
            n_results=8,
            include=["documents", "metadatas"],
        )

        context = ""
        citations = []
        for i, (doc_text, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            context += f"[{i+1}] {doc_text}\n\n"
            citations.append((i+1, meta["page"], doc_text[:100]))

        prompt = f"""You are a document assistant. Answer using ONLY the context below.
When you use information from a chunk, cite it using its number like [1], [2], etc.
If the answer is not found, say: "I could not find this in the document."

Context:
{context}

Question:
{query}

Answer (with citations like [1], [2]):"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.choices[0].message.content

        # Agar answer not found hai toh citations mat dikhao
        if "could not find" in answer.lower():
            return answer

        # Warna normal citations dikhao
        citation_block = "\n\n Citations:\n"
        for num, page, preview in citations:
            citation_block += f"  [{num}] Page {page} — \"{preview.strip()}...\"\n"

        return answer + citation_block