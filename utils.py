import uuid
import fitz
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------- PDF TEXT EXTRACTION ----------
def load_text_from_pdf(pdf_path: str) -> str:
    """Extract and return all text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text + "\n"
    return text


# ---------- CLEAN TEXT ----------
def clean_text(text: str) -> str:
    """Remove extra whitespace and blank lines."""
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


# ---------- CHUNK TEXT ----------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)


# ---------- EMBEDDING MODEL ----------
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and return a SentenceTransformer embedding model."""
    return SentenceTransformer(model_name)


# ---------- CHROMADB EMBEDDING FUNCTION WRAPPER ----------
class SentenceTransformerEmbeddingFunction:
    """
    Thin wrapper so SentenceTransformer integrates cleanly with ChromaDB's
    custom embedding-function interface.
    """

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input).tolist()


# ---------- CREATE CHROMA COLLECTION ----------
def create_chroma_collection(
    chunks: list[str],
    model: SentenceTransformer,
    collection_name: str = "rag_collection",
    persist_directory: str = "./chroma_db",
) -> chromadb.Collection:
    """
    Encode chunks and store them in a persistent ChromaDB collection.

    Drops and recreates the collection each call so the index always reflects
    the latest document set.

    Returns:
        The populated ChromaDB Collection object.
    """
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False),
    )

    # Always start fresh for a new document
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    embedding_fn = SentenceTransformerEmbeddingFunction(model)

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(uuid.uuid4()) for _ in chunks]

    # Add in batches to respect ChromaDB's internal limits
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
        )

    return collection


# ---------- RETRIEVE TOP-K CHUNKS ----------
def retrieve_chunks(
    query: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    k: int = 8,
) -> str:
    """
    Return the top-k most relevant chunks for a given query using ChromaDB.

    Args:
        query:      The user's question.
        model:      The embedding model (used only to keep the signature
                    consistent; ChromaDB handles embedding internally via the
                    stored embedding function).
        collection: The ChromaDB collection to search.
        k:          Number of results to retrieve.

    Returns:
        A single string of the top-k chunks joined by newlines.
    """
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents"],
    )
    return "\n".join(results["documents"][0])