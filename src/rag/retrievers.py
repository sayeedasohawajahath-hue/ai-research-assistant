"""
Retrievers for the seed collection and (optionally) an uploaded PDF.
"""
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

VECTORSTORE_DIR = "vectorstores/seed_collection"
SEED_COLLECTION_NAME = "seed_collection"

_embeddings = None
_seed_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def get_seed_retriever(k: int = 4):
    """Returns a retriever over the pre-built seed collection."""
    global _seed_vectorstore
    if _seed_vectorstore is None:
        _seed_vectorstore = Chroma(
            collection_name=SEED_COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=VECTORSTORE_DIR,
        )
    return _seed_vectorstore.as_retriever(search_kwargs={"k": k})


def build_upload_retriever(uploaded_file_path: str, k: int = 4):
    """
    Builds a temporary, in-session Chroma collection from an uploaded PDF.
    Returns a retriever over just that PDF's content.
    """
    loader = PyPDFLoader(uploaded_file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # in-memory collection, unique name per upload so it doesn't collide
    collection_name = f"upload_{os.path.basename(uploaded_file_path)}".replace(".", "_")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        # no persist_directory -> stays in-memory for this session
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})