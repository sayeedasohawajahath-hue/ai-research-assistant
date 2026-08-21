"""
One-off script: builds the seed Chroma collection from data/seed_docs/.
Run this once with: python -m src.ingestion.build_vectorstore
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

SEED_DOCS_DIR = "data/seed_docs"
VECTORSTORE_DIR = "vectorstores/seed_collection"
COLLECTION_NAME = "seed_collection"


def load_documents():
    docs = []
    for filename in os.listdir(SEED_DOCS_DIR):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(SEED_DOCS_DIR, filename)
            print(f"Loading {path} ...")
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    return docs


def build():
    docs = load_documents()
    print(f"Loaded {len(docs)} raw pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR,
    )
    print(f"Seed vectorstore built and persisted at {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build()