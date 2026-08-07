"""
Build the FAISS vector store from the PDFs in data/.

Run this once (and again whenever the source PDFs change):
    python memeory_llm.py
"""
import logging
import sys

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_PATH, DB_FAISS_PATH, EMBEDDING_MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_pdf_files(data_path):
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data folder not found at {data_path}. "
            "Create it and add at least one PDF before running this script."
        )
    loader = DirectoryLoader(str(data_path), glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise ValueError(
            f"No PDF files were loaded from {data_path}. "
            "Check that the folder contains valid .pdf files."
        )
    return documents


def create_chunks(extracted_data, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(extracted_data)


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def main():
    try:
        logger.info("Loading PDF files from %s", DATA_PATH)
        documents = load_pdf_files(DATA_PATH)
        logger.info("Loaded %d PDF page(s).", len(documents))

        logger.info("Splitting documents into chunks...")
        text_chunks = create_chunks(documents)
        logger.info("Created %d text chunk(s).", len(text_chunks))

        logger.info("Loading embedding model '%s' (first run downloads it)...", EMBEDDING_MODEL_NAME)
        embedding_model = get_embedding_model()

        logger.info("Building FAISS index...")
        db = FAISS.from_documents(text_chunks, embedding_model)

        DB_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
        db.save_local(str(DB_FAISS_PATH))
        logger.info("Vector store saved to %s", DB_FAISS_PATH)

    except Exception:
        logger.exception("Failed to build vector store.")
        sys.exit(1)


if __name__ == "__main__":
    main()