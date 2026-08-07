"""
Command-line tool for testing the RAG chain without the Streamlit UI.

Run:
    python connect_memo_llm.py
Type "exit" or "quit" to stop.
"""
import logging
import sys

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CUSTOM_PROMPT_TEMPLATE,
    DB_FAISS_PATH,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL_NAME,
    LLM_TEMPERATURE,
    RETRIEVER_TOP_K,
    require_groq_api_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_llm():
    api_key = require_groq_api_key()
    return ChatGroq(
        model_name=GROQ_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        groq_api_key=api_key,
    )


def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])


def load_vectorstore():
    if not DB_FAISS_PATH.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {DB_FAISS_PATH}. "
            "Run `python memeory_llm.py` first to build it."
        )
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(
        str(DB_FAISS_PATH), embedding_model, allow_dangerous_deserialization=True
    )


def build_qa_chain():
    db = load_vectorstore()
    return RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    )


def main():
    try:
        qa_chain = build_qa_chain()
    except Exception:
        logger.exception("Failed to initialize the QA chain.")
        sys.exit(1)

    print("Medical bot CLI ready. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_query = input("Write Query Here: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            break

        try:
            response = qa_chain.invoke({"query": user_query})
            print("\nRESULT:", response["result"])
            print("SOURCE DOCUMENTS:", response["source_documents"], "\n")
        except Exception:
            logger.exception("Error while answering the query.")


if __name__ == "__main__":
    main()