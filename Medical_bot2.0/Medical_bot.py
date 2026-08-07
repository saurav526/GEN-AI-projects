import logging

import streamlit as st
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


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_vectorstore():
    if not DB_FAISS_PATH.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {DB_FAISS_PATH}. "
            "Run `python memeory_llm.py` first to build the vector store."
        )
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(
        str(DB_FAISS_PATH), embedding_model, allow_dangerous_deserialization=True
    )


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )


@st.cache_resource(show_spinner=False)

def get_llm():
    require_groq_api_key()  # Fail fast if the key is missing
    return ChatGroq(
        model_name=GROQ_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        api_key=require_groq_api_key(),
    )


@st.cache_resource(show_spinner=False)
def get_qa_chain():
    vectorstore = get_vectorstore()
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    )


def format_sources(source_documents) -> str:
    if not source_documents:
        return ""
    lines = ["\n\n**Sources:**"]
    for doc in source_documents:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")
        lines.append(f"- {source} (page {page})")
    return "\n".join(lines)


def main():
    st.set_page_config(page_title="MediBot", page_icon="🩺")
    st.title("Ask Chatbot! 🩺")
    st.caption(
        "MediBot answers questions using only the information in its loaded "
        "medical reference documents. It is not a substitute for professional "
        "medical advice — always consult a qualified healthcare provider."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Ask a medical question...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    qa_chain = get_qa_chain()
                    response = qa_chain.invoke({"query": prompt})
                    result = response["result"]
                    result_to_show = result + format_sources(response.get("source_documents"))
                    st.markdown(result_to_show)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result_to_show}
                    )
                except FileNotFoundError as e:
                    logger.error("Vector store missing: %s", e)
                    st.error(str(e))
                except RuntimeError as e:
                    # Raised by require_groq_api_key() for a missing/invalid key
                    logger.error("Configuration error: %s", e)
                    st.error(str(e))
                except Exception as e:
                    logger.exception("Unhandled error while answering query.")
                    st.error(f"Something went wrong: {e}")


if __name__ == "__main__":
    main()