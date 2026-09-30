import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import Pinecone as LangchainPinecone
from langchain_tavily import TavilySearch

load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return str(val).strip()
    try:
        import streamlit as st
        # 1. Direct key
        if key in st.secrets:
            v = str(st.secrets[key]).strip()
            os.environ[key] = v
            return v
        # 2. Check nested sections like [general] or [secrets]
        for section in st.secrets.values():
            if isinstance(section, dict) and key in section:
                v = str(section[key]).strip()
                os.environ[key] = v
                return v
    except Exception:
        pass
    return default


PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
INDEX_NAME = get_secret("INDEX_NAME", "vertex-mera-index")
GROQ_API_KEY = get_secret("GROQ_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=GROQ_API_KEY
)

grading_model = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=GROQ_API_KEY
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = LangchainPinecone(
    index_name=INDEX_NAME,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


web_search_tool = TavilySearch(max_results=2)
