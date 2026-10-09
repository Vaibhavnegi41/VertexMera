import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import Pinecone as LangchainPinecone
from langchain_tavily import TavilySearch

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
load_dotenv()



def get_secret(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return str(val).strip()
    try:
        import streamlit as st
        if key in st.secrets:
            v = str(st.secrets[key]).strip()
            os.environ[key] = v
            return v
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
LANGCHAIN_API_KEY = get_secret("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = get_secret("LANGCHAIN_PROJECT", "VertexMera-CRAG")
LANGCHAIN_TRACING_V2 = get_secret("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = get_secret("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
POSTGRES_URI = get_secret("POSTGRES_URI", "postgresql://postgres:postgrespassword@localhost:5432/vertexmera")

if PINECONE_API_KEY:

    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT


model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=1024
)


grading_model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=300
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
