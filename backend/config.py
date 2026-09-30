"""
config.py — Centralised configuration and model initialization.

All LLMs, embedding models, vector stores, retrievers, and external tool
clients are created here so that every other module can simply import what
it needs without duplicating setup logic.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import Pinecone as LangchainPinecone
from langchain_tavily import TavilySearch

load_dotenv()

# ── Environment variables ────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── LLMs ─────────────────────────────────────────────────────────────────
model = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=GROQ_API_KEY
)

grading_model = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=GROQ_API_KEY
)

# ── Embeddings ───────────────────────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── Vector store & retriever ─────────────────────────────────────────────
vector_store = LangchainPinecone(
    index_name=INDEX_NAME,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# ── Web search tool ──────────────────────────────────────────────────────
web_search_tool = TavilySearch(max_results=2)
