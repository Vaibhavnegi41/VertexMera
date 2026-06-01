<div align="center">

# 🔥 VertexMera — CRAG AI System

### Corrective Retrieval Augmented Generation

<img src="https://readme-typing-svg.demolab.com?font=Syne&weight=700&size=22&pause=1000&color=FF8C00&center=true&vCenter=true&width=600&lines=Retrieve+%E2%86%92+Grade+%E2%86%92+Correct+%E2%86%92+Generate;Powered+by+LangGraph+%2B+Pinecone+%2B+Groq;Intelligent+Fallback+to+Web+Search" alt="Typing SVG" />

</div>

---

## 🧠 What is VertexMera?

**VertexMera** is a production-ready **Corrective RAG (CRAG)** system that goes beyond standard retrieval-augmented generation. Instead of blindly trusting retrieved documents, it **grades their relevance** — and if they fall short, it **automatically falls back to live web search** before generating an answer.

> Built as part of a final-year academic project with a real-world deployment on Streamlit Cloud.

---

## ⚡ How It Works

```
User Question
      │
      ▼
 🔴 Retrieve          ← Pinecone vector store (semantic search)
      │
      ▼
 🟠 Grade Documents   ← LLM scores each chunk for relevance
      │
      ├── Relevant? ──► 🟢 Generate Answer  ← RAG prompt (strict)
      │
      └── Not relevant? ──► 🟡 Web Search   ── 🟢 Generate Answer  ← Web prompt (synthesis)
```

The pipeline is built with **LangGraph** as a stateful graph — each node is a distinct, traceable step visible in the UI.

---

## 🚀 Features

- 🔍 **Semantic Retrieval** — HuggingFace embeddings + Pinecone vector store
- ⚖️ **Document Grading** — LLM judges relevance before answering
- 🌐 **Web Search Fallback** — Tavily search when documents are insufficient
- ✨ **Dual Prompt Strategy** — Separate prompts for RAG vs web-sourced answers
- 📄 **PDF Upload** — Embed your own documents directly from the UI
- 🔥 **Fire-themed UI** — Responsive design for mobile and desktop
- 🧩 **Modular Graph** — LangGraph nodes easy to extend or swap

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Orchestration** | LangGraph |
| **LLM** | Groq (llama-3.1-8b-instant) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Store** | Pinecone |
| **Web Search** | Tavily |
| **Framework** | LangChain |

---

## 📁 Project Structure

```
VertexMera/
├── app.py                  # Streamlit frontend
├── backend/
│   ├── __init__.py
│   └── main.py             # LangGraph CRAG pipeline
├── requirements.txt
└── .gitignore
```

---

## 🔧 Local Setup

**1. Clone the repo**
```bash
git clone https://github.com/Vaibhavnegi41/VertexMera.git
cd VertexMera
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create `.streamlit/secrets.toml`**
```toml
PINECONE_API_KEY = "your_pinecone_key"
INDEX_NAME       = "your_index_name"
TAVILY_API_KEY   = "your_tavily_key"
GROQ_API_KEY     = "your_groq_key"
```

**5. Run the app**
```bash
streamlit run app.py
```

---

## ☁️ Deployment

Deployed on **Streamlit Cloud** — secrets managed via the dashboard.

[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 👨‍💻 Author

**Vaibhav Negi**

[![GitHub](https://img.shields.io/badge/GitHub-Vaibhavnegi41-181717?style=flat&logo=github)](https://github.com/Vaibhavnegi41)

---

<div align="center">
  <sub>Built with 🔥 using LangGraph · Pinecone · Groq · Tavily · Streamlit</sub>
</div>
