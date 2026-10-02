<div align="center">

# 🔥 VertexMera — CRAG AI System

### Privacy-First Corrective Retrieval Augmented Generation

<img src="https://readme-typing-svg.demolab.com?font=Syne&weight=700&size=22&pause=1000&color=FF8C00&center=true&vCenter=true&width=700&lines=Sanitize+%E2%86%92+Retrieve+%E2%86%92+Grade+%E2%86%92+Correct+%E2%86%92+Generate;Powered+by+LangGraph+%2B+Pinecone+%2B+Groq;Zero-PII+Vector+Storage+%7C+Live+Web+Fallback" alt="Typing SVG" />

</div>

---

## 🧠 What is VertexMera?

**VertexMera** is a production-ready **Corrective RAG (CRAG)** system that goes beyond standard retrieval-augmented generation. Instead of blindly trusting retrieved documents, it **grades their relevance** — and if they fall short, it **automatically falls back to live web search** before generating an answer.

It is also **privacy-first**: a hybrid PII guardrail sanitizes documents *before* they are embedded and sanitizes questions *before* they reach any external LLM, so private data never leaves your pipeline.

> Because blindly trusting Google results is so 2020 — VertexMera actually reads, judges, and fact-checks before it answers.

---

## ⚡ How It Works

```
User Question
      │
      ▼
 🛡️ PII Guardrail     ← Regex + spaCy NER sanitization (~6 ms)
      │
      ▼
 🔴 Retrieve          ← Pinecone vector store (semantic search + cosine scores)
      │
      ├── Top score ≥ 0.85? ──► ⚡ Fast Path (skip grading) ──► 🟢 Generate Answer
      │
      ▼
 🟠 Grade Documents   ← LLM structured-output classifier (ScoreFormat)
      │
      ├── Relevant? ──► 🟢 Generate Answer  ← RAG prompt (strict)
      │
      └── Not relevant? ──► 🟡 Web Search (Tavily)
                                  │
                                  ▼
                          🛡️ PII Guardrail on web results
                                  │
                                  ▼
                          🟢 Generate Answer  ← Web prompt (synthesis)
```

The pipeline is a **LangGraph state machine**:
`START → PII Guardrail → Retrieve → Grade → [Web Search | Generate] → END`

Each node is a distinct, traceable step visible live in the UI.

---

## 🚀 Features

### 🛡️ 1. Hybrid PII Guardrails & Privacy-Preserving Ingestion
- **Two-pass hybrid sanitization**
  - **Pass 1 — Deterministic regex engine** for structured PII: emails, phone numbers, SSNs, credit cards, IP addresses, dates of birth, Indian **PAN / Aadhaar**, and API keys
  - **Pass 2 — Lightweight, optimized spaCy NER** for semantic entities: person names, organizations, and locations
- **Zero-PII vector storage** — documents are sanitized *before* chunking and embedding, so Pinecone only ever stores redacted tokens such as `[PERSON_NAME_1]` and `[EMAIL_1]`
- **Real-time question sanitization in ~6 ms**, so external LLMs never see private data

### 🔄 2. Corrective RAG (CRAG) State Graph
- **Deterministic state control** — LangGraph orchestrates a multi-stage state machine instead of a simple unverified RAG chain
- **Relevance grading** — a structured-output classifier (`ScoreFormat` boolean model) on Groq checks whether retrieved chunks contain substantive facts before they reach the generator
- **Modular graph** — nodes are easy to extend or swap

### ⚡ 3. Confidence-Score Pre-Filtering (Sub-Second Fast Path)
- **Similarity-aware retrieval** — cosine similarity scores are attached directly to retrieved chunks
- **Latency optimization** — if the top chunk scores **≥ 0.85**, the pipeline treats it as a high-confidence match and **skips the grading LLM call**, cutting end-to-end response time by **~1.2 seconds**

### 🌐 4. Automated Live Web Search Fallback (Tavily)
- **Autonomous knowledge correction** — when the uploaded document doesn't contain the answer (out-of-domain or missing context), a conditional branch triggers Tavily web search
- **Sanitized web context** — search results are routed through the PII guardrails before reaching the generation prompt, so live internet context is synthesized safely

### 🎯 5. Strict Anti-Hallucination & Grounded Generation
- **Dual-prompt strategy** — separate specialized prompts for document RAG (`rag_prompt`) and web synthesis (`web_prompt`)
- **Strict grounding rules** — the LLM answers only from the provided context, avoids meta-phrasing, highlights key terms in **bold**, and explicitly states when information is missing rather than guessing

### 🖥️ 6. Interactive UI with Full Pipeline Observability
- **Live pipeline trace** — step cards with distinct icons, color-coded status badges, and real-time execution timers for every executed graph node
- **Single-source context viewer** — expandable accordion showing retrieved chunks with character lengths
- **Smart document suggestions** — on upload, the document is analyzed and **4 suggested prompt questions** are generated automatically
- **Session persistence** — chat history syncs across page refreshes via browser `localStorage`
- **PDF upload** — embed your own documents directly from the UI
- **Fire-themed custom design system** — responsive for mobile and desktop

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit (custom design system) |
| **Orchestration** | LangGraph |
| **LLM** | Groq (llama-3.1-8b-instant) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Store** | Pinecone |
| **Web Search** | Tavily |
| **PII Detection** | Regex engine + spaCy NER |
| **Framework** | LangChain |

---

## 📁 Project Structure

```
VertexMera/
├── app.py                  # Streamlit frontend
├── backend/
│   ├── __init__.py
│   └── main.py             # LangGraph CRAG pipeline + PII guardrails
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

**4. Download the spaCy model** *(used for PII entity detection — replace with the model your project uses)*
```bash
python -m spacy download en_core_web_sm
```

**5. Create `.streamlit/secrets.toml`**
```toml
PINECONE_API_KEY = "your_pinecone_key"
INDEX_NAME       = "your_index_name"
TAVILY_API_KEY   = "your_tavily_key"
GROQ_API_KEY     = "your_groq_key"
```

**6. Run the app**
```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Vaibhav Negi**

[![GitHub](https://img.shields.io/badge/GitHub-Vaibhavnegi41-181717?style=flat&logo=github)](https://github.com/Vaibhavnegi41)

---

<div align="center">
  <sub>Built with 🔥 using LangGraph · Pinecone · Groq · Tavily · spaCy · Streamlit</sub>
</div>
