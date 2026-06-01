import streamlit as st
import sys
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import Pinecone as LangchainPinecone

st.set_page_config(
    page_title="CRAG AI System",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

@st.cache_resource(show_spinner="Warming up embedding engine...")
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource(show_spinner="Compiling CRAG pipeline...")
def load_chatbot():
    from backend.main import chatbot
    return chatbot

embedding_model = get_embedding_model()
chatbot = load_chatbot()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: #0a0a0a;
        color: #f0ece4;
    }

    /* Animated fire mesh background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background:
            radial-gradient(ellipse 80% 60% at 20% 10%, rgba(220, 60, 20, 0.18) 0%, transparent 60%),
            radial-gradient(ellipse 60% 50% at 80% 5%,  rgba(255, 140, 0, 0.14) 0%, transparent 55%),
            radial-gradient(ellipse 50% 40% at 50% 90%, rgba(34, 197, 94, 0.10) 0%, transparent 55%),
            radial-gradient(ellipse 70% 40% at 10% 80%, rgba(250, 204, 21, 0.08) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    /* ── HERO HEADER ── */
    .hero-wrap {
        text-align: center;
        padding: 3rem 1rem 1.5rem;
        position: relative;
    }

    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, rgba(220,60,20,0.2), rgba(255,140,0,0.2));
        border: 1px solid rgba(255,140,0,0.35);
        color: #fbbf24;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 20px;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.4rem, 7vw, 4.2rem);
        font-weight: 800;
        line-height: 1.05;
        background: linear-gradient(135deg, #ff4500 0%, #ff8c00 35%, #ffd700 65%, #22c55e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.6rem;
        letter-spacing: -0.02em;
    }

    .hero-sub {
        color: #9ca3a0;
        font-size: clamp(0.9rem, 2.5vw, 1.05rem);
        font-weight: 400;
        margin: 0;
        letter-spacing: 0.01em;
    }

    /* ── DIVIDER ── */
    .flame-divider {
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #ff4500, #ffd700, #22c55e);
        border-radius: 2px;
        margin: 1.2rem auto 2rem;
    }

    /* ── INPUT ── */
    .stTextInput > div > div > input {
        background: rgba(18, 18, 18, 0.9) !important;
        border: 1.5px solid rgba(255, 140, 0, 0.3) !important;
        border-radius: 12px !important;
        color: #f0ece4 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1rem !important;
        padding: 14px 18px !important;
        transition: border-color 0.25s, box-shadow 0.25s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ff8c00 !important;
        box-shadow: 0 0 0 3px rgba(255, 140, 0, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #4b5563 !important;
    }

    /* ── BUTTON ── */
    .stButton > button {
        background: linear-gradient(135deg, #dc3c14 0%, #ff6b00 50%, #ffd700 100%) !important;
        color: #0a0a0a !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        width: 100% !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 20px rgba(220, 60, 20, 0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(255, 140, 0, 0.45) !important;
        filter: brightness(1.08) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── SECTION LABELS ── */
    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #ff8c00;
        margin: 2rem 0 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(255,140,0,0.3), transparent);
    }

    /* ── STEP CARDS ── */
    .step-card {
        background: rgba(16, 16, 16, 0.85);
        border: 1px solid rgba(255, 140, 0, 0.15);
        border-left: 3px solid;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s, border-color 0.2s;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .step-card:hover {
        transform: translateX(4px);
    }
    .step-card.retrieve  { border-left-color: #ff4500; }
    .step-card.grade     { border-left-color: #ff8c00; }
    .step-card.search    { border-left-color: #ffd700; }
    .step-card.generate  { border-left-color: #22c55e; }
    .step-card.default   { border-left-color: #6b7280; }

    .step-icon {
        font-size: 1.3rem;
        line-height: 1;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .step-body {}
    .step-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #f0ece4;
        margin-bottom: 2px;
    }
    .step-sub {
        font-size: 0.8rem;
        color: #6b7280;
    }

    /* ── ANSWER BOX ── */
    .answer-box {
        background: rgba(16, 16, 16, 0.9);
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-top: 3px solid #22c55e;
        border-radius: 0 0 14px 14px;
        padding: 22px 24px;
        line-height: 1.75;
        font-size: 1rem;
        color: #d1fae5;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .answer-header {
        background: linear-gradient(90deg, rgba(34,197,94,0.15), transparent);
        border: 1px solid rgba(34,197,94,0.2);
        border-bottom: none;
        border-radius: 14px 14px 0 0;
        padding: 10px 24px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #22c55e;
    }

    /* ── CONTEXT CHUNK ── */
    .chunk-card {
        background: rgba(14, 14, 14, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        line-height: 1.65;
        color: #9ca3a0;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .chunk-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #ffd700;
        margin-bottom: 8px;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: #0d0d0d !important;
        border-right: 1px solid rgba(255,140,0,0.15) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {
        color: #9ca3a0 !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stSidebar"] h3 {
        color: #ff8c00 !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 1rem !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255,140,0,0.04) !important;
        border: 1.5px dashed rgba(255,140,0,0.25) !important;
        border-radius: 10px !important;
    }

    /* ── EXPANDER ── */
    [data-testid="stExpander"] {
        background: rgba(14, 14, 14, 0.7) !important;
        border: 1px solid rgba(255, 215, 0, 0.15) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary {
        color: #ffd700 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* ── SPINNER ── */
    .stSpinner > div {
        border-top-color: #ff8c00 !important;
    }

    /* ── FOOTER ── */
    .footer {
        text-align: center;
        color: #374151;
        font-size: 0.78rem;
        padding: 2rem 0 1rem;
        letter-spacing: 0.04em;
    }
    .footer span {
        color: #ff8c00;
    }

    /* ── RESPONSIVE ── */
    @media (max-width: 768px) {
        .hero-wrap { padding: 2rem 0.5rem 1rem; }
        .answer-box { padding: 16px; font-size: 0.95rem; }
        .step-card  { padding: 12px 14px; }
    }

    /* hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0 !important; max-width: 760px !important; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔥 Knowledge Base")
    st.markdown("Upload a PDF to embed it into Pinecone.")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Embed Document", use_container_width=True):
            with st.spinner("Chunking & embedding..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = splitter.split_documents(docs)

                    LangchainPinecone.from_documents(
                        chunks,
                        embedding=embedding_model,
                        index_name=INDEX_NAME
                    )

                    os.unlink(tmp_path)
                    st.success(f"✅ {len(chunks)} chunks embedded.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")
    st.markdown(
        "<div style='color:#4b5563;font-size:0.75rem;line-height:1.6'>"
        "🔴 Retrieve &nbsp;→&nbsp; ⚖️ Grade<br>"
        "🌐 Web fallback &nbsp;→&nbsp; 🟢 Generate"
        "</div>",
        unsafe_allow_html=True
    )

# ── HERO ─────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">Corrective RAG · v1.0</div>
    <div class="hero-title">VertexMera Explorer</div>
    <p class="hero-sub">Retrieve · Grade · Correct · Generate</p>
    <div class="flame-divider"></div>
</div>
""", unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────
query = st.text_input(
    label="question",
    label_visibility="collapsed",
    placeholder="Ask anything — e.g. What is gradient descent?"
)
submit = st.button("🔥 Generate Answer")

# ── PIPELINE ──────────────────────────────────────────────
if submit and query:
    with st.spinner("Running CRAG pipeline..."):
        try:
            results = chatbot.invoke({
                'question': query,
                'documents': [],
                'steps': [],
                'generation': "",
                'web_searched': False
            })

            # ── STEPS ──
            st.markdown('<div class="section-label">Pipeline trace</div>', unsafe_allow_html=True)

            steps_html = ""
            step_meta = {
                "retrieval": ("🔴", "retrieve",  "Queried Pinecone vector store"),
                "grade":     ("🟠", "grade",     "Scored document relevance"),
                "search":    ("🟡", "search",    "Fell back to Tavily web search"),
                "generation":("🟢", "generate",  "Generated final answer"),
            }

            for i, step in enumerate(results.get("steps", [])):
                key = next((k for k in step_meta if k in step.lower()), None)
                icon, cls, desc = step_meta[key] if key else ("⚪", "default", "Graph node executed")
                label = step.replace("_", " ").title()
                steps_html += f"""
                <div class="step-card {cls}">
                    <div class="step-icon">{icon}</div>
                    <div class="step-body">
                        <div class="step-title">Step {i+1} — {label}</div>
                        <div class="step-sub">{desc}</div>
                    </div>
                </div>"""

            st.markdown(steps_html, unsafe_allow_html=True)

            # ── ANSWER ──
            st.markdown('<div class="section-label">Final answer</div>', unsafe_allow_html=True)
            generation = results.get("generation", "No response generated.")
            st.markdown(f"""
            <div class="answer-header">✦ AI Response</div>
            <div class="answer-box">{generation}</div>
            """, unsafe_allow_html=True)

            # ── CONTEXT ──
            with st.expander("🔍 Retrieved context chunks"):
                retrieved_docs = results.get("documents", [])
                if not retrieved_docs:
                    st.warning("No chunks retrieved.")
                else:
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"""
                        <div class="chunk-card">
                            <div class="chunk-label">Chunk {i + 1}</div>
                            {doc.page_content}
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")

elif submit and not query:
    st.warning("Please enter a question first.")

# ── FOOTER ────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Powered by <span>LangGraph</span> · <span>Pinecone</span> · <span>Tavily</span> · <span>Streamlit</span>
</div>
""", unsafe_allow_html=True)