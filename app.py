import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import re
import json
import tempfile
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import Pinecone as LangchainPinecone

st.set_page_config(
    page_title="VertexMera — CRAG AI",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded"
)

load_dotenv()

def get_secret(key: str, default: str = "") -> str:
    """Read a config value from os.environ first, then streamlit.secrets (including nested sections)."""
    val = os.getenv(key)
    if val:
        return str(val).strip()
    try:
        if hasattr(st, "secrets"):
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

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

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

# ── SESSION STATE INIT ────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "latest_docs" not in st.session_state:
    st.session_state.latest_docs = []

# ── HELPER: FORMAT MARKDOWN TO CLEAN HTML ─────────────────
def format_markdown_to_html(text: str) -> str:
    """Converts markdown asterisks, code blocks, lists, and formatting into clean HTML."""
    if not text:
        return ""
    
    # 1. Escape HTML entities
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # 2. Code blocks ```...```
    def replace_code_block(match):
        code = match.group(2)
        return (
            f'<pre style="background:rgba(0,0,0,0.55);border:1px solid rgba(255,255,255,0.1);'
            f'padding:10px 14px;border-radius:8px;overflow-x:auto;font-family:monospace;'
            f'font-size:0.85rem;color:#f0ece4;margin:8px 0;"><code>{code}</code></pre>'
        )
    safe = re.sub(r'```([a-zA-Z0-9_-]*)\n?(.*?)```', replace_code_block, safe, flags=re.DOTALL)
    
    # 3. Inline code `...`
    safe = re.sub(
        r'`([^`]+)`',
        r'<code style="background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px;font-family:monospace;font-size:0.88em;color:#ffd700;">\1</code>',
        safe
    )
    
    # 4. Bold **text** or __text__
    safe = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#ffffff;font-weight:700;">\1</strong>', safe)
    safe = re.sub(r'__(.+?)__', r'<strong style="color:#ffffff;font-weight:700;">\1</strong>', safe)
    
    # 5. Italic *text* or _text_
    safe = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', safe)
    safe = re.sub(r'_([^_]+)_', r'<em>\1</em>', safe)
    
    # 6. Bullet lists
    lines = safe.split('\n')
    formatted_lines = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- ') or stripped.startswith('* ') or stripped.startswith('• '):
            item_text = stripped[2:]
            if not in_list:
                formatted_lines.append('<ul style="margin:6px 0;padding-left:20px;">')
                in_list = True
            formatted_lines.append(f'<li style="margin-bottom:4px;">{item_text}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(line)
    if in_list:
        formatted_lines.append('</ul>')
    
    res = "\n".join(formatted_lines)
    # 7. Convert double linebreaks to spacing, single linebreaks to <br>
    res = res.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return res


# ── STYLES ────────────────────────────────────────────────
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
        padding: 2.2rem 1rem 0.8rem;
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
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.2rem, 6vw, 3.8rem);
        font-weight: 800;
        line-height: 1.05;
        background: linear-gradient(135deg, #ff4500 0%, #ff8c00 35%, #ffd700 65%, #22c55e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.5rem;
        letter-spacing: -0.02em;
    }

    .hero-sub {
        color: #9ca3a0;
        font-size: clamp(0.9rem, 2.2vw, 1.02rem);
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
        margin: 0.8rem auto 1.2rem;
    }

    /* ── INPUT ── */
    .stTextInput > div > div > input {
        background: rgba(18, 18, 18, 0.9) !important;
        border: 1.5px solid rgba(255, 140, 0, 0.35) !important;
        border-radius: 12px !important;
        color: #f0ece4 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1rem !important;
        padding: 13px 18px !important;
        transition: border-color 0.25s, box-shadow 0.25s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ff8c00 !important;
        box-shadow: 0 0 0 3px rgba(255, 140, 0, 0.18) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #555e6d !important;
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
        margin: 1.5rem 0 0.8rem;
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

    /* ── CHAT CONTAINER (NO INNER SCROLLBAR - PAGE EXPANDS NATURALLY) ── */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 14px;
        padding: 6px 0;
        width: 100%;
    }

    .chat-row {
        display: flex;
        width: 100%;
        margin: 0;
    }

    .chat-row.user {
        justify-content: flex-end;
    }

    .chat-row.assistant {
        justify-content: flex-start;
    }

    .chat-bubble {
        border-radius: 14px;
        font-size: 0.94rem;
        line-height: 1.65;
        word-break: break-word;
        box-sizing: border-box;
        animation: fadeSlideIn 0.3s ease-out;
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .chat-bubble.user {
        background: linear-gradient(135deg, rgba(220,60,20,0.22), rgba(255,140,0,0.18));
        border: 1px solid rgba(255,140,0,0.35);
        color: #fde047;
        padding: 12px 18px;
        border-bottom-right-radius: 4px;
        max-width: 80%;
        width: fit-content;
        text-align: left;
    }

    .chat-bubble.assistant {
        background: rgba(15, 17, 16, 0.92);
        border: 1px solid rgba(34, 197, 94, 0.22);
        border-left: 3.5px solid #22c55e;
        color: #e2e8f0;
        padding: 14px 20px;
        border-bottom-left-radius: 4px;
        max-width: 90%;
        width: 100%;
    }

    .chat-role {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .chat-role.user-role { color: #ff8c00; }
    .chat-role.ai-role   { color: #22c55e; }

    .chat-content-body {
        font-size: 0.94rem;
        line-height: 1.68;
    }

    .chat-meta {
        font-size: 0.72rem;
        color: #6b7280;
        margin-top: 8px;
        padding-top: 6px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ── STEP CARDS ── */
    .step-card {
        background: rgba(16, 16, 16, 0.85);
        border: 1px solid rgba(255, 140, 0, 0.15);
        border-left: 3px solid;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
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
        font-size: 1.2rem;
        line-height: 1;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .step-body {}
    .step-title {
        font-weight: 600;
        font-size: 0.92rem;
        color: #f0ece4;
        margin-bottom: 2px;
    }
    .step-sub {
        font-size: 0.78rem;
        color: #6b7280;
    }

    /* ── CONTEXT CHUNKS ── */
    .chunk-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 8px;
    }
    .chunk-card {
        background: rgba(16, 18, 20, 0.85);
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-left: 3px solid #ffd700;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.84rem;
        line-height: 1.6;
        color: #cbd5e1;
        word-break: break-word;
    }
    .chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #ffd700;
        margin-bottom: 6px;
    }
    .chunk-content {
        max-height: 220px;
        overflow-y: auto;
        padding-right: 4px;
        white-space: pre-wrap;
    }

    /* ── SIDEBAR CHUNKS CARD ── */
    .sidebar-chunk {
        background: rgba(22, 22, 22, 0.9);
        border: 1px solid rgba(255, 140, 0, 0.25);
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 0.8rem;
        line-height: 1.5;
        color: #9ca3af;
    }
    .sidebar-chunk-title {
        font-weight: 700;
        color: #fbbf24;
        font-size: 0.75rem;
        margin-bottom: 4px;
    }

    /* ── SUGGESTED QUESTIONS ── */
    .suggestion-card {
        background: rgba(16, 16, 16, 0.8);
        border: 1px solid rgba(255, 140, 0, 0.2);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        color: #fbbf24;
        font-size: 0.88rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .suggestion-card:hover {
        border-color: #ff8c00;
        background: rgba(255, 140, 0, 0.08);
        transform: translateX(4px);
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
        background: rgba(14, 14, 14, 0.75) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stExpander"] summary {
        color: #ffd700 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* ── FOOTER ── */
    .footer {
        text-align: center;
        color: #374151;
        font-size: 0.78rem;
        padding: 2.5rem 0 1rem;
        letter-spacing: 0.04em;
    }
    .footer span {
        color: #ff8c00;
    }

    /* hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0 !important; max-width: 760px !important; }
</style>
""", unsafe_allow_html=True)


# ── LOCALSTORAGE SYNC ─────────────────────────────────────
def inject_localstorage_sync():
    """Inject JS to sync chat history with browser localStorage."""
    messages_json = json.dumps(st.session_state.messages)
    components.html(f"""
    <script>
        const STORAGE_KEY = 'vertexmera_chat_history';
        const currentMessages = {messages_json};
        if (currentMessages.length > 0) {{
            localStorage.setItem(STORAGE_KEY, JSON.stringify(currentMessages));
        }}
    </script>
    """, height=0)


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

                    # Generate suggested questions from the PDF
                    with st.spinner("Generating suggested questions..."):
                        try:
                            from backend.config import model as llm
                            sample_text = "\n\n".join([c.page_content for c in chunks[:5]])
                            suggestion_response = llm.invoke(
                                f"Based on the following document content, generate exactly 4 "
                                f"diverse and interesting questions that a student or researcher "
                                f"might ask about this material. Return ONLY the questions, one "
                                f"per line, numbered 1-4. No explanations.\n\n"
                                f"DOCUMENT:\n{sample_text[:3000]}"
                            )
                            raw_questions = suggestion_response.content.strip().split("\n")
                            cleaned = []
                            for q in raw_questions:
                                q = q.strip()
                                if q and len(q) > 10:
                                    for prefix in ["1.", "2.", "3.", "4.", "1)", "2)", "3)", "4)"]:
                                        if q.startswith(prefix):
                                            q = q[len(prefix):].strip()
                                    cleaned.append(q)
                            st.session_state.suggested_questions = cleaned[:4]
                        except Exception:
                            pass

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # ── SIDEBAR CHUNKS PANEL ──
    st.markdown("---")
    st.markdown("### 📑 Retrieved Context Chunks")
    
    current_docs = st.session_state.get("latest_docs", [])
    if current_docs:
        st.markdown(f"<div style='color:#ffd700;font-size:0.78rem;margin-bottom:8px'>Showing {len(current_docs)} chunk(s) from latest query:</div>", unsafe_allow_html=True)
        for idx, doc_text in enumerate(current_docs):
            with st.expander(f"Chunk #{idx+1} ({len(doc_text)} chars)", expanded=(idx == 0)):
                st.markdown(f"<div style='color:#cbd5e1;font-size:0.8rem;line-height:1.5;white-space:pre-wrap;'>{doc_text}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#4b5563;font-size:0.8rem;font-style:italic;'>No chunks retrieved yet. Ask a question to see relevant chunks here.</div>", unsafe_allow_html=True)

    # ── Chat history controls ──
    st.markdown("---")
    st.markdown("### 💬 Chat History")
    msg_count = len(st.session_state.messages) // 2
    st.markdown(f"<div style='color:#6b7280;font-size:0.82rem'>{msg_count} conversation{'s' if msg_count != 1 else ''}</div>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.suggested_questions = []
        st.session_state.latest_docs = []
        components.html("""
        <script>
            localStorage.removeItem('vertexmera_chat_history');
        </script>
        """, height=0)
        st.rerun()

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


# ── SUGGESTED QUESTIONS ──────────────────────────────────
if st.session_state.suggested_questions and len(st.session_state.messages) == 0:
    st.markdown('<div class="section-label">Suggested questions from your document</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, question in enumerate(st.session_state.suggested_questions):
        with cols[i % 2]:
            if st.button(f"💡 {question}", key=f"suggest_{i}", use_container_width=True):
                st.session_state._pending_question = question
                st.rerun()


# ── CHAT DISPLAY (NO INNER SCROLLBAR - PAGE EXPANDS NATURALLY) ──
if st.session_state.messages:
    st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)

    chat_html = '<div class="chat-container">'
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            content_safe = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            chat_html += f"""
            <div class="chat-row user">
                <div class="chat-bubble user">
                    <div class="chat-role user-role">You</div>
                    <div class="chat-content-body">{content_safe}</div>
                </div>
            </div>"""
        else:
            formatted_ai_content = format_markdown_to_html(content)

            steps_html = ""
            if msg.get("steps"):
                steps_info = " → ".join(msg["steps"])
                steps_html = f'<div class="chat-meta">Pipeline: {steps_info}</div>'

            source_tag = ""
            if msg.get("web_searched"):
                source_tag = '<span style="color:#ffd700;font-size:0.7rem;margin-left:8px;font-weight:600">🌐 Web</span>'
            else:
                source_tag = '<span style="color:#ff4500;font-size:0.7rem;margin-left:8px;font-weight:600">📄 RAG</span>'

            chat_html += f"""
            <div class="chat-row assistant">
                <div class="chat-bubble assistant">
                    <div class="chat-role ai-role">VertexMera {source_tag}</div>
                    <div class="chat-content-body">{formatted_ai_content}</div>
                    {steps_html}
                </div>
            </div>"""

    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)


# ── INPUT ─────────────────────────────────────────────────
pending = st.session_state.pop("_pending_question", None)

query = st.text_input(
    label="question",
    label_visibility="collapsed",
    placeholder="Ask anything — e.g. What is insightflow?",
    value=pending if pending else ""
)
submit = st.button("🔥 Generate Answer")

if pending and not submit:
    submit = True


# ── PIPELINE EXECUTION ────────────────────────────────────
if submit and query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.spinner("Running CRAG pipeline..."):
        try:
            start_time = time.time()

            results = chatbot.invoke({
                'question': query,
                'documents': [],
                'steps': [],
                'generation': '',
                'web_searched': False
            })

            elapsed = round(time.time() - start_time, 2)

            generation = results.get("generation", "No response generated.")
            steps = results.get("steps", [])
            web_searched = results.get("web_searched", False)
            raw_docs = results.get("documents", [])

            # Extract document strings for display
            doc_strings = []
            for d in raw_docs:
                if hasattr(d, "page_content"):
                    doc_strings.append(d.page_content)
                elif isinstance(d, dict) and "page_content" in d:
                    doc_strings.append(d["page_content"])
                elif isinstance(d, str):
                    doc_strings.append(d)

            st.session_state.latest_docs = doc_strings

            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": generation,
                "steps": steps,
                "web_searched": web_searched,
                "time": elapsed,
                "docs": doc_strings
            })

            inject_localstorage_sync()
            st.rerun()

        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()

elif submit and not query:
    st.warning("Please enter a question first.")


# ── LATEST RESULT DETAILS (TRACE & RELEVANT CHUNKS) ────────
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_msg = st.session_state.messages[-1]

    # Pipeline Trace
    if last_msg.get("steps"):
        st.markdown('<div class="section-label">Pipeline trace (latest)</div>', unsafe_allow_html=True)

        steps_html = ""
        step_meta = {
            "retrieval": ("🔴", "retrieve",  "Queried Pinecone vector store"),
            "grade":     ("🟠", "grade",     "Scored document relevance"),
            "search":    ("🟡", "search",    "Fell back to Tavily web search"),
            "generation":("🟢", "generate",  "Generated final answer"),
        }

        for i, step in enumerate(last_msg["steps"]):
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

        if last_msg.get("time"):
            steps_html += f'<div style="color:#4b5563;font-size:0.78rem;text-align:right;margin-top:4px">⏱ {last_msg["time"]}s</div>'

        st.markdown(steps_html, unsafe_allow_html=True)

    # Relevant Chunks Panel in Main View
    docs_to_show = last_msg.get("docs", [])
    if docs_to_show:
        with st.expander(f"📚 Retrieved Context Chunks ({len(docs_to_show)})", expanded=False):
            for i, chunk in enumerate(docs_to_show):
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-header">
                        <span>Chunk #{i+1}</span>
                        <span>{len(chunk)} characters</span>
                    </div>
                    <div class="chunk-content">{chunk}</div>
                </div>
                """, unsafe_allow_html=True)

inject_localstorage_sync()

# ── FOOTER ────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Powered by <span>LangGraph</span> · <span>Pinecone</span> · <span>Tavily</span> · <span>Streamlit</span>
</div>
""", unsafe_allow_html=True)