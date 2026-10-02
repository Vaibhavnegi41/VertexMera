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
from langchain_core.documents import Document

st.set_page_config(
    page_title="VertexMera — CRAG AI",
    page_icon=None,
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
    from backend.pii_guardrails import get_guardrails
    get_guardrails().mask("warmup")
    from backend.main import chatbot
    return chatbot

embedding_model = get_embedding_model()
chatbot = load_chatbot()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "latest_docs" not in st.session_state:
    st.session_state.latest_docs = []

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
            f'<pre style="background:rgba(20,20,22,0.9);border:1px solid rgba(255,255,255,0.12);'
            f'padding:10px 14px;border-radius:8px;overflow-x:auto;font-family:monospace;'
            f'font-size:0.85rem;color:#f4f4f5;margin:8px 0;"><code>{code}</code></pre>'
        )
    safe = re.sub(r'```([a-zA-Z0-9_-]*)\n?(.*?)```', replace_code_block, safe, flags=re.DOTALL)
    
    # 3. Inline code `...`
    safe = re.sub(
        r'`([^`]+)`',
        r'<code style="background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px;font-family:monospace;font-size:0.88em;color:#ffffff;">\1</code>',
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


# CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: #050505;
        color: #f4f4f5;
    }

    /* Ambient monochrome gradient mesh background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background:
            radial-gradient(ellipse 80% 60% at 20% 10%, rgba(255, 255, 255, 0.05) 0%, transparent 60%),
            radial-gradient(ellipse 60% 50% at 80% 5%,  rgba(200, 200, 200, 0.03) 0%, transparent 55%),
            radial-gradient(ellipse 50% 40% at 50% 90%, rgba(255, 255, 255, 0.03) 0%, transparent 55%),
            radial-gradient(ellipse 70% 40% at 10% 80%, rgba(180, 180, 180, 0.02) 0%, transparent 50%);
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
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #e4e4e7;
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
        background: linear-gradient(135deg, #ffffff 0%, #e4e4e7 35%, #a1a1aa 70%, #71717a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.5rem;
        letter-spacing: -0.02em;
    }

    .hero-sub {
        color: #a1a1aa;
        font-size: clamp(0.9rem, 2.2vw, 1.02rem);
        font-weight: 400;
        margin: 0;
        letter-spacing: 0.01em;
    }

    /* ── DIVIDER ── */
    .flame-divider {
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.6), transparent);
        border-radius: 2px;
        margin: 0.8rem auto 1.2rem;
    }

    /* ── INPUT ── */
    .stTextInput > div > div > input {
        background: rgba(18, 18, 20, 0.95) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1rem !important;
        padding: 13px 18px !important;
        transition: border-color 0.25s, box-shadow 0.25s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.12) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #71717a !important;
    }

    /* ── BUTTON ── */
    .stButton > button {
        background: linear-gradient(135deg, #ffffff 0%, #e4e4e7 50%, #d4d4d8 100%) !important;
        color: #09090b !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        width: 100% !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 18px rgba(255, 255, 255, 0.12) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(255, 255, 255, 0.25) !important;
        filter: brightness(1.05) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── HITL ACTION (MATCHED WITH MONOCHROME ASSISTANT BUBBLE) ── */
    .hitl-sub-banner {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.14);
        display: flex;
        flex-direction: column;
        gap: 3px;
        font-size: 0.84rem;
        color: #a1a1aa;
    }
    .hitl-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }
    .hitl-btn-wrap {
        max-width: 90%;
        margin: 4px 0 16px 0;
    }
    .hitl-btn-wrap .stButton > button {
        background: rgba(24, 24, 27, 0.95) !important;
        color: #ffffff !important;
        border: 1.5px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.4rem !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        width: auto !important;
        box-shadow: 0 2px 12px rgba(255, 255, 255, 0.08) !important;
        transition: all 0.2s ease !important;
    }
    .hitl-btn-wrap .stButton > button:hover {
        background: #ffffff !important;
        border-color: #ffffff !important;
        color: #09090b !important;
        box-shadow: 0 4px 18px rgba(255, 255, 255, 0.25) !important;
        transform: translateY(-1.5px) !important;
    }

    /* ── SECTION LABELS ── */
    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #e4e4e7;
        margin: 1.5rem 0 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.25), transparent);
    }

    /* ── CHAT CONTAINER ── */
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
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #f4f4f5;
        padding: 12px 18px;
        border-bottom-right-radius: 4px;
        max-width: 80%;
        width: fit-content;
        text-align: left;
    }

    .chat-bubble.assistant {
        background: rgba(18, 18, 20, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-left: 3.5px solid #ffffff;
        color: #e4e4e7;
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
    .chat-role.user-role { color: #a1a1aa; }
    .chat-role.ai-role   { color: #ffffff; }

    .chat-content-body {
        font-size: 0.94rem;
        line-height: 1.68;
    }

    .chat-meta {
        font-size: 0.72rem;
        color: #71717a;
        margin-top: 8px;
        padding-top: 6px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ── STEP CARDS ── */
    .step-card {
        background: rgba(18, 18, 20, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 3px solid #ffffff;
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
        border-color: rgba(255, 255, 255, 0.3);
    }
    .step-card.guardrail { border-left-color: #d4d4d8; }
    .step-card.retrieve  { border-left-color: #ffffff; }
    .step-card.grade     { border-left-color: #a1a1aa; }
    .step-card.search    { border-left-color: #e4e4e7; }
    .step-card.generate  { border-left-color: #ffffff; }
    .step-card.default   { border-left-color: #71717a; }

    .step-icon {
        font-size: 0.85rem;
        font-weight: 700;
        font-family: monospace;
        color: #ffffff;
        background: rgba(255, 255, 255, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
        line-height: 1.2;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .step-body {}
    .step-title {
        font-weight: 600;
        font-size: 0.92rem;
        color: #ffffff;
        margin-bottom: 2px;
    }
    .step-sub {
        font-size: 0.78rem;
        color: #a1a1aa;
    }

    /* ── CONTEXT CHUNKS ── */
    .chunk-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 8px;
    }
    .chunk-card {
        background: rgba(18, 18, 20, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-left: 3px solid #ffffff;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.84rem;
        line-height: 1.6;
        color: #e4e4e7;
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
        color: #ffffff;
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
        background: rgba(18, 18, 20, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 0.8rem;
        line-height: 1.5;
        color: #a1a1aa;
    }
    .sidebar-chunk-title {
        font-weight: 700;
        color: #ffffff;
        font-size: 0.75rem;
        margin-bottom: 4px;
    }

    /* ── SUGGESTED QUESTIONS ── */
    .suggestion-card {
        background: rgba(18, 18, 20, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        color: #f4f4f5;
        font-size: 0.88rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .suggestion-card:hover {
        border-color: #ffffff;
        background: rgba(255, 255, 255, 0.08);
        transform: translateX(4px);
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: #09090b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {
        color: #a1a1aa !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 1rem !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1.5px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
    }

    /* ── EXPANDER ── */
    [data-testid="stExpander"] {
        background: rgba(18, 18, 20, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stExpander"] summary {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* ── FOOTER ── */
    .footer {
        text-align: center;
        color: #52525b;
        font-size: 0.78rem;
        padding: 2.5rem 0 1rem;
        letter-spacing: 0.04em;
    }
    .footer span {
        color: #a1a1aa;
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


# Sidebar
with st.sidebar:
    st.markdown("### Knowledge Base")
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

                    from backend.pii_guardrails import mask_documents
                    with st.spinner("Applying Hybrid PII Guardrails (Masking & Redaction)..."):
                        docs = mask_documents(docs)

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
                    st.success(f"Success: {len(chunks)} PII-redacted chunks embedded securely.")

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


    # Chat history controls
    st.markdown("---")
    st.markdown("### Chat History")
    msg_count = len(st.session_state.messages) // 2
    st.markdown(f"<div style='color:#71717a;font-size:0.82rem'>{msg_count} conversation{'s' if msg_count != 1 else ''}</div>", unsafe_allow_html=True)

    if st.button("Clear Chat", use_container_width=True):
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
        "<div style='color:#71717a;font-size:0.75rem;line-height:1.6'>"
        "Retrieve &nbsp;→&nbsp; Grade<br>"
        "Web fallback &nbsp;→&nbsp; Generate"
        "</div>",
        unsafe_allow_html=True
    )


# Hero Header
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">Corrective RAG · v1.0</div>
    <div class="hero-title">VertexMera Explorer</div>
    <p class="hero-sub">Retrieve · Grade · Correct · Generate</p>
    <div class="flame-divider"></div>
</div>
""", unsafe_allow_html=True)


# Suggested Questions
if st.session_state.suggested_questions and len(st.session_state.messages) == 0:
    st.markdown('<div class="section-label">Suggested questions from your document</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, question in enumerate(st.session_state.suggested_questions):
        with cols[i % 2]:
            if st.button(f"[?] {question}", key=f"suggest_{i}", use_container_width=True):
                st.session_state._pending_question = question
                st.rerun()


# Chat Display
if st.session_state.messages:
    st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)

    chat_html = '<div class="chat-container">'
    for idx, msg in enumerate(st.session_state.messages):
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
            if msg.get("is_detailed"):
                source_tag = '<span style="color:#ffffff;font-size:0.7rem;margin-left:8px;font-weight:600;letter-spacing:0.04em">[Web · Deep Dive]</span>'
            elif msg.get("web_searched"):
                source_tag = '<span style="color:#e4e4e7;font-size:0.7rem;margin-left:8px;font-weight:600;letter-spacing:0.04em">[Web]</span>'
            else:
                source_tag = '<span style="color:#a1a1aa;font-size:0.7rem;margin-left:8px;font-weight:600;letter-spacing:0.04em">[RAG]</span>'

            hitl_invite = ""
            if idx == len(st.session_state.messages) - 1 and msg.get("web_searched") and not msg.get("is_detailed"):
                hitl_invite = """
                <div class="hitl-sub-banner">
                    <div class="hitl-tag">LIVE WEB INTELLIGENCE</div>
                    <div>Would you like an in-depth, structured technical explanation with all search findings?</div>
                </div>
                """

            chat_html += f"""
            <div class="chat-row assistant">
                <div class="chat-bubble assistant">
                    <div class="chat-role ai-role">VertexMera {source_tag}</div>
                    <div class="chat-content-body">{formatted_ai_content}</div>
                    {hitl_invite}
                    {steps_html}
                </div>
            </div>"""

    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # ── HITL (HUMAN-IN-THE-LOOP) ACTION BUTTON ──
    # Rendered directly beneath the assistant message with matching width & styling
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        latest_msg = st.session_state.messages[-1]
        if latest_msg.get("web_searched") and not latest_msg.get("is_detailed"):
            st.markdown('<div class="hitl-btn-wrap">', unsafe_allow_html=True)
            if st.button("Yes, Generate Detailed Explanation", key=f"hitl_detail_{len(st.session_state.messages)}"):
                with st.spinner("Synthesizing in-depth web analysis..."):
                    try:
                        from backend.nodes import generate_detailed_web_node

                        hitl_start = time.time()
                        last_user_q = ""
                        for m in reversed(st.session_state.messages[:-1]):
                            if m["role"] == "user":
                                last_user_q = m["content"]
                                break

                        doc_objs = [
                            Document(page_content=d) if isinstance(d, str) else d
                            for d in latest_msg.get("docs", [])
                        ]

                        detail_res = generate_detailed_web_node({
                            "question": last_user_q,
                            "raw_question": last_user_q,
                            "documents": doc_objs,
                            "steps": list(latest_msg.get("steps", [])),
                            "web_searched": True,
                            "detail_requested": True
                        })

                        hitl_elapsed = round(time.time() - hitl_start, 2)
                        latest_msg["content"] = detail_res["generation"]
                        latest_msg["steps"] = detail_res["steps"]
                        latest_msg["is_detailed"] = True
                        latest_msg["time"] = hitl_elapsed

                        inject_localstorage_sync()
                        st.rerun()
                    except Exception as hitl_err:
                        st.error(f"Error expanding explanation: {hitl_err}")
            st.markdown('</div>', unsafe_allow_html=True)


# Query Input
pending = st.session_state.pop("_pending_question", None)

query = st.text_input(
    label="question",
    label_visibility="collapsed",
    placeholder="Ask anything — e.g. What is insightflow?",
    value=pending if pending else ""
)
submit = st.button("Generate Answer")

if pending and not submit:
    submit = True


# Pipeline Execution
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
                'web_searched': False,
                'pii_map': {},
                'pii_redacted': False
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


# Latest Result Details
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_msg = st.session_state.messages[-1]

    # Pipeline Trace
    if last_msg.get("steps"):
        st.markdown('<div class="section-label">Pipeline trace (latest)</div>', unsafe_allow_html=True)

        steps_html = ""
        step_meta = {
            "guardrail":  ("[01]", "guardrail", "Applied Hybrid PII Redaction Guardrail (Deterministic + Semantic NER)"),
            "pii":        ("[01]", "guardrail", "Applied Hybrid PII Redaction Guardrail (Deterministic + Semantic NER)"),
            "retrieval":  ("[02]", "retrieve",  "Queried Pinecone vector store"),
            "grade":      ("[03]", "grade",     "Scored document relevance"),
            "search":     ("[04]", "search",    "Fell back to Tavily web search"),
            "detailed":   ("[05]", "detailed",  "Synthesized comprehensive deep-dive explanation (HITL)"),
            "generation": ("[06]", "generate",  "Generated final answer"),
        }

        for i, step in enumerate(last_msg["steps"]):
            key = next((k for k in step_meta if k in step.lower()), None)
            icon, cls, desc = step_meta[key] if key else (f"[{i+1:02d}]", "default", "Graph node executed")
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
            steps_html += f'<div style="color:#71717a;font-size:0.78rem;text-align:right;margin-top:4px">{last_msg["time"]}s</div>'

        st.markdown(steps_html, unsafe_allow_html=True)

    # Relevant Chunks Panel in Main View
    docs_to_show = last_msg.get("docs", [])
    if docs_to_show:
        with st.expander(f"Retrieved Context Chunks ({len(docs_to_show)})", expanded=False):
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

# Footer
st.markdown("""
<div class="footer">
    Powered by <span>LangGraph</span> · <span>Pinecone</span> · <span>Tavily</span> · <span>Streamlit</span>
</div>
""", unsafe_allow_html=True)