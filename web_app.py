"""
Numobel RAG Web App — Advanced UI
===================================
Streamlit-based chat interface with:
  • Multi-session sidebar (ChatGPT-style)
  • Conversation memory
  • Product image display
  • Global token tracking
  • Beautiful dark-themed UI
"""

import streamlit as st
import json
import os
import uuid
from datetime import datetime

import rag_chatbot as rag

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Numobel AI Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session Storage ─────────────────────────────────────────────────────────
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp', 'sessions')
os.makedirs(SESSIONS_DIR, exist_ok=True)

def _sessions_file():
    return os.path.join(SESSIONS_DIR, 'sessions.json')

def load_all_sessions() -> dict:
    """Load all sessions from disk."""
    path = _sessions_file()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_all_sessions(sessions: dict):
    """Persist all sessions to disk."""
    with open(_sessions_file(), 'w', encoding='utf-8') as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)

def create_new_session() -> str:
    """Create a new session and return its ID."""
    sid = str(uuid.uuid4())[:8]
    sessions = load_all_sessions()
    sessions[sid] = {
        'title': 'New Chat',
        'created': datetime.now().isoformat(),
        'messages': [],
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
    }
    save_all_sessions(sessions)
    return sid


# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0ff !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #b0b0cc !important;
    }

    /* ── Session buttons in sidebar ── */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        padding: 10px 16px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.04);
        color: #c8c8e0 !important;
        font-size: 14px;
        font-weight: 400;
        transition: all 0.2s ease;
        margin-bottom: 4px;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.3);
        color: #ffffff !important;
    }

    /* ── New Chat button ── */
    .new-chat-btn > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 12px 20px !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease !important;
    }

    .new-chat-btn > button:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-1px);
    }

    /* ── Chat messages ── */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
    }

    /* ── Header ── */
    .main-header {
        text-align: center;
        padding: 24px 0 16px 0;
    }

    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a78bfa, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .main-header p {
        font-size: 0.95rem;
        opacity: 0.6;
    }

    /* ── Token bar ── */
    .token-bar {
        display: flex;
        justify-content: center;
        gap: 24px;
        padding: 10px 20px;
        background: rgba(99, 102, 241, 0.08);
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        margin-top: 8px;
        font-size: 13px;
    }

    .token-bar span {
        font-weight: 500;
    }

    .token-label { opacity: 0.6; }
    .token-value { font-weight: 600; color: #a78bfa; }

    /* ── Image cards ── */
    .product-images-container {
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.06);
        margin: 12px 0;
    }

    .product-images-container img {
        border-radius: 12px;
        transition: transform 0.3s ease;
    }

    .product-images-container img:hover {
        transform: scale(1.03);
    }

    /* ── Active session highlight ── */
    .active-session > button {
        background: rgba(99, 102, 241, 0.2) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ── Delete button ── */
    .del-btn > button {
        background: transparent !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        color: #ef4444 !important;
        font-size: 12px !important;
        padding: 4px 10px !important;
        border-radius: 8px !important;
    }

    .del-btn > button:hover {
        background: rgba(239, 68, 68, 0.15) !important;
    }

    /* ── Divider ── */
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 16px 0;
    }

    /* ── Welcome card ── */
    .welcome-card {
        text-align: center;
        padding: 60px 40px;
        margin: 40px auto;
        max-width: 600px;
    }

    .welcome-card h2 {
        font-size: 1.5rem;
        margin-bottom: 12px;
    }

    .welcome-hint {
        display: inline-block;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 10px 18px;
        margin: 6px;
        font-size: 14px;
        color: #a78bfa;
        cursor: default;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize DB ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading Knowledge Base...")
def load_db():
    collection = rag.get_collection()
    count = collection.count()
    if count == 0:
        rag.ingest_data(collection)
    return collection

collection = load_db()


# ─── Session State Init ──────────────────────────────────────────────────────
if 'current_session' not in st.session_state:
    sessions = load_all_sessions()
    if sessions:
        # Pick most recent
        st.session_state.current_session = list(sessions.keys())[-1]
    else:
        st.session_state.current_session = create_new_session()

if 'total_prompt_tokens' not in st.session_state:
    st.session_state.total_prompt_tokens = 0
if 'total_completion_tokens' not in st.session_state:
    st.session_state.total_completion_tokens = 0


def get_current_messages() -> list:
    """Get messages for the current session from disk."""
    sessions = load_all_sessions()
    sid = st.session_state.current_session
    if sid in sessions:
        return sessions[sid].get('messages', [])
    return []

def save_current_messages(messages: list):
    """Save messages for the current session to disk."""
    sessions = load_all_sessions()
    sid = st.session_state.current_session
    if sid in sessions:
        sessions[sid]['messages'] = messages
        sessions[sid]['total_prompt_tokens'] = st.session_state.total_prompt_tokens
        sessions[sid]['total_completion_tokens'] = st.session_state.total_completion_tokens
        save_all_sessions(sessions)

def update_session_title(title: str):
    """Update the title of the current session."""
    sessions = load_all_sessions()
    sid = st.session_state.current_session
    if sid in sessions:
        # Truncate long titles
        sessions[sid]['title'] = title[:50] + ('...' if len(title) > 50 else '')
        save_all_sessions(sessions)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏭 Numobel AI")
    st.markdown(f"<p style='font-size:12px; opacity:0.5'>Models: {rag.EMBED_MODEL} + {rag.LLM_MODEL}</p>", unsafe_allow_html=True)

    # New Chat button
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("➕  New Chat", use_container_width=True):
        new_sid = create_new_session()
        st.session_state.current_session = new_sid
        st.session_state.total_prompt_tokens = 0
        st.session_state.total_completion_tokens = 0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown("##### 💬 Chat History")

    sessions = load_all_sessions()
    # Sort by created date descending
    sorted_sids = sorted(sessions.keys(), key=lambda s: sessions[s].get('created', ''), reverse=True)

    for sid in sorted_sids:
        session = sessions[sid]
        title = session.get('title', 'New Chat')
        is_active = (sid == st.session_state.current_session)

        col1, col2 = st.columns([5, 1])

        with col1:
            css_class = "active-session" if is_active else ""
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            if st.button(f"{'🔵 ' if is_active else '💬 '}{title}", key=f"sel_{sid}", use_container_width=True):
                st.session_state.current_session = sid
                # Restore token counts
                st.session_state.total_prompt_tokens = session.get('total_prompt_tokens', 0)
                st.session_state.total_completion_tokens = session.get('total_completion_tokens', 0)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="del-btn">', unsafe_allow_html=True)
            if st.button("🗑", key=f"del_{sid}"):
                del sessions[sid]
                save_all_sessions(sessions)
                if sid == st.session_state.current_session:
                    remaining = load_all_sessions()
                    if remaining:
                        st.session_state.current_session = list(remaining.keys())[-1]
                    else:
                        st.session_state.current_session = create_new_session()
                    st.session_state.total_prompt_tokens = 0
                    st.session_state.total_completion_tokens = 0
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 12px; opacity: 0.4; padding: 8px;">
        Memory: {rag.MEMORY_LIMIT} turns<br>
        Query Rewrite: {'On' if rag.REWRITE_QUERY else 'Off'}<br>
        Images: {'Native' if rag.USE_NATIVE_IMAGES else 'Collage'}<br>
        DB Chunks: {collection.count()}
    </div>
    """, unsafe_allow_html=True)


# ─── Main Chat Area ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏭 Numobel Product Assistant</h1>
    <p>Ask me anything about our products — wood finishes, toys, acoustics, furniture & more</p>
</div>
""", unsafe_allow_html=True)

# Load messages for current session
messages = get_current_messages()

# Show welcome card if no messages
if not messages:
    st.markdown("""
    <div class="welcome-card">
        <h2>👋 How can I help you today?</h2>
        <p style="opacity: 0.5; margin-bottom: 20px;">Try one of these example queries:</p>
        <div>
            <span class="welcome-hint">What Rubio Monocoat products do you have?</span>
            <span class="welcome-hint">Show me acoustic panels</span>
            <span class="welcome-hint">Compare wood cleaners</span>
            <span class="welcome-hint">What toys are available under ₹1000?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Image Rendering ─────────────────────────────────────────────────────────
def render_images(image_urls: list[str]):
    """Display product images using native Streamlit layout or Pillow collage."""
    if not image_urls:
        return

    if rag.USE_NATIVE_IMAGES:
        # Native Streamlit grid
        st.markdown('<div class="product-images-container">', unsafe_allow_html=True)
        st.caption("📸 Related Product Images")
        cols_per_row = min(len(image_urls), 4)
        cols = st.columns(cols_per_row)
        for idx, url in enumerate(image_urls[:8]):   # max 8 images
            with cols[idx % cols_per_row]:
                try:
                    st.image(url, use_container_width=True)
                except Exception:
                    pass
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Pillow collage (fallback)
        try:
            from PIL import Image
            import requests
            from io import BytesIO

            pil_images = []
            for url in image_urls[:6]:
                try:
                    resp = requests.get(url, timeout=5)
                    img = Image.open(BytesIO(resp.content)).convert('RGB')
                    img.thumbnail((300, 300))
                    pil_images.append(img)
                except Exception:
                    continue

            if pil_images:
                cols_count = min(len(pil_images), 3)
                rows_count = (len(pil_images) + cols_count - 1) // cols_count
                cell_w, cell_h = 300, 300
                collage = Image.new('RGB', (cols_count * cell_w, rows_count * cell_h), (20, 20, 30))

                for i, img in enumerate(pil_images):
                    r, c = divmod(i, cols_count)
                    # Center the image in its cell
                    x_offset = c * cell_w + (cell_w - img.width) // 2
                    y_offset = r * cell_h + (cell_h - img.height) // 2
                    collage.paste(img, (x_offset, y_offset))

                st.image(collage, caption="📸 Product Images", use_container_width=True)
        except ImportError:
            st.warning("Install Pillow (`pip install Pillow`) for collage mode.")


# ── Render Chat History ──
for msg in messages:
    with st.chat_message(msg['role'], avatar="🧑‍💻" if msg['role'] == 'user' else "🤖"):
        st.markdown(msg['content'])

        # Show images if stored in the message
        if msg['role'] == 'assistant' and msg.get('images'):
            render_images(msg['images'])


# ─── Chat Input ──────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about Numobel products..."):
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    rag.log.debug("WEBUI_INPUT | session=%s | %s", st.session_state.current_session, prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            # 1. Rewrite query if needed
            search_query = rag.rewrite_query(prompt, messages)

            # 2. Retrieve context
            hits = rag.retrieve(collection, search_query)

            for i, h in enumerate(hits, 1):
                rag.log.debug("WEB_RETRIEVED_%d | dist=%.4f | %s",
                              i, h['distance'], h['metadata'].get('product_name', '?'))

            # 3. Generate answer with token counts
            if not hits:
                result = {
                    'content': "I don't have that information.",
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                }
            else:
                result = rag.generate_answer(prompt, hits, messages)

            answer = result['content']
            rag.log.debug("WEBUI_OUTPUT | %s", answer.replace('\n', '\\n'))

        # Display the answer
        st.markdown(answer)

        # Collect and display images
        image_urls = rag.collect_images_from_hits(hits)
        if image_urls:
            render_images(image_urls)

    # Update token counts
    st.session_state.total_prompt_tokens += result['prompt_tokens']
    st.session_state.total_completion_tokens += result['completion_tokens']

    # Store messages
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": answer, "images": image_urls})
    save_current_messages(messages)

    # Auto-title from first user message
    sessions = load_all_sessions()
    sid = st.session_state.current_session
    if sid in sessions and sessions[sid].get('title') == 'New Chat':
        update_session_title(prompt)


# ─── Token Counter Bar ───────────────────────────────────────────────────────
total_tokens = st.session_state.total_prompt_tokens + st.session_state.total_completion_tokens
st.markdown(f"""
<div class="token-bar">
    <div>
        <span class="token-label">Prompt</span>&nbsp;
        <span class="token-value">{st.session_state.total_prompt_tokens:,}</span>
    </div>
    <div>
        <span class="token-label">Completion</span>&nbsp;
        <span class="token-value">{st.session_state.total_completion_tokens:,}</span>
    </div>
    <div>
        <span class="token-label">Total</span>&nbsp;
        <span class="token-value">{total_tokens:,}</span>
    </div>
</div>
""", unsafe_allow_html=True)
