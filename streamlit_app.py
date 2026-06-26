import streamlit as st
import requests
import uuid
import json
import time
import os
import re

from app.sse import iter_sse_events

# Safely fetch from Streamlit Secrets or fallback to local URL
try:
    API_URL_BASE = st.secrets.get("API_URL_BASE", os.environ.get("API_URL_BASE", "http://127.0.0.1:8000")).rstrip("/")
except Exception:
    API_URL_BASE = os.environ.get("API_URL_BASE", "http://127.0.0.1:8000").rstrip("/")

# ─── API HELPERS ──────────────────────────────────────────────────────────────
def _api_get(path: str, **params) -> dict | list | None:
    """GET to the FastAPI backend; displays an error in the sidebar on failure."""
    try:
        url = f"{API_URL_BASE}{path}"
        # Increased timeout to 60s to account for Hugging Face Spaces cold starts (waking up from sleep)
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.sidebar.error(f"❌ Backend Connection Error!\n\nTried to reach: `{API_URL_BASE}`\n\nError: {e}")
        return None


def fetch_sessions() -> list[dict]:
    """Return list of recent chat sessions from the API."""
    result = _api_get("/chat/sessions", limit=30)
    return result if isinstance(result, list) else []


def fetch_history(session_id: str) -> list[dict]:
    """Return message history for a session from the API."""
    result = _api_get(f"/chat/history/{session_id}", limit=50)
    return result if isinstance(result, list) else []


def delete_session(session_id: str) -> bool:
    """Delete all messages in a session."""
    try:
        resp = requests.delete(f"{API_URL_BASE}/chat/session/{session_id}", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Omni-Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DARK THEME CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }

    /* Main Backgrounds */
    .stApp { background-color: #0c0c0f; color: #f0f0f5; }
    
    /* Prevent Streamlit from dimming the screen while thinking */
    [data-stale="true"], [data-testid="stApp"] [data-stale="true"] {
        opacity: 1 !important;
        transition: none !important;
        filter: none !important;
        pointer-events: auto !important;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0d 0%, #0d0d14 100%) !important;
        border-right: 1px solid #1a1a28;
    }
    [data-testid="stSidebarNav"] { display: none; }
    
    /* Input Area styling */
    .stChatInput { background-color: #16161e !important; border-color: #2a2a3d !important; }
    .stChatInput textarea { color: #f0f0f5 !important; font-family: 'Inter', sans-serif !important; }
    .stChatInput textarea::placeholder { color: #5a5a7a !important; }
    
    /* Chat message styling */
    [data-testid="chatAvatarIcon-user"] { background-color: #7c3aed !important; }
    [data-testid="chatAvatarIcon-assistant"] { background: linear-gradient(135deg, #06b6d4, #7c3aed) !important; }
    .stChatMessage { background-color: transparent !important; }
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, rgba(124,58,237,0.15) 0%, rgba(30,50,120,0.2) 100%) !important;
        border: 1px solid rgba(124,58,237,0.2) !important;
        border-radius: 16px !important;
        padding: 12px 18px !important;
        margin-bottom: 12px !important;
    }
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #13131c !important;
        border: 1px solid #1e1e30 !important;
        border-radius: 16px !important;
        padding: 12px 18px !important;
        margin-bottom: 12px !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(6,182,212,0.08)) !important;
        border: 1px solid rgba(124,58,237,0.35) !important;
        color: #e0e0f0 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
        font-size: 0.85em !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(6,182,212,0.2)) !important;
        border-color: #7c3aed !important;
        box-shadow: 0 0 20px rgba(124,58,237,0.25) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Status container */
    [data-testid="stStatusWidget"] {
        background-color: #0f0f1a !important;
        border: 1px solid #1e1e30 !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton>button {
        text-align: left !important;
        font-size: 0.82em !important;
        padding: 6px 10px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    /* Text input styling */
    .stTextInput input {
        background-color: #13131c !important;
        border: 1px solid #22223a !important;
        color: #f0f0f5 !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
    }
    
    /* Password / API key input */
    .stTextInput input[type="password"] {
        background-color: #13131c !important;
        border: 1px solid #22223a !important;
        color: #f0f0f5 !important;
        border-radius: 8px !important;
        letter-spacing: 2px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #0f0f1a !important;
        border: 1px solid #1a1a2e !important;
        border-radius: 8px !important;
        color: #a78bfa !important;
    }
    .streamlit-expanderContent {
        background-color: #0c0c14 !important;
        border: 1px solid #1a1a2e !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #0f0f1a !important;
        border: 1px dashed #22223a !important;
        border-radius: 10px !important;
    }
    
    /* Toggle / checkbox */
    .stCheckbox label { color: #a0a0c0 !important; font-size: 0.85em !important; }
    
    /* Select box */
    .stSelectbox > div > div {
        background-color: #13131c !important;
        border: 1px solid #22223a !important;
        color: #f0f0f5 !important;
        border-radius: 8px !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0c0c0f; }
    ::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }
    
    /* Divider */
    hr { border-color: #1a1a28 !important; margin: 10px 0 !important; }

    /* Main title gradient */
    .main-title {
        background: linear-gradient(135deg, #a78bfa, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2.4rem;
        text-align: center;
        padding: 10px 0 5px 0;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        color: #5a5a7a;
        text-align: center;
        font-size: 0.9rem;
        margin-bottom: 30px;
    }

    /* Session history item */
    .session-item {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 8px;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.15s ease;
        border: 1px solid transparent;
    }
    .session-item:hover {
        background: rgba(124,58,237,0.1);
        border-color: rgba(124,58,237,0.2);
    }
    .session-item.active {
        background: rgba(124,58,237,0.18);
        border-color: rgba(124,58,237,0.4);
    }

    /* API key status badges */
    .api-badge-own {
        display: inline-block;
        background: rgba(52,211,153,0.15);
        color: #34d399;
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.72em;
        font-weight: 600;
    }
    .api-badge-builtin {
        display: inline-block;
        background: rgba(167,139,250,0.12);
        color: #a78bfa;
        border: 1px solid rgba(167,139,250,0.25);
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.72em;
        font-weight: 600;
    }
    /* Info box */
    .info-box {
        background: rgba(56,189,248,0.07);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.78em;
        color: #7dd3fc;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ─── TYPEWRITER STREAMING ─────────────────────────────────────────────────────
def stream_text(text: str):
    """Yield word-level chunks without artificial delay."""
    chunks = re.split(r'(\s+)', text)
    for chunk in chunks:
        yield chunk


# ─── INIT STATE ───────────────────────────────────────────────────────────────
defaults = {
    "session_id": None,
    "messages": [],
    "sidebar_sessions": [],
    "uploaded_file_ids": [],
    "active_url": "",
    "user_groq_key": "",
    "sessions_loaded": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Always refresh sidebar sessions
st.session_state.sidebar_sessions = fetch_sessions()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Logo / Title ────────────────────────    # ── Groq API Key Section ────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.72em; color:#4a4a6a; text-transform:uppercase; "
        "letter-spacing:1px; margin-bottom:4px;'>🔑 Groq API Key</p>",
        unsafe_allow_html=True,
    )
    # Always show the input — empty = use built-in key, filled = use user's key
    raw_key = st.text_input(
        "Your Groq API key (optional)",
        value=st.session_state.user_groq_key,
        placeholder="Leave blank to use built-in key  •  gsk_...",
        type="password",
        label_visibility="collapsed",
        key="groq_key_input",
        help="Optional: paste your own Groq key. If left blank, the built-in key is used automatically.",
    )
    stripped_key = raw_key.strip()
    st.session_state.user_groq_key = stripped_key

    if stripped_key:
        # User has entered their own key
        if stripped_key.startswith("gsk_") and len(stripped_key) > 20:
            st.markdown(
                "<div class='api-badge-own'>✅ Your key is active</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='color:#f87171; font-size:0.75em; margin-top:4px;'>"
                "⚠️ Groq keys start with <code>gsk_</code> — check your key</div>",
                unsafe_allow_html=True,
            )
    else:
        # No user key — built-in key will be used automatically
        st.session_state.user_groq_key = ""
        st.markdown(
            "<div class='api-badge-builtin'>⚡ Built-in key active</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='info-box' style='margin-top:6px;'>Built-in key is always available. "
            "Enter your own <a href='https://console.groq.com/keys' target='_blank' "
            "style='color:#38bdf8;'>Groq key</a> above if you hit rate limits.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── RAG Inputs ────────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.72em; color:#4a4a6a; text-transform:uppercase; "
        "letter-spacing:1px; margin-bottom:6px;'>📎 Knowledge Upload (RAG)</p>",
        unsafe_allow_html=True,
    )
    uploaded_pdf = st.file_uploader(
        "Upload PDF", type=["pdf"], label_visibility="collapsed", key="pdf_uploader"
    )

    if uploaded_pdf and f"uploaded_{uploaded_pdf.name}" not in st.session_state:
        with st.spinner("Uploading & indexing PDF..."):
            try:
                res = requests.post(
                    f"{API_URL_BASE}/upload/pdf",
                    files={"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")},
                )
                res.raise_for_status()
                data = res.json()
                st.session_state.uploaded_file_ids.append(data["file_id"])
                st.session_state[f"uploaded_{uploaded_pdf.name}"] = True
                st.success(f"✅ Indexed: {uploaded_pdf.name}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    if st.session_state.uploaded_file_ids:
        st.markdown(
            f"<p style='font-size:0.75em; color:#34d399;'>"
            f"📄 {len(st.session_state.uploaded_file_ids)} file(s) ready for RAG</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<p style='font-size:0.72em; color:#4a4a6a; text-transform:uppercase; "
        "letter-spacing:1px; margin-top:10px; margin-bottom:4px;'>🔗 URL Context</p>",
        unsafe_allow_html=True,
    )
    url_input = st.text_input(
        "Active URL",
        placeholder="https://example.com...",
        label_visibility="collapsed",
        value=st.session_state.active_url,
        key="url_input",
    )
    if url_input != st.session_state.active_url:
        st.session_state.active_url = url_input

    st.markdown("---")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    # ── Recent Chats ──────────────────────────────────────────────────────────
    sessions = st.session_state.sidebar_sessions
    session_count = len(sessions)

    st.markdown(
        f"<p style='font-size:0.72em; color:#4a4a6a; text-transform:uppercase; "
        f"letter-spacing:1px; margin-bottom:6px;'>💬 Recent Chats "
        f"<span style=\"color:#3a3a5a;\">({session_count})</span></p>",
        unsafe_allow_html=True,
    )

    if not sessions:
        st.markdown(
            "<p style='font-size:0.78em; color:#2a2a4a; font-style:italic; padding-left:4px;'>"
            "No previous chats yet. Start chatting!</p>",
            unsafe_allow_html=True,
        )
    else:
        for s in sessions:
            sid   = s["id"]   if isinstance(s, dict) else s.id
            title = s["title"] if isinstance(s, dict) else (s.title or "New Chat")
            updated = s.get("updated_at", "") if isinstance(s, dict) else ""

            # Format relative timestamp
            time_label = ""
            if updated:
                try:
                    from datetime import datetime, timezone
                    dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    diff_mins = int((now - dt).total_seconds() / 60)
                    if diff_mins < 1:
                        time_label = "just now"
                    elif diff_mins < 60:
                        time_label = f"{diff_mins}m ago"
                    elif diff_mins < 1440:
                        time_label = f"{diff_mins // 60}h ago"
                    else:
                        time_label = f"{diff_mins // 1440}d ago"
                except Exception:
                    time_label = ""

            display = (title[:26] + "…") if len(title) > 28 else title
            is_active = st.session_state.session_id == sid

            # Show each session as a row: button + (optional) delete
            col_btn, col_del = st.columns([5, 1], gap="small")
            with col_btn:
                btn_style = "primary" if is_active else "secondary"
                if st.button(
                    f"{'▶ ' if is_active else '💬 '}{display}",
                    key=f"sess_{sid}",
                    use_container_width=True,
                    help=f"{title}\n{time_label}" if time_label else title,
                ):
                    st.session_state.session_id = sid
                    history = fetch_history(sid)
                    st.session_state.messages = [
                        {
                            "role": m["role"] if isinstance(m, dict) else m.role,
                            "content": m["content"] if isinstance(m, dict) else m.content,
                        }
                        for m in history
                    ]
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{sid}", help="Delete this chat"):
                    if delete_session(sid):
                        if st.session_state.session_id == sid:
                            st.session_state.session_id = None
                            st.session_state.messages = []
                        st.session_state.sidebar_sessions = fetch_sessions()
                        st.rerun()

            # Show timestamp below each item
            if time_label:
                st.markdown(
                    f"<div style='font-size:0.65em; color:#2a2a5a; "
                    f"margin:-6px 0 4px 4px;'>{time_label}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Status indicator ──────────────────────────────────────────────────────
    key_status = (
        "<span style='color:#34d399;'>🔑 Your Key</span>"
        if st.session_state.user_groq_key
        else "<span style='color:#a78bfa;'>⚡ Built-in Key</span>"
    )
    st.markdown(
        f"<div style='font-size:0.75em; color:#3a3a5a; padding: 4px 0;'>"
        f"<span style='color:#34d399;'>●</span> Agent Online &nbsp;|&nbsp; {key_status}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─── MAIN CHAT AREA ───────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🤖 Universal Omni-Agent</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='main-subtitle'>Intelligent routing · Web Search · RAG Documents · Finance · Cricket Scores · General AI</div>",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; margin-bottom: 40px;">
        <div style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin-top: 10px;">
            <span style="background: rgba(167, 139, 250, 0.12); color: #a78bfa; padding: 6px 14px; border-radius: 20px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(167,139,250,0.2);">💬 GENERAL</span>
            <span style="background: rgba(52, 211, 153, 0.12); color: #34d399; padding: 6px 14px; border-radius: 20px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(52,211,153,0.2);">📚 RAG DOCS</span>
            <span style="background: rgba(56, 189, 248, 0.12); color: #38bdf8; padding: 6px 14px; border-radius: 20px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(56,189,248,0.2);">🌐 WEB SEARCH</span>
            <span style="background: rgba(251, 191, 36, 0.12); color: #fbbf24; padding: 6px 14px; border-radius: 20px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(251,191,36,0.2);">📈 FINANCE</span>
            <span style="background: rgba(248, 113, 113, 0.12); color: #f87171; padding: 6px 14px; border-radius: 20px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(248,113,113,0.2);">☀️ WEATHER</span>
            <span style="background: rgba(52, 211, 153, 0.12); color: #34d399; padding: 6px 14px; border-radius: 20px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(52,211,153,0.2);">🏏 CRICKET</span>
        </div>
        <p style="color: #3a3a5a; font-size: 0.85em; margin-top: 15px;">Ask me anything — I'll intelligently route to the right specialist agent.</p>
        <p style="color: #2a2a4a; font-size: 0.8em; margin-top: 5px;">Try: <em>"current score"</em> · <em>"live cricket score"</em> · <em>"AAPL stock price"</em></p>
    </div>
    """, unsafe_allow_html=True)

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ─── CHAT INPUT ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything — weather, news, cricket score, stocks, documents..."):
    if not st.session_state.session_id:
        st.session_state.session_id = str(uuid.uuid4())

    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine which API key to use
    active_groq_key = st.session_state.user_groq_key

    # Call FastAPI SSE stream endpoint
    with st.chat_message("assistant"):
        status = st.status("🤖 Agent Thinking...", expanded=True)

        try:
            params = {
                "session_id":    st.session_state.session_id,
                "message":       prompt,
                "file_ids":      ",".join(st.session_state.uploaded_file_ids),
                "active_url":    st.session_state.active_url or "",
                "user_groq_key": active_groq_key,
            }

            response = requests.get(
                f"{API_URL_BASE}/chat/stream",
                params=params,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            final_result = None
            stream_error = None

            for event in iter_sse_events(response.iter_lines(decode_unicode=True)):
                event_type = event["event"]
                event_data = event["data"]
                if event_type == "log":
                    data = json.loads(event_data)
                    node = data.get("node", "system").upper()
                    msg_text = data.get("message", "")

                    if "🔀 Routing" in msg_text:
                        status.write(f"🔀 **Routing Decision**: {msg_text.replace('🔀 Routing to: ', '')}")
                    elif "⚙️ Using:" in msg_text:
                        status.write(f"🛠️ **Tool**: {msg_text.replace('⚙️ Using: ', '')}")
                    elif "cricket" in msg_text.lower() or "score" in msg_text.lower():
                        status.write(f"🏏 **[{node}]** {msg_text}")
                    elif "weather" in msg_text.lower():
                        status.write(f"☀️ **[{node}]** {msg_text}")
                    elif "search" in msg_text.lower() or "web" in msg_text.lower():
                        status.write(f"🔍 **[{node}]** {msg_text}")
                    elif "qdrant" in msg_text.lower() or "retriev" in msg_text.lower():
                        status.write(f"📚 **[{node}]** {msg_text}")
                    else:
                        status.write(f"⚡ **[{node}]** {msg_text}")

                elif event_type == "result":
                    final_result = json.loads(event_data)
                    route = final_result.get("route", "general")
                    route_labels = {
                        "web":     "🌐 Web Search Complete",
                        "rag":     "📚 Document Analysis Complete",
                        "finance": "📈 Finance / Cricket Data Retrieved",
                    }
                    label = route_labels.get(route, "✅ Response Ready")
                    status.update(label=label, state="complete", expanded=False)

                elif event_type == "error":
                    err_data = json.loads(event_data) if event_data.startswith("{") else {"error": event_data}
                    stream_error = err_data.get("error", event_data)
                    status.update(label="❌ Error occurred", state="error", expanded=True)
                    status.write(f"Error: {stream_error}")

            # Stream final response
            if final_result:
                bot_reply  = final_result.get("answer", "")
                route_used = final_result.get("route", "general")
                sources    = final_result.get("sources", [])

                tags = {
                    "finance":   ("📈 FINANCE / CRICKET", "#fbbf24"),
                    "web":       ("🌐 WEB SEARCH",        "#38bdf8"),
                    "rag":       ("📚 DOCS",              "#34d399"),
                    "general":   ("💬 GENERAL",           "#a78bfa"),
                    "instagram": ("📸 INSTAGRAM",         "#e1306c"),
                }
                tag_label, tag_color = tags.get(route_used, ("💬 GENERAL", "#a78bfa"))

                route_html = (
                    f"<span style='font-size:0.7em; background: rgba(255,255,255,0.07); "
                    f"color: {tag_color}; padding: 3px 10px; border-radius: 4px; "
                    f"font-weight: 700; margin-bottom: 12px; display: inline-block; "
                    f"border: 1px solid {tag_color}33;'>{tag_label}</span>\n\n"
                )

                st.markdown(route_html, unsafe_allow_html=True)
                st.write_stream(stream_text(bot_reply))

                if sources:
                    valid_sources = [s for s in sources if s.get("type") in ("web", "arxiv") and s.get("url")]
                    if valid_sources:
                        # ── Show article images (news results from NewsAPI) ────────
                        news_with_images = [
                            s for s in valid_sources
                            if s.get("image_url") and s.get("type") == "web"
                        ]
                        if news_with_images:
                            st.markdown("---")
                            cols = st.columns(min(len(news_with_images), 3))
                            for col, s in zip(cols, news_with_images[:3]):
                                with col:
                                    try:
                                        st.image(s["image_url"], use_container_width=True)
                                    except Exception:
                                        pass  # Skip broken images silently
                                    st.markdown(
                                        f"<small>**{s.get('title', '')[:60]}**<br>"
                                        f"[Read →]({s['url']})</small>",
                                        unsafe_allow_html=True,
                                    )

                        # ── Source link row ────────────────────────────────────────
                        source_lines = []
                        for i, s in enumerate(valid_sources[:5], 1):
                            title = s.get("title", f"Source {i}")[:60]
                            url   = s.get("url", "")
                            icon  = "📄" if s.get("type") == "arxiv" else "🔗"
                            if url:
                                source_lines.append(f"{icon} [{i}. {title}]({url})")
                        if source_lines and not news_with_images:
                            # Only show plain text source list if no images were shown
                            st.markdown("\n\n---\n**Sources:** " + " · ".join(source_lines))

                full_content = route_html + bot_reply
                st.session_state.messages.append({"role": "assistant", "content": full_content})
            else:
                message = stream_error or "The backend stream ended without returning an answer."
                status.update(label="❌ No response returned", state="error", expanded=True)
                st.error(message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ {message}",
                })

        except requests.exceptions.Timeout:
            status.update(label="⏱️ Request Timed Out", state="error", expanded=True)
            status.write("The request took too long. Please try again.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⏱️ Request timed out. Please try again.",
            })
        except Exception as e:
            status.update(label="❌ Connection Error", state="error", expanded=True)
            status.error(f"Backend error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}",
            })

    # Refresh sidebar
    st.session_state.sidebar_sessions = fetch_sessions()
    st.rerun()
