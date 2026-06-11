import streamlit as st
import requests
import uuid
import asyncio
import json
import time
import re

from app.sse import iter_sse_events

# Try to directly access the database for history if running in same folder
try:
    from app.database.sql_db import Session, Message, engine
    from sqlalchemy import select, desc
    from sqlalchemy.ext.asyncio import AsyncSession
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

API_URL_BASE = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Universal Omni-Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }

    /* Main Backgrounds */
    .stApp { background-color: #0c0c0f; color: #f0f0f5; }
    
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
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #0f0f1a !important;
        border: 1px dashed #22223a !important;
        border-radius: 10px !important;
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
</style>
""", unsafe_allow_html=True)


# ─── DATABASE HELPERS ───
async def fetch_sessions():
    if not DB_AVAILABLE: return []
    try:
        async with AsyncSession(engine) as session:
            result = await session.execute(
                select(Session).order_by(desc(Session.updated_at)).limit(20)
            )
            return result.scalars().all()
    except Exception:
        return []

async def fetch_history(session_id: str):
    if not DB_AVAILABLE: return []
    try:
        async with AsyncSession(engine) as session:
            result = await session.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.asc())
            )
            return result.scalars().all()
    except Exception:
        return []


# ─── SIMULATED TYPEWRITER STREAMING ───
def stream_text(text: str, delay: float = 0.004):
    """Fast typewriter effect — yields word-level chunks."""
    chunks = re.split(r'(\s+)', text)
    for chunk in chunks:
        yield chunk
        if chunk.strip():  # Only sleep on non-whitespace
            time.sleep(delay)


# ─── INIT STATE ───
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sidebar_sessions" not in st.session_state:
    st.session_state.sidebar_sessions = []
if "uploaded_file_ids" not in st.session_state:
    st.session_state.uploaded_file_ids = []
if "active_url" not in st.session_state:
    st.session_state.active_url = ""

# Load sidebar sessions
if DB_AVAILABLE:
    try:
        st.session_state.sidebar_sessions = asyncio.run(fetch_sessions())
    except Exception:
        st.session_state.sidebar_sessions = []


# ─── SIDEBAR ───
with st.sidebar:
    # Logo / Title
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 5px 0;'>
        <div style='font-size:2rem;'>🤖</div>
        <div style='font-size:1.1rem; font-weight:700; color:#a78bfa; letter-spacing:-0.3px;'>Omni-Agent</div>
        <div style='font-size:0.72rem; color:#3a3a5a; margin-top:2px;'>Powered by Groq · Llama 3.3-70b</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # New Chat button
    if st.button("➕  New Chat", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.uploaded_file_ids = []
        st.session_state.active_url = ""
        st.rerun()
    
    st.markdown("---")
    
    # RAG Inputs
    st.markdown("<p style='font-size:0.72em; color:#4a4a6a; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>📎 Knowledge Upload (RAG)</p>", unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_pdf and f"uploaded_{uploaded_pdf.name}" not in st.session_state:
        with st.spinner("Uploading & indexing PDF..."):
            try:
                res = requests.post(
                    f"{API_URL_BASE}/upload/pdf", 
                    files={"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                )
                res.raise_for_status()
                data = res.json()
                st.session_state.uploaded_file_ids.append(data["file_id"])
                st.session_state[f"uploaded_{uploaded_pdf.name}"] = True
                st.success(f"✅ Indexed: {uploaded_pdf.name}")
            except Exception as e:
                st.error(f"Upload failed: {e}")
    
    # Show uploaded file names
    if st.session_state.uploaded_file_ids:
        st.markdown(f"<p style='font-size:0.75em; color:#34d399;'>📄 {len(st.session_state.uploaded_file_ids)} file(s) ready for RAG</p>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:0.72em; color:#4a4a6a; text-transform:uppercase; letter-spacing:1px; margin-top:10px; margin-bottom:4px;'>🔗 URL Context</p>", unsafe_allow_html=True)
    url_input = st.text_input("Active URL", placeholder="https://example.com...", label_visibility="collapsed", value=st.session_state.active_url)
    if url_input != st.session_state.active_url:
        st.session_state.active_url = url_input
    
    st.markdown("---")
    
    # Recent Chats
    st.markdown("<p style='font-size:0.72em; color:#4a4a6a; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>💬 Recent Chats</p>", unsafe_allow_html=True)
    
    if not st.session_state.sidebar_sessions:
        st.markdown("<p style='font-size:0.78em; color:#2a2a4a; font-style:italic; padding-left:4px;'>No previous chats yet.</p>", unsafe_allow_html=True)
    else:
        for s in st.session_state.sidebar_sessions:
            title = s.title if s.title else "New Chat"
            # Truncate long titles
            display = (title[:28] + "...") if len(title) > 31 else title
            is_active = st.session_state.session_id == s.id
            btn_style = "border-left: 2px solid #7c3aed !important;" if is_active else ""
            if st.button(f"💬 {display}", key=f"btn_{s.id}", use_container_width=True):
                st.session_state.session_id = s.id
                try:
                    history = asyncio.run(fetch_history(s.id))
                    st.session_state.messages = [{"role": m.role, "content": m.content} for m in history]
                except Exception:
                    st.session_state.messages = []
                st.rerun()
    
    st.markdown("---")
    # Status indicator
    st.markdown("""
    <div style='font-size:0.75em; color:#3a3a5a; padding: 4px 0;'>
        <span style='color:#34d399;'>●</span> Agent Online &nbsp;|&nbsp; 
        <span style='color:#a78bfa;'>⚡ Groq</span>
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN CHAT AREA ───
st.markdown("<div class='main-title'>🤖 Universal Omni-Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='main-subtitle'>Intelligent routing · Web Search · RAG Documents · Finance · General AI</div>", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; margin-bottom: 40px;">
        <div style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin-top: 10px;">
            <span style="background: rgba(167, 139, 250, 0.12); color: #a78bfa; padding: 6px 14px; border-radius: 20px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(167,139,250,0.2);">💬 GENERAL</span>
            <span style="background: rgba(52, 211, 153, 0.12); color: #34d399; padding: 6px 14px; border-radius: 20px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(52,211,153,0.2);">📚 RAG DOCS</span>
            <span style="background: rgba(56, 189, 248, 0.12); color: #38bdf8; padding: 6px 14px; border-radius: 20px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(56,189,248,0.2);">🌐 WEB SEARCH</span>
            <span style="background: rgba(251, 191, 36, 0.12); color: #fbbf24; padding: 6px 14px; border-radius: 20px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(251,191,36,0.2);">📈 FINANCE</span>
            <span style="background: rgba(248, 113, 113, 0.12); color: #f87171; padding: 6px 14px; border-radius: 20px; font-size: 0.8em; font-weight: 600; border: 1px solid rgba(248,113,113,0.2);">☀️ WEATHER</span>
        </div>
        <p style="color: #3a3a5a; font-size: 0.85em; margin-top: 15px;">Ask me anything — I'll intelligently route to the right specialist agent.</p>
    </div>
    """, unsafe_allow_html=True)

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ─── CHAT INPUT ───
if prompt := st.chat_input("Ask me anything — weather, news, documents, stocks..."):
    if not st.session_state.session_id:
        st.session_state.session_id = str(uuid.uuid4())
        
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Call FastAPI SSE stream endpoint
    with st.chat_message("assistant"):
        # Live status box with tool activity
        status = st.status("🤖 Agent Thinking...", expanded=True)
        
        try:
            params = {
                "session_id": st.session_state.session_id,
                "message": prompt,
                "file_ids": ",".join(st.session_state.uploaded_file_ids),
                "active_url": st.session_state.active_url or "",
            }
            
            response = requests.get(
                f"{API_URL_BASE}/chat/stream", 
                params=params, 
                stream=True,
                timeout=120,  # 2 minute timeout
            )
            response.raise_for_status()
            
            final_result = None
            stream_error = None
            
            # Consume SSE events without an optional third-party client.
            for event in iter_sse_events(response.iter_lines(decode_unicode=True)):
                event_type = event["event"]
                event_data = event["data"]
                if event_type == "log":
                    data = json.loads(event_data)
                    node = data.get("node", "system").upper()
                    msg_text = data.get("message", "")
                    
                    # Categorize log messages for better display
                    if "🔀 Routing" in msg_text:
                        status.write(f"🔀 **Routing Decision**: {msg_text.replace('🔀 Routing to: ', '')}")
                    elif "⚙️ Using:" in msg_text:
                        status.write(f"🛠️ **Tool**: {msg_text.replace('⚙️ Using: ', '')}")
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
                        "web": "🌐 Web Search Complete",
                        "rag": "📚 Document Analysis Complete",
                        "finance": "📈 Finance Data Retrieved",
                    }
                    label = route_labels.get(route, "✅ Response Ready")
                    status.update(label=label, state="complete", expanded=False)
                    
                elif event_type == "error":
                    err_data = json.loads(event_data) if event_data.startswith("{") else {"error": event_data}
                    stream_error = err_data.get("error", event_data)
                    status.update(label="❌ Error occurred", state="error", expanded=True)
                    status.write(f"Error: {stream_error}")
            
            # Stream the final response (OUTSIDE the for loop)
            if final_result:
                bot_reply = final_result.get("answer", "")
                route_used = final_result.get("route", "general")
                sources = final_result.get("sources", [])
                
                # Route badge
                tags = {
                    "finance": ("📈 FINANCE", "#fbbf24"),
                    "web":     ("🌐 WEB SEARCH", "#38bdf8"),
                    "rag":     ("📚 DOCS", "#34d399"),
                    "general": ("💬 GENERAL", "#a78bfa"),
                }
                tag_label, tag_color = tags.get(route_used, ("💬 GENERAL", "#a78bfa"))
                
                route_html = (
                    f"<span style='font-size:0.7em; background: rgba(255,255,255,0.07); "
                    f"color: {tag_color}; padding: 3px 10px; border-radius: 4px; "
                    f"font-weight: 700; margin-bottom: 12px; display: inline-block; "
                    f"border: 1px solid {tag_color}33;'>{tag_label}</span>\n\n"
                )
                
                # Show tag instantly
                st.markdown(route_html, unsafe_allow_html=True)
                
                # Stream the text
                st.write_stream(stream_text(bot_reply))
                
                # Show sources if any
                if sources:
                    valid_sources = [s for s in sources if s.get("type") in ("web", "arxiv") and s.get("url")]
                    if valid_sources:
                        source_lines = []
                        for i, s in enumerate(valid_sources[:5], 1):
                            title = s.get("title", f"Source {i}")[:60]
                            url = s.get("url", "")
                            icon = "📄" if s.get("type") == "arxiv" else "🔗"
                            if url:
                                source_lines.append(f"{icon} [{i}. {title}]({url})")
                        if source_lines:
                            sources_md = "\n\n---\n**Sources:** " + " · ".join(source_lines)
                            st.markdown(sources_md)
                
                # Save full message to session state
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
            st.session_state.messages.append({"role": "assistant", "content": "⏱️ Request timed out. Please try again."})
        except Exception as e:
            status.update(label="❌ Connection Error", state="error", expanded=True)
            status.error(f"Backend error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
    
    # Refresh sidebar sessions
    if DB_AVAILABLE:
        try:
            st.session_state.sidebar_sessions = asyncio.run(fetch_sessions())
        except Exception:
            pass
    
    st.rerun()
