import os
import streamlit as st
from rag import RAGPipeline

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="DocMind — RAG Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'Syne', sans-serif; }
code, .citation-block { font-family: 'DM Mono', monospace; }

/* Background */
.stApp {
    background: #0a0a0f;
    color: #e8e6f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111118 !important;
    border-right: 1px solid #2a2a3a;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* Title */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #6b6b8a;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Chat messages */
.msg-user {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 16px 16px 4px 16px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    margin-left: 20%;
    color: #c4b5fd;
    font-size: 0.95rem;
}
.msg-assistant {
    background: #111120;
    border: 1px solid #1e1e32;
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    margin-right: 20%;
    color: #e2e0f0;
    font-size: 0.95rem;
    line-height: 1.7;
}
.msg-label-user {
    font-size: 0.7rem;
    color: #6b6b8a;
    text-align: right;
    margin-bottom: 0.3rem;
    margin-right: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.msg-label-assistant {
    font-size: 0.7rem;
    color: #6b6b8a;
    margin-bottom: 0.3rem;
    margin-left: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Citation block */
.citation-block {
    background: #0d0d1a;
    border: 1px solid #1e1e3a;
    border-left: 3px solid #7c3aed;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 0.8rem;
    font-size: 0.78rem;
    color: #8b8baa;
    line-height: 1.8;
}
.citation-title {
    color: #7c3aed;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}

/* Status badges */
.badge-ready {
    display: inline-block;
    background: #052e16;
    color: #34d399;
    border: 1px solid #065f46;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-waiting {
    display: inline-block;
    background: #1c1917;
    color: #78716c;
    border: 1px solid #292524;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
}

/* Input styling */
.stTextInput > div > div > input {
    background: #111120 !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 12px !important;
    color: #e2e0f0 !important;
    font-family: 'Syne', sans-serif !important;
    padding: 0.8rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #111120;
    border: 1.5px dashed #2a2a4a;
    border-radius: 12px;
    padding: 1rem;
}

/* Divider */
hr { border-color: #1e1e32; }

/* Chunk count */
.chunk-info {
    color: #6b6b8a;
    font-size: 0.78rem;
    margin-top: 0.5rem;
}

/* Scrollable chat area */
.chat-container {
    max-height: 60vh;
    overflow-y: auto;
    padding-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ---------- SESSION STATE ----------
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("")
    st.markdown("---")

    st.markdown("**Upload PDF**")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if st.button(" Index Document", use_container_width=True):
            # Save uploaded file temporarily (Windows compatible)
            import tempfile
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())

            with st.spinner("Indexing document..."):
                rag = RAGPipeline(tmp_path)
                rag.setup()
                st.session_state.rag = rag
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []

            st.success("Document indexed!")

    st.markdown("---")

    # Status
    st.markdown("**Status**")
    if st.session_state.rag:
        st.markdown(f'<span class="badge-ready">● Ready</span>', unsafe_allow_html=True)
        st.markdown(f'<p class="chunk-info"> {st.session_state.pdf_name}<br>{len(st.session_state.rag.documents)} chunks indexed</p>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-waiting">○ No document loaded</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Clear chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown('<p style="color:#3a3a5a;font-size:0.7rem;">Built with ChromaDB · BAAI BGE · Groq LLaMA</p>', unsafe_allow_html=True)


# ---------- MAIN AREA ----------
st.markdown('<h1 class="hero-title">RAG-DOC</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Ask anything about your document — powered by RAG</p>', unsafe_allow_html=True)

# Chat history display
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<p class="msg-label-user">You</p><div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # Split answer and citations
            if "\n\n Citations:" in msg["content"]:
                answer_part, citation_part = msg["content"].split("\n\n Citations:", 1)
                citation_lines = citation_part.strip().replace("\n", "<br>")
                st.markdown(
                    f'<p class="msg-label-assistant">DocMind</p>'
                    f'<div class="msg-assistant">{answer_part}'
                    f'<div class="citation-block"><div class="citation-title"> Citations</div>{citation_lines}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f'<p class="msg-label-assistant">DocMind</p><div class="msg-assistant">{msg["content"]}</div>', unsafe_allow_html=True)
else:
    if st.session_state.rag:
        st.markdown('<p style="color:#3a3a5a;text-align:center;margin-top:3rem;font-size:0.9rem;">Document indexed ✓ — Ask your first question below</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#3a3a5a;text-align:center;margin-top:3rem;font-size:0.9rem;">← Upload a PDF from the sidebar to get started</p>', unsafe_allow_html=True)

st.markdown("---")

# Input area
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Ask a question",
        placeholder="What does this document say about...?",
        label_visibility="collapsed",
        key="query_input"
    )
with col2:
    send = st.button("Send →", use_container_width=True)

# Handle send
if send and query:
    if not st.session_state.rag:
        st.warning("Please upload and index a PDF first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            answer = st.session_state.rag.ask(query)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()