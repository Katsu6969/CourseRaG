"""Streamlit web interface for CourseRAG (Warm Paper / Beige & Espresso theme)."""

from __future__ import annotations

import streamlit as st

from src.config import RAGSettings
from src.pipeline import CourseRAGPipeline

st.set_page_config(
    page_title="CourseRAG — Technical Study Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F6F1E7;
        color: #2C221E;
        font-family: "Georgia", "Segoe UI", system-ui, sans-serif;
    }
    [data-testid="stHeader"] {
        background: rgba(246, 241, 231, 0.88);
    }
    [data-testid="stSidebar"] {
        background-color: #ECE3D3;
        border-right: 1px solid #D6C8B2;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #2C221E !important;
        font-family: "Segoe UI", system-ui, sans-serif;
    }
    [data-testid="stChatMessage"] table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.89rem;
    }
    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessage"] td {
        border: 1px solid #D6C8B2;
        padding: 6px 9px;
    }
    [data-testid="stChatMessage"] th {
        background-color: #ECE3D3;
    }
    .banner-box {
        background: linear-gradient(135deg, #FAF7F0 0%, #EFE6D5 100%);
        border: 1px solid #D6C8B2;
        border-left: 5px solid #A65A38;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        box-shadow: 0 4px 16px rgba(44, 34, 30, 0.05);
    }
    .banner-box h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        color: #2C221E;
        font-family: "Georgia", serif;
        letter-spacing: -0.02em;
    }
    .banner-box p {
        margin: 5px 0 0;
        color: #5E4F43;
        font-size: 0.96rem;
        font-family: "Segoe UI", system-ui, sans-serif;
    }
    .badge-group {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 6px;
        font-family: "Segoe UI", system-ui, sans-serif;
    }
    .mode-pill {
        background-color: #E7EFE4;
        color: #354C34;
        border: 1px solid #B8CCB3;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .stats-pill {
        background-color: #EFE6D5;
        color: #5E4F43;
        border: 1px solid #D6C8B2;
        padding: 4px 11px;
        border-radius: 999px;
        font-size: 0.77rem;
        white-space: nowrap;
    }
    .intro-panel {
        background-color: #FAF7F0;
        border: 1px solid #DDD0BC;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 16px;
        font-family: "Segoe UI", system-ui, sans-serif;
    }
    .intro-panel h3 {
        margin: 0 0 8px;
        font-size: 1.12rem;
        color: #2C221E;
    }
    .intro-panel p {
        margin: 0;
        color: #5E4F43;
        font-size: 0.94rem;
        line-height: 1.6;
    }
    [data-testid="stChatMessage"] {
        background-color: #FAF7F0;
        border: 1px solid #DFD3C0;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(44, 34, 30, 0.03);
        font-family: "Segoe UI", system-ui, sans-serif;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] td,
    [data-testid="stChatMessage"] th {
        color: #2C221E !important;
        line-height: 1.68;
        font-size: 0.96rem;
    }
    [data-testid="stChatMessage"] h3 {
        color: #8C4627 !important;
        font-size: 1.15rem;
        margin-top: 2px;
    }
    [data-testid="stChatMessage"] pre {
        background-color: #EFE8DA !important;
        border: 1px solid #D6C8B2 !important;
        border-left: 4px solid #A65A38 !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatMessage"] code {
        color: #2C221E !important;
        font-family: Consolas, "Cascadia Mono", monospace !important;
    }
    .doc-item {
        background-color: #FAF7F0;
        border: 1px solid #D6C8B2;
        border-radius: 9px;
        padding: 8px 11px;
        margin-bottom: 7px;
        font-size: 0.83rem;
        font-family: "Segoe UI", system-ui, sans-serif;
    }
    .doc-item strong {
        color: #2C221E;
        display: block;
        word-break: break-word;
    }
    .doc-item span {
        color: #6B5B4E !important;
        font-size: 0.77rem;
    }
    .chunk-card {
        background-color: #F6F1E7;
        border: 1px solid #D6C8B2;
        border-radius: 9px;
        padding: 10px 13px;
        margin-bottom: 9px;
        font-size: 0.86rem;
    }
    .chunk-meta {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 6px;
        font-weight: 600;
        color: #8C4627;
        margin-bottom: 5px;
        font-size: 0.81rem;
    }
    .telemetry-line {
        color: #6B5B4E;
        font-size: 0.80rem;
        margin-top: 6px;
        margin-bottom: 4px;
        font-family: "Segoe UI", system-ui, sans-serif;
    }
    div.stButton > button {
        background-color: #FAF7F0;
        color: #2C221E;
        border: 1px solid #CDBCA2;
        border-radius: 9px;
        text-align: left;
        font-size: 0.86rem;
        padding: 7px 12px;
        transition: all 0.15s ease;
    }
    div.stButton > button:hover {
        background-color: #EFE4D2;
        border-color: #A65A38;
        color: #2C221E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> CourseRAGPipeline:
    settings = RAGSettings()
    # Support Streamlit Community Cloud secrets alongside local .env files
    try:
        secrets = st.secrets
        for attr, key in (
            ("llm_provider", "LLM_PROVIDER"),
            ("groq_api_key", "GROQ_API_KEY"),
            ("groq_model", "GROQ_MODEL"),
            ("gemini_api_key", "GEMINI_API_KEY"),
            ("gemini_model", "GEMINI_MODEL"),
            ("openai_api_key", "OPENAI_API_KEY"),
            ("openai_model", "OPENAI_MODEL"),
        ):
            if key in secrets and str(secrets[key]).strip():
                val = str(secrets[key]).strip()
                setattr(settings, attr, val.lower() if attr == "llm_provider" else val)
    except Exception:
        pass
    return CourseRAGPipeline(settings=settings, load_default_handbook=True)


pipeline = get_pipeline()

if "chat_turns" not in st.session_state:
    st.session_state.chat_turns = []
if "uploaded_tracker" not in st.session_state:
    st.session_state.uploaded_tracker = set()

MODE_LABELS = {
    "Hybrid (Dense + BM25 RRF)": "hybrid",
    "Dense Semantic (Qdrant BGE)": "dense",
    "BM25 Lexical Only": "bm25",
}

PROVIDER_LABELS = {
    "Auto (Cloud if configured, else Local)": "auto",
    "Groq Cloud (openai/gpt-oss-120b)": "groq",
    "Google Gemini (2.5 Flash)": "gemini",
    "OpenAI (GPT-4o-mini)": "openai",
    "Local Extractive Mode (Offline)": "local",
}

STARTER_QUESTIONS = [
    "How does CPython handle reference cycles in memory?",
    "Why is Python's dict compact and how are collisions resolved?",
    "Compare BFS, Dijkstra, and Bellman-Ford shortest path algorithms",
    "How does Timsort achieve O(n) best-case sorting in Python?",
    "Show how 0/1 Knapsack reduces space complexity to 1D",
]

with st.sidebar:
    st.markdown("### Indexed Documents")
    docs = pipeline.list_documents()
    for doc in docs:
        col_info, col_del = st.columns([5, 1])
        with col_info:
            tag = "Built-in" if doc.is_builtin else "Uploaded"
            st.markdown(
                f"""
                <div class="doc-item">
                    <strong>{doc.source_name}</strong>
                    <span>{tag} · {doc.page_count} pages · {doc.chunk_count} chunks</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_del:
            if not doc.is_builtin:
                if st.button("✕", key=f"rm_{doc.source_name}", help=f"Remove {doc.source_name}"):
                    pipeline.remove_document(doc.source_name)
                    st.session_state.uploaded_tracker.discard(doc.source_name)
                    st.rerun()

    uploaded_files = st.file_uploader(
        "Add study notes or PDFs",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        help="Upload lecture slides, textbook chapters, or notes to query alongside the built-in guide.",
    )
    if uploaded_files:
        newly_indexed = 0
        for uf in uploaded_files:
            sig = f"{uf.name}:{uf.size}"
            if sig not in st.session_state.uploaded_tracker:
                try:
                    pipeline.index_bytes(uf.name, uf.getvalue(), is_builtin=False)
                    st.session_state.uploaded_tracker.add(sig)
                    newly_indexed += 1
                except Exception as exc:
                    st.error(f"Could not index {uf.name}: {exc}")
        if newly_indexed:
            st.rerun()

    all_doc_names = [d.source_name for d in pipeline.list_documents()]
    selected_sources: list[str] | None = None
    if len(all_doc_names) > 1:
        chosen = st.multiselect(
            "Filter search scope",
            options=all_doc_names,
            default=all_doc_names,
        )
        if chosen and len(chosen) < len(all_doc_names):
            selected_sources = chosen

    st.divider()
    st.markdown("### Retrieval & Model")

    selected_mode_label = st.selectbox("Retrieval strategy", list(MODE_LABELS.keys()), index=0)
    retrieval_mode = MODE_LABELS[selected_mode_label]

    top_k = st.slider("Retrieved excerpts (top-k)", min_value=2, max_value=8, value=5)

    selected_prov_label = st.selectbox("Answer synthesis", list(PROVIDER_LABELS.keys()), index=0)
    chosen_provider = PROVIDER_LABELS[selected_prov_label]
    pipeline.settings.llm_provider = chosen_provider

    if chosen_provider == "groq":
        key_in = st.text_input(
            "Groq API Key",
            value=pipeline.settings.groq_api_key,
            type="password",
            placeholder="gsk_...",
        )
        if key_in.strip():
            pipeline.settings.groq_api_key = key_in.strip()
    elif chosen_provider == "gemini":
        key_in = st.text_input(
            "Gemini API Key",
            value=pipeline.settings.gemini_api_key,
            type="password",
            placeholder="AIza...",
        )
        if key_in.strip():
            pipeline.settings.gemini_api_key = key_in.strip()
    elif chosen_provider == "openai":
        key_in = st.text_input(
            "OpenAI API Key",
            value=pipeline.settings.openai_api_key,
            type="password",
            placeholder="sk-...",
        )
        if key_in.strip():
            pipeline.settings.openai_api_key = key_in.strip()

    st.divider()
    st.markdown("### Sample Prompts")
    for sample_q in STARTER_QUESTIONS:
        if st.button(sample_q, use_container_width=True):
            st.session_state.queued_prompt = sample_q

    if st.session_state.chat_turns:
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_turns = []
            st.rerun()

# Main Header Banner
active_prov, active_model, _ = pipeline.settings.resolve_active_provider()
mode_badge = (
    "Engine: Local Extractive RAG"
    if active_prov == "local"
    else f"Engine: {active_prov.upper()} ({active_model})"
)
corpus_stats = pipeline.stats()

st.markdown(
    f"""
    <div class="banner-box">
        <div>
            <h1>CourseRAG</h1>
            <p>Hybrid dense-vector + BM25 retrieval workspace for Python internals, algorithms, and uploaded study notes.</p>
        </div>
        <div class="badge-group">
            <span class="mode-pill">{mode_badge}</span>
            <span class="stats-pill">{corpus_stats['documents']} doc(s) · {corpus_stats['pages']} pages · {corpus_stats['chunks']} chunks</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.chat_turns:
    st.markdown(
        """
        <div class="intro-panel">
            <h3>Ask a technical question or upload your own course files</h3>
            <p>
                Every query runs through contextual query expansion, parallel dense vector similarity
                (<code>BAAI/bge-small-en-v1.5</code> in Qdrant) and lexical keyword scoring (<code>BM25Okapi</code>),
                fused via Reciprocal Rank Fusion. Expand the <strong>Retrieved Excerpts &amp; Hybrid Scores</strong>
                panel under any answer to inspect page-level citations and component rankings.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render conversation history
for turn in st.session_state.chat_turns:
    avatar_icon = "🧑‍🎓" if turn["role"] == "user" else "📘"
    with st.chat_message(turn["role"], avatar=avatar_icon):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("meta"):
            meta = turn["meta"]
            st.markdown(
                f'<div class="telemetry-line">'
                f'Strategy: <strong>{meta["mode"].upper()}</strong> · '
                f'Retrieval: <strong>{meta["retrieval_ms"]} ms</strong> · '
                f'Synthesis: <strong>{meta["provider"]}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
            chunks = meta.get("chunks", [])
            if chunks:
                with st.expander(f"Retrieved Excerpts & Hybrid Scores ({len(chunks)})", expanded=False):
                    for idx, item in enumerate(chunks, start=1):
                        d_rank = f"#{item['dense_rank']}" if item["dense_rank"] else "—"
                        b_rank = f"#{item['bm25_rank']}" if item["bm25_rank"] else "—"
                        preview = item["text"][:360] + ("…" if len(item["text"]) > 360 else "")
                        st.markdown(
                            f"""
                            <div class="chunk-card">
                                <div class="chunk-meta">
                                    <span>[{idx}] {item['source']} · Page {item['page']} · {item['section']}</span>
                                    <span>Score: {item['score']} (Dense: {item['dense_score']} [{d_rank}] | BM25: {item['bm25_score']} [{b_rank}])</span>
                                </div>
                                <div style="color:#3E322B; white-space:pre-wrap;">{preview}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

queued = st.session_state.pop("queued_prompt", None)
user_question = queued or st.chat_input("Ask about Python internals, data structures, algorithms, or uploaded notes...")

if user_question:
    prior_history = [
        {"role": t["role"], "content": t["content"]}
        for t in st.session_state.chat_turns
    ]
    st.session_state.chat_turns.append({"role": "user", "content": user_question})

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar="📘"):
        with st.spinner("Searching knowledge base and preparing cited response..."):
            result = pipeline.ask(
                question=user_question,
                chat_history=prior_history,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                source_filter=selected_sources,
            )

        st.markdown(result.answer)
        serialized_chunks = [
            {
                "source": r.chunk.source_name,
                "page": r.chunk.page_number,
                "section": r.chunk.section_title,
                "score": r.score,
                "dense_score": r.dense_score,
                "bm25_score": r.bm25_score,
                "dense_rank": r.dense_rank,
                "bm25_rank": r.bm25_rank,
                "text": r.chunk.text,
            }
            for r in result.retrieved_chunks
        ]
        turn_meta = {
            "mode": retrieval_mode,
            "retrieval_ms": result.retrieval_ms,
            "total_ms": result.total_ms,
            "provider": f"{result.provider_used} ({result.model_used})",
            "rewritten_query": result.rewritten_query,
            "chunks": serialized_chunks,
        }
        st.markdown(
            f'<div class="telemetry-line">'
            f'Strategy: <strong>{retrieval_mode.upper()}</strong> · '
            f'Retrieval: <strong>{result.retrieval_ms} ms</strong> · '
            f'Synthesis: <strong>{turn_meta["provider"]}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if serialized_chunks:
            with st.expander(f"Retrieved Excerpts & Hybrid Scores ({len(serialized_chunks)})", expanded=False):
                for idx, item in enumerate(serialized_chunks, start=1):
                    d_rank = f"#{item['dense_rank']}" if item["dense_rank"] else "—"
                    b_rank = f"#{item['bm25_rank']}" if item["bm25_rank"] else "—"
                    preview = item["text"][:360] + ("…" if len(item["text"]) > 360 else "")
                    st.markdown(
                        f"""
                        <div class="chunk-card">
                            <div class="chunk-meta">
                                <span>[{idx}] {item['source']} · Page {item['page']} · {item['section']}</span>
                                <span>Score: {item['score']} (Dense: {item['dense_score']} [{d_rank}] | BM25: {item['bm25_score']} [{b_rank}])</span>
                            </div>
                            <div style="color:#3E322B; white-space:pre-wrap;">{preview}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.session_state.chat_turns.append(
            {
                "role": "assistant",
                "content": result.answer,
                "meta": turn_meta,
            }
        )
