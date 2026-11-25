"""Streamlit web interface for CourseRAG with interactive Dark / Light mode switcher."""

from __future__ import annotations

import html
import streamlit as st

from src.config import RAGSettings
from src.pipeline import CourseRAGPipeline

st.set_page_config(
    page_title="CourseRAG — Technical Study Assistant",
    page_icon=":material/terminal:",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "bg_main": "#0B0F19",
        "bg_sidebar": "#111827",
        "bg_card": "#131C2E",
        "bg_card_alt": "#0F172A",
        "bg_hero_grad": "linear-gradient(135deg, #131C31 0%, #0F172A 65%, #161F38 100%)",
        "bg_input": "#0F172A",
        "bg_hover": "#1E293B",
        "bg_code": "#090D16",
        "border": "#1E293B",
        "border_strong": "#334155",
        "text_primary": "#F1F5F9",
        "text_secondary": "#94A3B8",
        "text_muted": "#64748B",
        "accent": "#6366F1",
        "accent_hover": "#4F46E5",
        "accent_text": "#A5B4FC",
        "accent_soft": "rgba(99, 102, 241, 0.14)",
        "accent_border": "rgba(99, 102, 241, 0.36)",
        "status_bg": "rgba(16, 185, 129, 0.12)",
        "status_text": "#34D399",
        "status_border": "rgba(52, 211, 153, 0.28)",
        "table_head": "#1E293B",
        "table_stripe": "rgba(30, 41, 59, 0.35)",
        "shadow": "0 4px 20px rgba(2, 6, 23, 0.35)",
    },
    "light": {
        "bg_main": "#F8FAFC",
        "bg_sidebar": "#F1F5F9",
        "bg_card": "#FFFFFF",
        "bg_card_alt": "#F8FAFC",
        "bg_hero_grad": "linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 70%, #EEF2FF 100%)",
        "bg_input": "#FFFFFF",
        "bg_hover": "#EEF2FF",
        "bg_code": "#F1F5F9",
        "border": "#E2E8F0",
        "border_strong": "#CBD5E1",
        "text_primary": "#0F172A",
        "text_secondary": "#475569",
        "text_muted": "#64748B",
        "accent": "#4F46E5",
        "accent_hover": "#4338CA",
        "accent_text": "#4338CA",
        "accent_soft": "rgba(79, 70, 229, 0.09)",
        "accent_border": "rgba(79, 70, 229, 0.28)",
        "status_bg": "#ECFDF5",
        "status_text": "#047857",
        "status_border": "#A7F3D0",
        "table_head": "#F1F5F9",
        "table_stripe": "#F8FAFC",
        "shadow": "0 2px 10px rgba(15, 23, 42, 0.05)",
    },
}


def inject_theme_css(mode: str) -> None:
    t = THEMES.get(mode, THEMES["dark"])
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Core surfaces, Streamlit CSS variables & typography */
        :root, html, body, .stApp {{
            --primary-color: {t['accent']} !important;
            --background-color: {t['bg_main']} !important;
            --secondary-background-color: {t['bg_sidebar']} !important;
            --text-color: {t['text_primary']} !important;
            color-scheme: {mode};
        }}
        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {{
            background-color: {t['bg_main']} !important;
            color: {t['text_primary']} !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
        }}
        [data-testid="stHeader"] button,
        [data-testid="stToolbar"] button,
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="collapsedControl"] button {{
            color: {t['text_secondary']} !important;
        }}

        /* Sidebar styling */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {{
            background-color: {t['bg_sidebar']} !important;
            border-right: 1px solid {t['border']} !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: {t['text_primary']} !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            margin-bottom: 0.45rem !important;
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {{
            color: {t['text_secondary']} !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.86rem !important;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: {t['border']} !important;
            margin: 1rem 0 !important;
        }}

        /* Preserve Material Icons everywhere */
        [data-testid="stIconMaterial"],
        .material-icons {{
            font-family: "Material Symbols Rounded" !important;
        }}

        /* Selectbox (React Aria + BaseWeb), multiselect, text inputs & popovers */
        .stApp [data-testid="stSelectbox"] .react-aria-ComboBox [role="group"],
        .stApp [data-testid="stSelectbox"] div[data-rac][role="group"],
        .stApp [data-testid="stMultiSelect"] [role="group"],
        .stApp [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        .stApp [data-testid="stSidebar"] [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
        .stApp [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        .stApp [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
        .stApp [data-testid="stSidebar"] [data-testid="stTextInput"] div[data-baseweb="input"],
        .stApp [data-testid="stSidebar"] [data-testid="stTextInput"] div[data-baseweb="base-input"],
        .stApp [data-testid="stTextInput"] div[data-baseweb="input"],
        .stApp [data-testid="stTextInput"] div[data-baseweb="base-input"] {{
            background: {t['bg_input']} !important;
            background-color: {t['bg_input']} !important;
            border: 1px solid {t['border_strong']} !important;
            border-radius: 9px !important;
            color: {t['text_primary']} !important;
        }}
        .stApp [data-testid="stSelectbox"] input[role="combobox"],
        .stApp [data-testid="stSelectbox"] button[aria-haspopup="listbox"],
        .stApp [data-testid="stSidebar"] div[data-baseweb="select"] > div > div,
        .stApp div[data-baseweb="select"] > div > div,
        .stApp [data-testid="stSidebar"] [data-baseweb="select"] span,
        .stApp [data-baseweb="select"] span,
        .stApp [data-baseweb="input"] input {{
            background: transparent !important;
            background-color: transparent !important;
            color: {t['text_primary']} !important;
        }}
        .stApp [data-testid="stSelectbox"] button[aria-haspopup="listbox"] svg,
        .stApp [data-testid="stSidebar"] [data-baseweb="select"] svg,
        .stApp [data-baseweb="select"] svg {{
            fill: {t['text_secondary']} !important;
            color: {t['text_secondary']} !important;
        }}
        .react-aria-Popover,
        div[data-rac][role="listbox"],
        div[role="listbox"],
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        ul[role="listbox"] {{
            background: {t['bg_card']} !important;
            background-color: {t['bg_card']} !important;
            border: 1px solid {t['border_strong']} !important;
            border-radius: 10px !important;
        }}
        div[data-rac][role="option"],
        div[role="option"],
        li[role="option"],
        [data-baseweb="menu"] li {{
            color: {t['text_primary']} !important;
            background-color: transparent !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.88rem !important;
        }}
        div[role="option"][data-hovered],
        div[role="option"][data-focused],
        div[role="option"][data-selected],
        div[role="option"]:hover,
        div[role="option"][aria-selected="true"],
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {{
            background-color: {t['bg_hover']} !important;
        }}

        /* File uploader dropzone */
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {t['bg_card_alt']} !important;
            border: 1px dashed {t['border_strong']} !important;
            border-radius: 10px !important;
            padding: 12px !important;
        }}
        [data-testid="stFileUploaderDropzone"] div,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span:not([data-testid="stIconMaterial"]) {{
            color: {t['text_secondary']} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background-color: {t['bg_card']} !important;
            color: {t['text_primary']} !important;
            border: 1px solid {t['border_strong']} !important;
            border-radius: 8px !important;
        }}

        /* Buttons (secondary & active primary pill) */
        div.stButton > button {{
            background-color: {t['bg_card']} !important;
            color: {t['text_primary']} !important;
            border: 1px solid {t['border_strong']} !important;
            border-radius: 9px !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.84rem !important;
            font-weight: 500 !important;
            padding: 0.42rem 0.75rem !important;
            transition: all 0.14s ease !important;
        }}
        div.stButton > button:hover {{
            background-color: {t['bg_hover']} !important;
            border-color: {t['accent']} !important;
            color: {t['text_primary']} !important;
        }}
        div.stButton > button[kind="primary"] {{
            background-color: {t['accent']} !important;
            color: #FFFFFF !important;
            border: 1px solid {t['accent']} !important;
            font-weight: 600 !important;
        }}
        div.stButton > button[kind="primary"] p,
        div.stButton > button[kind="primary"] span:not([data-testid="stIconMaterial"]) {{
            color: #FFFFFF !important;
        }}
        div.stButton > button[kind="primary"]:hover {{
            background-color: {t['accent_hover']} !important;
            border-color: {t['accent_hover']} !important;
        }}

        /* Hero workspace header */
        .workspace-header {{
            background: {t['bg_hero_grad']};
            border: 1px solid {t['border']};
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 18px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 16px;
            box-shadow: {t['shadow']};
        }}
        .eyebrow-tag {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {t['accent_text']};
            background: {t['accent_soft']};
            border: 1px solid {t['accent_border']};
            padding: 3px 9px;
            border-radius: 6px;
            margin-bottom: 8px;
        }}
        .workspace-header h1 {{
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: {t['text_primary']};
            font-family: 'Inter', sans-serif;
        }}
        .workspace-header p {{
            margin: 6px 0 0;
            color: {t['text_secondary']};
            font-size: 0.93rem;
            line-height: 1.5;
            max-width: 640px;
        }}
        .header-Telemetry {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 7px;
        }}
        .engine-badge {{
            background: {t['status_bg']};
            color: {t['status_text']};
            border: 1px solid {t['status_border']};
            padding: 5px 12px;
            border-radius: 999px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            font-weight: 600;
            white-space: nowrap;
        }}
        .corpus-badge {{
            background: {t['bg_card_alt']};
            color: {t['text_secondary']};
            border: 1px solid {t['border_strong']};
            padding: 4px 11px;
            border-radius: 999px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            white-space: nowrap;
        }}

        /* Empty-state architecture feature grid */
        .feature-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 18px;
        }}
        @media (max-width: 900px) {{
            .feature-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .feature-card {{
            background: {t['bg_card']};
            border: 1px solid {t['border']};
            border-radius: 12px;
            padding: 15px 17px;
            box-shadow: {t['shadow']};
        }}
        .feature-card .f-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            color: {t['accent_text']};
            margin-bottom: 5px;
        }}
        .feature-card h4 {{
            margin: 0 0 5px;
            font-size: 0.95rem;
            font-weight: 600;
            color: {t['text_primary']};
        }}
        .feature-card p {{
            margin: 0;
            font-size: 0.84rem;
            line-height: 1.52;
            color: {t['text_secondary']};
        }}

        /* Sidebar document item card */
        .doc-card {{
            background: {t['bg_card']};
            border: 1px solid {t['border']};
            border-left: 3px solid {t['accent']};
            border-radius: 9px;
            padding: 9px 11px;
            margin-bottom: 8px;
        }}
        .doc-card strong {{
            display: block;
            color: {t['text_primary']};
            font-size: 0.82rem;
            font-weight: 600;
            word-break: break-word;
        }}
        .doc-card .doc-sub {{
            font-family: 'JetBrains Mono', monospace;
            color: {t['text_muted']};
            font-size: 0.72rem;
            margin-top: 2px;
        }}

        /* Chat messages & markdown content */
        [data-testid="stChatMessage"] {{
            background-color: {t['bg_card']} !important;
            border: 1px solid {t['border']} !important;
            border-radius: 14px !important;
            padding: 16px 20px !important;
            margin-bottom: 12px !important;
            box-shadow: {t['shadow']};
        }}
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] td,
        [data-testid="stChatMessage"] th {{
            color: {t['text_primary']} !important;
            line-height: 1.68 !important;
            font-size: 0.95rem !important;
        }}
        [data-testid="stChatMessage"] h1,
        [data-testid="stChatMessage"] h2,
        [data-testid="stChatMessage"] h3,
        [data-testid="stChatMessage"] h4 {{
            color: {t['text_primary']} !important;
            letter-spacing: -0.015em;
        }}
        [data-testid="stChatMessage"] blockquote {{
            border-left: 3px solid {t['accent']} !important;
            background: {t['accent_soft']};
            padding: 8px 14px;
            border-radius: 0 8px 8px 0;
            color: {t['text_secondary']} !important;
        }}
        [data-testid="stChatMessage"] table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.87rem;
            margin: 10px 0;
        }}
        [data-testid="stChatMessage"] th,
        [data-testid="stChatMessage"] td {{
            border: 1px solid {t['border_strong']} !important;
            padding: 7px 10px !important;
        }}
        [data-testid="stChatMessage"] th {{
            background-color: {t['table_head']} !important;
            font-weight: 600 !important;
        }}
        [data-testid="stChatMessage"] tr:nth-child(even) td {{
            background-color: {t['table_stripe']};
        }}
        [data-testid="stChatMessage"] [data-testid="stCode"],
        [data-testid="stChatMessage"] pre {{
            background-color: {t['bg_code']} !important;
            border: 1px solid {t['border_strong']} !important;
            border-left: 3px solid {t['accent']} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stChatMessage"] pre {{
            padding: 12px 14px !important;
            margin: 8px 0 !important;
        }}
        [data-testid="stChatMessage"] [data-testid="stCode"] pre {{
            border: none !important;
            background: transparent !important;
        }}
        [data-testid="stChatMessage"] pre code,
        [data-testid="stChatMessage"] [data-testid="stCode"] code,
        [data-testid="stChatMessage"] pre span {{
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
            font-family: 'JetBrains Mono', Consolas, monospace !important;
            font-size: 0.85rem !important;
        }}
        [data-testid="stChatMessage"] pre code {{
            color: {t['text_primary']} !important;
        }}
        [data-testid="stChatMessage"] p > code,
        [data-testid="stChatMessage"] li > code,
        [data-testid="stChatMessage"] strong > code,
        [data-testid="stChatMessage"] em > code,
        [data-testid="stChatMessage"] td > code,
        [data-testid="stChatMessage"] th > code,
        [data-testid="stChatMessage"] h3 > code,
        [data-testid="stChatMessage"] h4 > code {{
            background-color: {t['accent_soft']} !important;
            color: {t['accent_text']} !important;
            border: 1px solid {t['accent_border']} !important;
            padding: 1px 6px !important;
            border-radius: 5px !important;
            font-family: 'JetBrains Mono', Consolas, monospace !important;
            font-size: 0.83em !important;
        }}

        /* Telemetry strip & expander chunk cards */
        .telemetry-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
            margin-bottom: 6px;
        }}
        .t-pill {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.73rem;
            color: {t['text_secondary']};
            background: {t['bg_card_alt']};
            border: 1px solid {t['border']};
            padding: 3px 9px;
            border-radius: 6px;
        }}
        .t-pill strong {{
            color: {t['accent_text']};
        }}

        [data-testid="stExpander"],
        [data-testid="stExpander"] details {{
            background-color: {t['bg_card_alt']} !important;
            border: 1px solid {t['border_strong']} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] details[open] > summary,
        [data-testid="stExpander"] summary:hover {{
            background-color: {t['bg_card_alt']} !important;
            color: {t['text_primary']} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary svg,
        [data-testid="stExpander"] summary span {{
            color: {t['text_secondary']} !important;
            fill: {t['text_secondary']} !important;
            font-size: 0.84rem !important;
            font-weight: 500 !important;
        }}

        .chunk-card {{
            background-color: {t['bg_card']};
            border: 1px solid {t['border']};
            border-radius: 9px;
            padding: 11px 14px;
            margin-bottom: 9px;
        }}
        .chunk-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 6px;
        }}
        .chunk-title {{
            font-size: 0.82rem;
            font-weight: 600;
            color: {t['accent_text']};
        }}
        .chunk-scores {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.73rem;
            color: {t['text_secondary']};
            background: {t['bg_card_alt']};
            border: 1px solid {t['border']};
            padding: 2px 8px;
            border-radius: 5px;
        }}
        .chunk-body {{
            font-size: 0.84rem;
            line-height: 1.55;
            color: {t['text_secondary']};
            white-space: pre-wrap;
        }}

        /* Bottom Chat Input Container */
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"] {{
            background-color: {t['bg_main']} !important;
        }}
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div {{
            background-color: {t['bg_input']} !important;
            border-radius: 13px !important;
        }}
        [data-testid="stChatInput"] {{
            border: 1px solid {t['border_strong']} !important;
        }}
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] [data-baseweb="textarea"],
        [data-testid="stChatInput"] [data-baseweb="base-input"] {{
            background-color: {t['bg_input']} !important;
            border: none !important;
        }}
        [data-testid="stChatInput"]:focus-within {{
            border-color: {t['accent']} !important;
            box-shadow: 0 0 0 2px {t['accent_soft']} !important;
        }}
        [data-testid="stChatInput"] textarea {{
            background-color: transparent !important;
            color: {t['text_primary']} !important;
            caret-color: {t['accent']} !important;
            font-family: 'Inter', sans-serif !important;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: {t['text_muted']} !important;
            opacity: 1 !important;
        }}
        [data-testid="stChatInput"] button {{
            background-color: {t['accent']} !important;
            color: #FFFFFF !important;
            border-radius: 9px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_pipeline() -> CourseRAGPipeline:
    settings = RAGSettings()
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

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "dark"
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


def render_turn_diagnostics(meta: dict) -> None:
    st.markdown(
        f"""
        <div class="telemetry-bar">
            <span class="t-pill">Strategy: <strong>{html.escape(meta['mode'].upper())}</strong></span>
            <span class="t-pill">Retrieval: <strong>{meta['retrieval_ms']} ms</strong></span>
            <span class="t-pill">Engine: <strong>{html.escape(meta['provider'])}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chunks = meta.get("chunks", [])
    if chunks:
        with st.expander(f"Retrieved Excerpts & Hybrid Scores ({len(chunks)})", expanded=False):
            for idx, item in enumerate(chunks, start=1):
                d_rank = f"#{item['dense_rank']}" if item["dense_rank"] else "—"
                b_rank = f"#{item['bm25_rank']}" if item["bm25_rank"] else "—"
                preview = html.escape(
                    item["text"][:380] + ("…" if len(item["text"]) > 380 else "")
                )
                src_label = html.escape(
                    f"[{idx}] {item['source']} · p. {item['page']} · {item['section']}"
                )
                score_label = (
                    f"RRF {item['score']}  |  Dense {item['dense_score']} ({d_rank})  |  "
                    f"BM25 {item['bm25_score']} ({b_rank})"
                )
                st.markdown(
                    f"""
                    <div class="chunk-card">
                        <div class="chunk-header">
                            <span class="chunk-title">{src_label}</span>
                            <span class="chunk-scores">{score_label}</span>
                        </div>
                        <div class="chunk-body">{preview}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


with st.sidebar:
    st.markdown("### Appearance")
    col_dark, col_light = st.columns(2)
    with col_dark:
        if st.button(
            "Dark",
            icon=":material/dark_mode:",
            type="primary" if st.session_state.ui_theme == "dark" else "secondary",
            use_container_width=True,
            key="btn_theme_dark",
        ):
            if st.session_state.ui_theme != "dark":
                st.session_state.ui_theme = "dark"
                st.rerun()
    with col_light:
        if st.button(
            "Light",
            icon=":material/light_mode:",
            type="primary" if st.session_state.ui_theme == "light" else "secondary",
            use_container_width=True,
            key="btn_theme_light",
        ):
            if st.session_state.ui_theme != "light":
                st.session_state.ui_theme = "light"
                st.rerun()

    st.divider()
    st.markdown("### Knowledge Base")
    docs = pipeline.list_documents()
    for doc in docs:
        col_info, col_del = st.columns([5, 1])
        with col_info:
            tag = "Built-in" if doc.is_builtin else "Uploaded"
            st.markdown(
                f"""
                <div class="doc-card">
                    <strong>{html.escape(doc.source_name)}</strong>
                    <div class="doc-sub">{tag} · {doc.page_count}p · {doc.chunk_count} chunks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_del:
            if not doc.is_builtin:
                if st.button("✕", key=f"rm_{doc.source_name}", help=f"Remove {doc.source_name}"):
                    pipeline.remove_document(doc.source_name)
                    st.session_state.uploaded_tracker = {
                        s for s in st.session_state.uploaded_tracker
                        if not s.startswith(f"{doc.source_name}:")
                    }
                    st.rerun()

    uploaded_files = st.file_uploader(
        "Upload notes or textbooks",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        help="Index extra PDFs, DOCX files, Markdown, or plain text alongside the built-in guide.",
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
            "Search scope",
            options=all_doc_names,
            default=all_doc_names,
        )
        if chosen and len(chosen) < len(all_doc_names):
            selected_sources = chosen

    st.divider()
    st.markdown("### Retrieval & Model")

    selected_mode_label = st.selectbox("Retrieval strategy", list(MODE_LABELS.keys()), index=0)
    retrieval_mode = MODE_LABELS[selected_mode_label]

    top_k = st.slider("Top-K excerpts", min_value=2, max_value=8, value=5)

    selected_prov_label = st.selectbox("Synthesis engine", list(PROVIDER_LABELS.keys()), index=0)
    chosen_provider = PROVIDER_LABELS[selected_prov_label]
    pipeline.settings.llm_provider = chosen_provider

    if chosen_provider == "groq":
        key_in = st.text_input(
            "Groq API Key",
            value=pipeline.settings.groq_api_key,
            type="password",
            placeholder="Paste Groq API key...",
        )
        if key_in.strip():
            pipeline.settings.groq_api_key = key_in.strip()
    elif chosen_provider == "gemini":
        key_in = st.text_input(
            "Gemini API Key",
            value=pipeline.settings.gemini_api_key,
            type="password",
            placeholder="Paste Gemini API key...",
        )
        if key_in.strip():
            pipeline.settings.gemini_api_key = key_in.strip()
    elif chosen_provider == "openai":
        key_in = st.text_input(
            "OpenAI API Key",
            value=pipeline.settings.openai_api_key,
            type="password",
            placeholder="Paste OpenAI API key...",
        )
        if key_in.strip():
            pipeline.settings.openai_api_key = key_in.strip()

    st.divider()
    st.markdown("### Sample Queries")
    for idx_q, sample_q in enumerate(STARTER_QUESTIONS):
        if st.button(sample_q, key=f"side_q_{idx_q}", use_container_width=True):
            st.session_state.queued_prompt = sample_q

    if st.session_state.chat_turns:
        st.divider()
        if st.button("Clear conversation", icon=":material/delete_sweep:", use_container_width=True):
            st.session_state.chat_turns = []
            st.rerun()

# Apply active theme stylesheet
inject_theme_css(st.session_state.ui_theme)

# Workspace Hero Banner
active_prov, active_model, _ = pipeline.settings.resolve_active_provider()
mode_badge = (
    "● LOCAL · extractive-mmr"
    if active_prov == "local"
    else f"● {active_prov.upper()} · {active_model}"
)
corpus_stats = pipeline.stats()

st.markdown(
    f"""
    <div class="workspace-header">
        <div>
            <span class="eyebrow-tag">Hybrid Dense + BM25 RAG</span>
            <h1>CourseRAG</h1>
            <p>Source-grounded technical study workspace for Python internals, algorithms, data structures, and custom lecture notes.</p>
        </div>
        <div class="header-Telemetry">
            <span class="engine-badge">{html.escape(mode_badge)}</span>
            <span class="corpus-badge">{corpus_stats['documents']} doc · {corpus_stats['pages']} pages · {corpus_stats['chunks']} chunks</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.chat_turns:
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="f-tag">01 / SEMANTIC + LEXICAL</div>
                <h4>Reciprocal Rank Fusion</h4>
                <p>Combines 384-d BGE cosine vectors in Qdrant with code-aware BM25Okapi keyword scoring so exact symbols and concepts both match.</p>
            </div>
            <div class="feature-card">
                <div class="f-tag">02 / STRUCTURE-AWARE</div>
                <h4>Section &amp; Table Preservation</h4>
                <p>Tracks chapter headings across pages, extracts multi-column PDF tables cleanly, and keeps Python code blocks intact.</p>
            </div>
            <div class="feature-card">
                <div class="f-tag">03 / VERIFIABLE SOURCES</div>
                <h4>Inline Page Provenance</h4>
                <p>Cites exact document page numbers inline and exposes component Dense, BM25, and RRF ranks below every response.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render conversation turns
for turn in st.session_state.chat_turns:
    avatar_icon = ":material/person:" if turn["role"] == "user" else ":material/terminal:"
    with st.chat_message(turn["role"], avatar=avatar_icon):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("meta"):
            render_turn_diagnostics(turn["meta"])

queued = st.session_state.pop("queued_prompt", None)
user_question = queued or st.chat_input(
    "Ask about Python internals, Big-O complexity, graph algorithms, or uploaded notes..."
)

if user_question:
    prior_history = [
        {"role": t["role"], "content": t["content"]}
        for t in st.session_state.chat_turns
    ]
    st.session_state.chat_turns.append({"role": "user", "content": user_question})

    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar=":material/terminal:"):
        with st.spinner("Retrieving passages and generating cited answer..."):
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
        render_turn_diagnostics(turn_meta)

        st.session_state.chat_turns.append(
            {
                "role": "assistant",
                "content": result.answer,
                "meta": turn_meta,
            }
        )
