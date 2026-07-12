"""  streamlit run app.py
"""

import streamlit as st
from utils.rag_pipeline import RAGPipeline
from utils.session import init_session, save_message, get_conversation_for_context
from utils.dataset_loader import load_dataset

# Configuración de página
st.set_page_config(
    page_title="Agente ISO 27002 ",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --primary:    #0a0e1a;
        --secondary:  #111827;
        --accent:     #00d4ff;
        --accent2:    #7c3aed;
        --success:    #10b981;
        --warning:    #f59e0b;
        --text:       #e2e8f0;
        --muted:      #64748b;
        --border:     #1e293b;
        --card:       #0f172a;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--primary);
        color: var(--text);
    }

    /* Header */
    .agent-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 20px 0 10px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 24px;
    }
    .agent-header .icon {
        font-size: 2.2rem;
        line-height: 1;
    }
    .agent-header h1 {
        font-family: 'Space Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--accent);
        margin: 0;
        letter-spacing: -0.5px;
    }
    .agent-header p {
        font-size: 0.75rem;
        color: var(--muted);
        margin: 2px 0 0;
    }

    /* Chat bubbles */
    .msg-user {
        background: linear-gradient(135deg, #1e3a5f, #1e2a4a);
        border: 1px solid #2563eb44;
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        color: #e2e8f0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .msg-agent {
        background: var(--card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 4px 16px 16px 16px;
        padding: 16px 18px;
        margin: 8px 0;
        max-width: 85%;
        color: #e2e8f0;
        font-size: 0.92rem;
        line-height: 1.7;
    }
    .msg-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--muted);
        margin-bottom: 6px;
    }
    .msg-label.user-label { color: #3b82f6; }
    .msg-label.agent-label { color: var(--accent); }

    /* Sources badge */
    .source-tag {
        display: inline-block;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.72rem;
        font-family: 'Space Mono', monospace;
        color: var(--muted);
        margin: 4px 3px;
        cursor: default;
    }

    /* Input area */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--secondary) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(0,212,255,0.12) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(14,165,233,0.35) !important;
    }

    /* Divider */
    hr { border-color: var(--border) !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--primary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    section[data-testid="stSidebar"] {
    display: none !important;
}
button[data-testid="collapsedControl"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# Inicializar sesión
init_session()


# Carga Pipeline RAG
@st.cache_resource(show_spinner="Inicializando base de conocimiento ISO/IEC 27002…")
def get_pipeline():
    docs = load_dataset("dataset/")
    pipeline = RAGPipeline()
    pipeline.build_vectorstore(docs)
    return pipeline

# Fine Tuning
k_docs = 4           # fragmentos que recupera el RAG por consulta
show_sources = False # mostrar nombres de fuentes bajo cada respuesta
show_score = False    # mostrar similitud semántica bajo cada respuesta
history_turns = 5    # turnos de conversación a pasar como contexto

try:
    pipeline = get_pipeline()
    system_ready = True
except Exception as e:
    st.error(f"⚠️ Error al cargar el sistema: {e}")
    system_ready = False


# Área principal
st.markdown("""
<div class='agent-header'>
    <div class='icon'>🔐</div>
    <div>
        <h1>Agente ISO/IEC 27002</h1>
        <p>Asistente especializado en seguridad de la información</p>
    </div>
</div>
""", unsafe_allow_html=True)


if not st.session_state.messages:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0f172a,#1e1b4b); border:1px solid #312e81;
                border-radius:14px; padding:24px; margin:16px 0'>
        <div style='font-family:"Space Mono",monospace; color:#818cf8; font-size:0.75rem; 
                    letter-spacing:2px; margin-bottom:12px'>BIENVENIDO AL AGENTE</div>
        <p style='margin:0; color:#e2e8f0; line-height:1.7; font-size:0.95rem'>
            Soy tu asistente especializado en la norma <strong style='color:#00d4ff'>ISO/IEC 27002</strong>. 
            Puedo ayudarte a comprender controles de seguridad, buenas prácticas y gestión de 
            seguridad de la información. No te preocupes, todas mis respuestas están basadas en el contenido 
            oficial de la norma :).
        </p>
        <div style='margin-top:14px; display:flex; gap:8px; flex-wrap:wrap'>
            <span style='background:#1e293b; border:1px solid #334155; border-radius:6px; 
                        padding:4px 12px; font-size:0.75rem; color:#94a3b8'>📋 Controles</span>
            <span style='background:#1e293b; border:1px solid #334155; border-radius:6px; 
                        padding:4px 12px; font-size:0.75rem; color:#94a3b8'>🔒 Seguridad</span>
            <span style='background:#1e293b; border:1px solid #334155; border-radius:6px; 
                        padding:4px 12px; font-size:0.75rem; color:#94a3b8'>📚 Aprendizaje</span>
            <span style='background:#1e293b; border:1px solid #334155; border-radius:6px; 
                        padding:4px 12px; font-size:0.75rem; color:#94a3b8'>🎓 Universitario</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# HISTORIAL
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='msg-label user-label'>Estudiante</div>
            <div class='msg-user'>{msg["content"]}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='msg-label agent-label'>Agente ISO 27002</div>
            <div class='msg-agent'>{msg["content"]}</div>
            """, unsafe_allow_html=True)

            # Fuentes
            if show_sources and "sources" in msg and msg["sources"]:
                sources_html = "".join(
                    f"<span class='source-tag'>📄 {s}</span>"
                    for s in msg["sources"]
                )
                st.markdown(
                    f"<div style='margin-top:4px; margin-bottom:12px'>"
                    f"<span style='font-size:0.7rem;color:#475569;'>Fuentes: </span>"
                    f"{sources_html}</div>",
                    unsafe_allow_html=True
                )

            # Scores
            if show_score and "scores" in msg and msg["scores"]:
                scores_html = "".join(
                    f"<span class='source-tag'>sim: {s:.3f}</span>"
                    for s in msg["scores"]
                )
                st.markdown(
                    f"<div style='margin-bottom:12px'>"
                    f"<span style='font-size:0.7rem;color:#475569;'>Similitud semántica: </span>"
                    f"{scores_html}</div>",
                    unsafe_allow_html=True
                )


# ── Input del usuario ────────────────────────────────────────
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Tu pregunta",
            value="",
            placeholder="Escribe tu pregunta sobre ISO/IEC 27002…",
            label_visibility="collapsed"
        )
    with col_btn:
        submitted = st.form_submit_button("Enviar ➤", use_container_width=True)

# ── Procesar consulta ────────────────────────────────────────
if submitted and query.strip() and system_ready:
    save_message("user", query)

    # Obtener historial reciente para dar contexto al agente
    history = get_conversation_for_context(max_turns=history_turns)

    with st.spinner("Buscando en la base de conocimiento…"):
        result = pipeline.query(query.strip(), k=k_docs, history=history)

    save_message(
        "assistant",
        result["answer"],
        sources=result.get("sources", []),
        scores=result.get("scores", [])
    )
    st.rerun()

elif submitted and not system_ready:
    st.error("El sistema no está listo. Verifica que el dataset esté cargado.")