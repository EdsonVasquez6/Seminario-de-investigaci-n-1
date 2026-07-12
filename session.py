"""
session.py
──────────
Gestión del estado de sesión en Streamlit.
Mantiene el historial de conversación y preferencias del usuario.
"""

import streamlit as st
from typing import List, Dict, Any, Optional


def init_session() -> None:
    """Inicializa todas las variables de sesión necesarias."""
    defaults = {
        "messages":         [],   # Historial de mensajes
        "total_queries":    0,    # Contador de consultas
        "pending_question": None, # Pregunta sugerida pendiente
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_message(
    role: str,
    content: str,
    sources: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
) -> None:
    """
    Guarda un mensaje en el historial de sesión.

    Parámetros:
        role    : "user" o "assistant"
        content : Texto del mensaje
        sources : Lista de fuentes recuperadas (solo para assistant)
        scores  : Lista de scores de similitud  (solo para assistant)
    """
    message: Dict[str, Any] = {
        "role":    role,
        "content": content,
    }
    if sources is not None:
        message["sources"] = sources
    if scores is not None:
        message["scores"] = scores

    st.session_state.messages.append(message)

    if role == "user":
        st.session_state.total_queries += 1


def get_history() -> List[Dict[str, Any]]:
    """Retorna el historial completo de mensajes."""
    return st.session_state.get("messages", [])


def get_conversation_for_context(max_turns: int = 5) -> List[Dict[str, str]]:
    """
    Retorna los últimos N turnos de conversación en formato
    [{"role": ..., "content": ...}] para enviar como contexto adicional.

    Parámetros:
        max_turns : Número máximo de turnos (user + assistant = 1 turno)
    """
    history = get_history()
    # Tomar los últimos max_turns * 2 mensajes (cada turno = user + assistant)
    recent = history[-(max_turns * 2):]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def clear_history() -> None:
    """Limpia el historial de conversación."""
    st.session_state.messages = []
    st.session_state.total_queries = 0
