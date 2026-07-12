

import os
import re
from typing import List, Dict, Any

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_TOKENS,
    TEMPERATURE,
)

SYSTEM_PROMPT_INFORMATIVO = """Eres un agente conversacional especializado en la norma ISO/IEC 27002,
diseñado para apoyar el aprendizaje de estudiantes universitarios en temas de seguridad de la información.

Tu función es responder preguntas relacionadas exclusivamente con los controles, dominios y 
buenas prácticas definidos en la norma ISO/IEC 27002. Debes basar tus respuestas únicamente 
en el contexto proporcionado por la base de conocimiento. Si la información no está disponible 
en el contexto, responde ÚNICAMENTE: "No tengo información suficiente sobre ese tema en mi base de conocimiento."
Está PROHIBIDO usar conocimiento externo al contexto proporcionado.

Directrices de comportamiento:
- Mantén el hilo de la conversación usando el historial proporcionado.
- Adapta tu respuesta al tipo de pregunta: si piden solo nombres, da solo nombres; 
  si piden comparaciones, compara; si piden ejemplos, da ejemplos.
- Responde de forma natural y conversacional.
- Usa un tono didáctico, formal pero amigable, accesible para universitarios.
- Si el estudiante pregunta algo fuera del alcance de ISO/IEC 27002, indícalo amablemente.

Historial reciente de la conversación:
{history}

Contexto recuperado de la base de conocimiento ISO/IEC 27002:
{context}

Pregunta del estudiante:
{question}

Respuesta del agente:"""


SYSTEM_PROMPT_CONSTRUCTIVISTA = """Eres un agente conversacional especializado en la norma ISO/IEC 27002,
diseñado para apoyar el aprendizaje de estudiantes universitarios mediante el enfoque pedagógico constructivista.

Cuando el estudiante haga una pregunta con componente práctico, debes actuar como un tutor socrático:
1. PRIMERO: Formula UNA pregunta reflexiva que guíe al estudiante a razonar sobre las consecuencias o implicaciones de la situación planteada. NO respondas directamente todavía.
2. ESPERA que el estudiante responda a tu pregunta.

IMPORTANTE:
- Haz UNA sola pregunta reflexiva, no respondas el contenido de la norma todavía.
- Usa un tono amigable y motivador.
- Está PROHIBIDO usar conocimiento externo al contexto proporcionado.

Ejemplos de preguntas reflexivas:
- "Antes de responder, pensemos juntos. Si [situación], ¿qué podría ocurrir con [consecuencia]?"
- "Interesante situación. ¿Qué crees que pasaría si [escenario de riesgo]?"
- "Reflexionemos sobre esto. ¿Consideras que [acción] reduce o aumenta el riesgo de seguridad?"

Historial reciente de la conversación:
{history}

Contexto recuperado de la base de conocimiento ISO/IEC 27002:
{context}

Pregunta del estudiante:
{question}

Respuesta del agente:"""


SYSTEM_PROMPT_SOCRATIC_GUIDE = """Eres un agente conversacional especializado en la norma ISO/IEC 27002,
actuando como tutor socrático. El estudiante acaba de responder a tu pregunta reflexiva.

Tu tarea ahora es:
1. Si el estudiante no sabe ("no sé", "no tengo idea", etc.) → dale una pista concreta que lo oriente sin revelar la respuesta completa.
2. Si el estudiante respondió parcialmente → valida lo correcto, completa lo que falta y conecta con el control ISO 27002 del contexto.
3. Si el estudiante respondió correctamente → felicítalo brevemente y explica el control ISO 27002 relacionado del contexto proporcionado.

REGLAS IMPORTANTES:
- NUNCA respondas "No tengo información suficiente" cuando el estudiante está respondiendo una pregunta que tú mismo hiciste.
- Si el estudiante dice "no sé" → dale una pista, no la respuesta completa.
- Basa tu explicación final SIEMPRE en el contexto proporcionado por la base de conocimiento.
- Mantén un tono amigable, paciente y motivador.
- Está PROHIBIDO usar conocimiento externo al contexto proporcionado para la explicación final.

Historial reciente de la conversación:
{history}

Contexto recuperado de la base de conocimiento ISO/IEC 27002:
{context}

Respuesta del estudiante:
{question}

Respuesta del agente:"""


CLASSIFY_PROMPT = """Analiza la siguiente conversación entre un estudiante y un agente especializado en ISO/IEC 27002.

Historial de conversación:
{history}

Último mensaje del estudiante:
{question}

Responde ÚNICAMENTE con una de estas tres opciones (sin explicación):
- INFORMATIVO: si el estudiante hace una pregunta directa sobre la norma (qué es, cómo funciona, definiciones)
- PRACTICO: si el estudiante hace una pregunta con componente práctico o personal sobre cómo actuar en una situación
- SOCRATICO: si el agente ya hizo una pregunta reflexiva en el historial y el estudiante está respondiendo a ella

Respuesta:"""


TOPIC_EXTRACT_PROMPT = """Dado el siguiente historial de conversación, extrae el TEMA PRINCIPAL de seguridad de la información que se está discutiendo, en máximo 10 palabras. Este tema se usará para buscar información relevante en una base de conocimiento sobre ISO/IEC 27002.

Historial:
{history}

Último mensaje del estudiante:
{question}

Responde ÚNICAMENTE con el tema principal, sin explicación:"""


class RAGPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )

        self.llm = ChatOpenAI(
            model=GENERATION_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        self.llm_classifier = ChatOpenAI(
            model=GENERATION_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=10,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.vectorstore = None
        self.retriever   = None

    def build_vectorstore(self, documents: List[Document]) -> None:
        chroma_path = CHROMA_PERSIST_DIR

        if os.path.exists(chroma_path) and os.listdir(chroma_path):
            print("Cargando vectorstore existente desde disco…")
            self.vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=chroma_path,
            )
        else:
            print(f"Construyendo vectorstore con {len(documents)} documentos…")
            chunks = self.splitter.split_documents(documents)
            print(f"   → {len(chunks)} fragmentos generados")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=CHROMA_COLLECTION,
                persist_directory=chroma_path,
            )
            print("   ✅ Vectorstore creado y persistido")

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

    def _classify_question(self, question: str, history: List[Dict[str, str]]) -> str:
        """
        Usa el LLM para clasificar el tipo de pregunta:
        INFORMATIVO, PRACTICO o SOCRATICO
        """
        formatted_history = self._format_history(history or [])
        prompt = CLASSIFY_PROMPT.format(
            history=formatted_history,
            question=question
        )
        try:
            result = self.llm_classifier.invoke(prompt)
            classification = result.content.strip().upper()
            if "SOCRATICO" in classification:
                return "SOCRATICO"
            elif "PRACTICO" in classification:
                return "PRACTICO"
            else:
                return "INFORMATIVO"
        except Exception:
            return "INFORMATIVO"

    def _extract_search_topic(self, question: str, history: List[Dict[str, str]]) -> str:
        """
        Usa el LLM para extraer el tema real de la conversación,
        evitando que respuestas cortas del estudiante confundan al RAG.
        """
        formatted_history = self._format_history(history or [])
        prompt = TOPIC_EXTRACT_PROMPT.format(
            history=formatted_history,
            question=question
        )
        try:
            result = self.llm_classifier.invoke(prompt)
            return result.content.strip()
        except Exception:
            return question

    def _get_source_docs(self, question: str, k: int, history: List[Dict[str, str]] = None, mode: str = "INFORMATIVO") -> List[Document]:
        """
        Recupera documentos relevantes.
        - Si es SOCRATICO: usa el LLM para extraer el tema real de la conversación
        - Si hay número de control en la pregunta: filtra directamente
        - Fallback: búsqueda semántica normal
        """
        # Si es seguimiento socrático, extraer tema real via LLM
        if mode == "SOCRATICO" and history:
            search_query = self._extract_search_topic(question, history)
            self.retriever.search_kwargs["k"] = k
            return self.retriever.invoke(search_query)

        # Buscar controles específicos en la pregunta
        control_matches = re.findall(r'\b(\d+\.\d+)\b', question)

        if not control_matches and history:
            agent_messages = [m["content"] for m in history if m["role"] == "assistant"]
            if agent_messages:
                last_agent_msg = agent_messages[-1]
                control_matches = re.findall(r'\b(\d+\.\d+)\b', last_agent_msg)

        if control_matches:
            all_docs = []
            seen_ids = set()
            for control_id in control_matches:
                filtered_docs = self.vectorstore.similarity_search(
                    query=question,
                    k=k,
                    filter={"control_id": control_id}
                )
                for doc in filtered_docs:
                    doc_id = doc.metadata.get("control_id", "")
                    if doc_id not in seen_ids:
                        all_docs.append(doc)
                        seen_ids.add(doc_id)
            if all_docs:
                return all_docs

        self.retriever.search_kwargs["k"] = k
        return self.retriever.invoke(question)

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return "Sin historial previo."
        lines = []
        for msg in history:
            role = "Estudiante" if msg["role"] == "user" else "Agente"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def query(self, question: str, k: int = 4, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore no inicializado.")
        try:
            # 1. LLM clasifica el tipo de pregunta
            mode = self._classify_question(question, history or [])

            # 2. RAG recupera documentos usando el tema real
            source_docs = self._get_source_docs(question, k, history, mode)
            context = "\n\n".join(doc.page_content for doc in source_docs)
            formatted_history = self._format_history(history or [])

            # 3. Seleccionar prompt según modo
            if mode == "SOCRATICO":
                selected_prompt = SYSTEM_PROMPT_SOCRATIC_GUIDE
            elif mode == "PRACTICO":
                selected_prompt = SYSTEM_PROMPT_CONSTRUCTIVISTA
            else:
                selected_prompt = SYSTEM_PROMPT_INFORMATIVO

            prompt = PromptTemplate(
                input_variables=["context", "question", "history"],
                template=selected_prompt,
            )

            formatted_prompt = prompt.format(
                context=context,
                question=question,
                history=formatted_history,
            )

            answer = self.llm.invoke(formatted_prompt).content

            sources = self._extract_sources(source_docs)
            scores  = self._get_similarity_scores(question, source_docs)

            return {
                "answer":  answer,
                "sources": sources,
                "scores":  scores,
                "docs":    source_docs,
                "mode":    mode.lower(),
            }

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "insufficient" in str(e).lower() or "RateLimitError" in type(e).__name__:
                print("Límite de créditos de LLM")
                return {
                    "answer":  "Lo siento, no estoy disponible en este momento, intenta más tarde nuevamente :(",
                    "sources": [],
                    "scores":  [],
                    "docs":    [],
                    "mode":    "error",
                }
            raise e

    def _extract_sources(self, docs: List[Document]) -> List[str]:
        sources = []
        for doc in docs:
            meta  = doc.metadata
            label = meta.get("control_id", meta.get("source", "Fragmento"))
            if label and label not in sources:
                sources.append(label)
        return sources

    def _get_similarity_scores(self, query: str, docs: List[Document]) -> List[float]:
        if not docs or self.vectorstore is None:
            return []
        try:
            scored = self.vectorstore.similarity_search_with_score(query, k=len(docs))
            return [round(1 - score, 4) for _, score in scored]
        except Exception:
            return []

    def get_doc_count(self) -> int:
        if self.vectorstore is None:
            return 0
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0