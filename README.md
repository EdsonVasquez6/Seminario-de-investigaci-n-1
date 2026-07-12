# 🔐 Agente ISO/IEC 27002 — README

**Proyecto de tesis:** Experiencia de usuario: Agente IA para el aprendizaje de la norma ISO 27002  
**Autor:** Edson Gerardo Vasquez Solis — Universidad de Lima

---

## 📁 Estructura del proyecto

```
iso27002_agent/
│
├── app.py                  ← Aplicación principal (Streamlit)
├── config.py               ← Configuración y variables (API key aquí)
├── requirements.txt        ← Dependencias
│
├── utils/
│   ├── rag_pipeline.py     ← Pipeline RAG (LangChain + Gemini + ChromaDB)
│   ├── dataset_loader.py   ← Cargador del dataset JSON/TXT
│   └── session.py          ← Manejo de sesión Streamlit
│
├── dataset/                ← 📌 AQUÍ va tu dataset
│   └── ejemplo_estructura.json  ← Ejemplo de estructura JSON
│
└── vectorstore/            ← Se crea automáticamente al indexar (ChromaDB)
```

---

## ⚡ Instalación rápida

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Obtener API Key de Google Gemini (gratis)
1. Ve a: https://aistudio.google.com/app/apikey
2. Crea una API Key gratuita
3. Copia la key

### 3. Configurar API Key
Crea un archivo `.env` en la raíz del proyecto:
```
GEMINI_API_KEY=tu_api_key_aqui
```
O edita directamente `config.py` línea 20.

### 4. Agregar tu dataset
Coloca tu archivo JSON en la carpeta `dataset/`. Ver estructura abajo.

### 5. Correr la aplicación
```bash
streamlit run app.py
```

---

## 📋 Estructura del dataset JSON

```json
[
  {
    "control_id":       "5.1",
    "title":            "Políticas para la seguridad de la información",
    "domain":           "Controles Organizacionales",
    "type":             "Preventivo",
    "objective":        "Objetivo del control...",
    "content":          "Descripción completa del control...",
    "implementation":   "Guía de implementación...",
    "examples":         ["Ejemplo 1", "Ejemplo 2"],
    "keywords":         ["política", "seguridad", "gestión"]
  }
]
```

**Campos requeridos:** `control_id`, `title`, `content`  
**Campos opcionales:** `domain`, `type`, `objective`, `implementation`, `examples`, `keywords`

---

## 🔄 Flujo del sistema

```
Estudiante pregunta
      ↓
Streamlit (app.py)
      ↓
RAGPipeline.query()
      ↓
Embedding de la pregunta (Gemini embedding-001)
      ↓
Búsqueda semántica en ChromaDB (similitud coseno)
      ↓
Top-K fragmentos recuperados de ISO 27002
      ↓
Prompt = System Prompt + Contexto + Pregunta
      ↓
Gemini 2.0 Flash genera respuesta
      ↓
Respuesta + Fuentes mostradas al estudiante
```

---

## ⚙️ Parámetros configurables (config.py)

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| GENERATION_MODEL | gemini-2.0-flash | LLM para generar respuestas |
| EMBEDDING_MODEL | models/embedding-001 | Modelo de embeddings |
| CHUNK_SIZE | 800 | Tamaño de fragmentos en caracteres |
| CHUNK_OVERLAP | 150 | Solapamiento entre fragmentos |
| TEMPERATURE | 0.2 | Creatividad del modelo (0=exacto, 1=creativo) |
| DEFAULT_K | 4 | Fragmentos recuperados por consulta |

---

## 🗄️ Re-indexar el dataset

Si cambias el dataset, elimina la carpeta `vectorstore/` y vuelve a correr la app:
```bash
rm -rf vectorstore/
streamlit run app.py
```

---

## 💰 Costos estimados (API Gemini)

El tier gratuito de Google AI Studio incluye:
- **Gemini 2.0 Flash:** 15 RPM / 1,500 req/día gratis
- **Embeddings:** 1,500 req/día gratis

Para una tesis con uso moderado (pruebas + experimento), el tier gratuito es suficiente.
