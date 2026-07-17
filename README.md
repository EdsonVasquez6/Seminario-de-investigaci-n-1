

## 📁 Estructura del proyecto

```
iso27002_agent/
│
├── app.py                  ← Aplicación principal (Streamlit)
├── config.py               ← Configuración y variables 
├── reindex.py               ← Script para reconstruir el vectorstore manualmente
├── requirements.txt        ← Dependencias
│
├── utils/
│   ├── rag_pipeline.py     ← Pipeline RAG (LangChain + OpenAI GPT-4o-mini + ChromaDB)
│   ├── dataset_loader.py   ← Cargador del dataset JSON/TXT
│   └── session.py          ← Manejo de sesión Streamlit
│
├── dataset/               
│   └── ejemplo_estructura2.json  ← Dataset con los 93 controles ISO/IEC 27002
│
└── vectorstore/             ← Se crea automáticamente al indexar (ChromaDB).
```

---

## ⚡ Instalación rápida

### 1. Crear y activar un entorno virtual (recomendado)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar la API Key
Crear un archivo `.env` en la raíz del proyecto:
```
OPENAI_API_KEY=tu_api_key_aqui
```

### 4. Agregar tu dataset
Coloca el archivo JSON en la carpeta `dataset/`. 

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
Clasificación de intención (informativo / práctico / socrático)
      ↓
Recuperación híbrida:
   - Si la pregunta referencia un control explícito → filtro por metadata (control_id)
   - Si no → búsqueda semántica en ChromaDB (similitud coseno) con embeddings de OpenAI
      ↓
Top-K fragmentos recuperados de ISO/IEC 27002
      ↓
Selección del prompt según el modo (informativo, constructivista o retroalimentación socrática)
      ↓
GPT-4o-mini genera la respuesta
      ↓
Respuesta mostrada al estudiante (con estrategia socrática si aplica)
```

---

## ⚙️ Parámetros configurables (config.py)

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| GENERATION_MODEL | gpt-4o-mini | LLM para generar respuestas |
| EMBEDDING_MODEL | text-embedding-3-small | Modelo de embeddings de OpenAI |
| CHUNK_SIZE | 800 | Tamaño de fragmentos en caracteres |
| CHUNK_OVERLAP | 150 | Solapamiento entre fragmentos |
| TEMPERATURE | 0.2 | Creatividad del modelo (0=exacto, 1=creativo) |
| DEFAULT_K | 4 | Fragmentos recuperados por consulta |
| CHROMA_PERSIST_DIR | vectorstore/ | Carpeta donde se persiste la base vectorial |
| CHROMA_COLLECTION | iso27002_knowledge_base | Nombre de la colección en ChromaDB |

---

## 🗄️ Re-indexar el dataset

Si se cambia el dataset, elimina la carpeta `vectorstore/` y volver a correr la app (o usar `reindex.py`):
```bash
rm -rf vectorstore/          # macOS/Linux
rmdir /s vectorstore          # Windows

python reindex.py
# o simplemente:
streamlit run app.py
```

---

## 🧪 Evaluación del sistema


| Métrica | Herramienta | Qué mide |
|---|---|---|
| Recall@k, MRR@k | Script propio | Precisión de la fase de recuperación |
| Faithfulness, Answer Relevancy, Answer Correctness | RAGAS | Calidad de la fase de generación |
| Usabilidad percibida | Escala SUS | Experiencia de usuario (aplicada a estudiantes) |

Estos scripts de evaluación no forman parte del flujo principal de la aplicación y se ejecutan de forma independiente sobre un conjunto de preguntas de prueba.

---

