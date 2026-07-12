# 🔐 Agente ISO/IEC 27002 — README

**Proyecto de tesis:** Experiencia de usuario: Agente conversacional para el aprendizaje de la norma ISO/IEC 27002 referido a temas de seguridad de la información para estudiantes universitarios
**Autor:** Edson Gerardo Vasquez Solis — Universidad de Lima

---

## 📁 Estructura del proyecto

```
iso27002_agent/
│
├── app.py                  ← Aplicación principal (Streamlit)
├── config.py               ← Configuración y variables (NO incluye API key real)
├── reindex.py               ← Script para reconstruir el vectorstore manualmente
├── requirements.txt        ← Dependencias
│
├── utils/
│   ├── rag_pipeline.py     ← Pipeline RAG (LangChain + OpenAI GPT-4o-mini + ChromaDB)
│   ├── dataset_loader.py   ← Cargador del dataset JSON/TXT
│   └── session.py          ← Manejo de sesión Streamlit
│
├── dataset/                ← 📌 AQUÍ va tu dataset
│   └── ejemplo_estructura2.json  ← Dataset con los 93 controles ISO/IEC 27002
│
└── vectorstore/             ← Se crea automáticamente al indexar (ChromaDB). No se incluye en el repositorio.
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

### 3. Obtener API Key de OpenAI
1. Ve a: https://platform.openai.com/api-keys
2. Crea una nueva API Key
3. Copia la key (no la compartas ni la subas a ningún repositorio)

### 4. Configurar la API Key
Crea un archivo `.env` en la raíz del proyecto (este archivo **no debe subirse a GitHub**):
```
OPENAI_API_KEY=tu_api_key_aqui
```

### 5. Agregar tu dataset
Coloca tu archivo JSON en la carpeta `dataset/`. Ver estructura abajo.

### 6. Correr la aplicación
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

Si cambias el dataset, elimina la carpeta `vectorstore/` y vuelve a correr la app (o usa `reindex.py`):
```bash
rm -rf vectorstore/          # macOS/Linux
rmdir /s vectorstore          # Windows

python reindex.py
# o simplemente:
streamlit run app.py
```

---

## 🧪 Evaluación del sistema

El proyecto incluye scripts adicionales para evaluar el desempeño del agente mediante métricas cuantitativas:

| Métrica | Herramienta | Qué mide |
|---|---|---|
| Recall@k, MRR@k | Script propio | Precisión de la fase de recuperación |
| Faithfulness, Answer Relevancy, Answer Correctness | RAGAS | Calidad de la fase de generación |
| Usabilidad percibida | Escala SUS | Experiencia de usuario (aplicada a estudiantes) |

Estos scripts de evaluación no forman parte del flujo principal de la aplicación y se ejecutan de forma independiente sobre un conjunto de preguntas de prueba.

---

## 💰 Costos estimados (API OpenAI)

El uso de este proyecto genera costos reales asociados a:
- **GPT-4o-mini:** costo por token de entrada/salida en cada consulta del estudiante.
- **text-embedding-3-small:** costo por token al generar embeddings del dataset y de cada consulta.

Para una tesis con uso moderado (indexación del dataset + pruebas + experimentación con un grupo reducido de estudiantes), el costo total es bajo, pero **no es gratuito** como ocurre con otros proveedores que ofrecen tiers gratuitos. Se recomienda establecer límites de gasto (*usage limits*) desde el panel de OpenAI para evitar cargos inesperados.

---

## ⚠️ Seguridad

- Nunca subas tu archivo `.env` ni tu API key a este repositorio.
- Verifica que `config.py` no contenga ninguna key hardcodeada como valor por defecto.
- La carpeta `vectorstore/` no se incluye en el repositorio, ya que se regenera automáticamente a partir del dataset.
