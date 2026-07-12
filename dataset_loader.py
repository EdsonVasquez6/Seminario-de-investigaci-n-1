"""
dataset_loader.py
─────────────────
Carga el dataset de ISO/IEC 27002 desde la carpeta /dataset/
Soporta los formatos:
  - JSON  (recomendado para estructurar controles)
  - TXT   (un archivo por control o sección)

Estructura JSON recomendada:
[
  {
    "control_id":   "5.1",
    "title":        "Políticas para la seguridad de la información",
    "domain":       "Controles Organizacionales",
    "type":         "Preventivo",
    "content":      "Texto completo del control...",
    "keywords":     ["política", "seguridad", "gestión"]
  },
  ...
]

"""

import os
import json
from typing import List
from langchain_core.documents import Document


def load_dataset(dataset_dir: str = "dataset/") -> List[Document]:
    """
    Carga todos los archivos del directorio dataset/ y los convierte
    en objetos Document de LangChain con metadatos enriquecidos.

    Parámetros:
        dataset_dir : Ruta a la carpeta que contiene los archivos del dataset

    Retorna:
        Lista de Document listos para ser indexados en ChromaDB
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"❌ No se encontró la carpeta: '{dataset_dir}'\n"
            f"   Crea la carpeta y agrega tus archivos JSON o TXT."
        )

    documents: List[Document] = []
    files = os.listdir(dataset_dir)

    if not files:
        raise ValueError(
            f"❌ La carpeta '{dataset_dir}' está vacía.\n"
            f"   Agrega tu dataset de ISO/IEC 27002 en formato JSON o TXT."
        )

    json_files = [f for f in files if f.endswith(".json")]
    txt_files  = [f for f in files if f.endswith(".txt")]

    # ── Cargar archivos JSON ──────────────────────────────────────────────────
    for filename in json_files:
        path = os.path.join(dataset_dir, filename)
        docs = _load_json(path, filename)
        documents.extend(docs)
        print(f"   ✅ JSON  [{filename}]: {len(docs)} registros cargados")

    # ── Cargar archivos TXT ───────────────────────────────────────────────────
    for filename in txt_files:
        path = os.path.join(dataset_dir, filename)
        docs = _load_txt(path, filename)
        documents.extend(docs)
        print(f"   ✅ TXT   [{filename}]: {len(docs)} fragmentos cargados")

    if not documents:
        raise ValueError(
            "❌ No se pudieron cargar documentos. "
            "Verifica que los archivos sean .json o .txt."
        )

    print(f"\n📚 Total: {len(documents)} documentos cargados desde '{dataset_dir}'")
    return documents


# ── Cargador JSON ─────────────────────────────────────────────────────────────
def _load_json(path: str, filename: str) -> List[Document]:
    """
    Carga un archivo JSON con la estructura de controles ISO 27002.
    Acepta tanto una lista [] como un objeto con clave 'controls' o 'data'.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalizar estructura del JSON
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # Buscar la lista dentro de claves comunes
        for key in ["controls", "data", "items", "norma", "iso27002"]:
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
        else:
            # Si no hay lista anidada, tratar el dict como un solo item
            items = [data]
    else:
        return []

    documents = []
    for item in items:
        if not isinstance(item, dict):
            continue

        # Construir el texto del documento
        content = _build_content(item)
        if not content.strip():
            continue

        # Metadatos para trazabilidad
        metadata = {
            "source":      filename,
            "control_id":  str(item.get("control_id", item.get("id", ""))),
            "title":       item.get("title", item.get("titulo", "")),
            "domain":      item.get("domain", item.get("dominio", "")),
            "type":        item.get("type",   item.get("tipo", "")),
            "keywords":    ", ".join(item.get("keywords", item.get("palabras_clave", []))),
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def _build_content(item: dict) -> str:
    """
    Construye el texto completo de un control a partir de los campos JSON.
    Funciona con claves en inglés o español.
    """
    parts = []

    # Control ID y título
    control_id = item.get("control_id", item.get("id", ""))
    title      = item.get("title",      item.get("titulo", ""))
    if control_id and title:
        parts.append(f"Control {control_id}: {title}")
    elif title:
        parts.append(f"Control: {title}")

    # Dominio
    domain = item.get("domain", item.get("dominio", ""))
    if domain:
        parts.append(f"Dominio: {domain}")

    # Tipo de control
    ctrl_type = item.get("type", item.get("tipo", ""))
    if ctrl_type:
        parts.append(f"Tipo: {ctrl_type}")

    # Contenido principal
    for key in ["content", "contenido", "description", "descripcion",
                "definition", "definicion", "text", "texto", "body"]:
        val = item.get(key, "")
        if val:
            parts.append(str(val))
            break

    # Objetivo del control
    for key in ["objective", "objetivo", "purpose", "proposito"]:
        val = item.get(key, "")
        if val:
            parts.append(f"Objetivo: {val}")
            break

    # Implementación
    for key in ["implementation", "implementacion", "guidance", "guia"]:
        val = item.get(key, "")
        if val:
            parts.append(f"Implementación: {val}")
            break

    # Ejemplos
    for key in ["examples", "ejemplos", "example", "ejemplo"]:
        val = item.get(key, "")
        if val:
            if isinstance(val, list):
                val = "; ".join(str(v) for v in val)
            parts.append(f"Ejemplos: {val}")
            break

    # Keywords como texto adicional para mejorar recuperación
    for key in ["keywords", "palabras_clave", "tags"]:
        val = item.get(key, [])
        if val:
            if isinstance(val, list):
                val = ", ".join(val)
            parts.append(f"Conceptos clave: {val}")
            break

    return "\n".join(parts)


# ── Cargador TXT ──────────────────────────────────────────────────────────────
def _load_txt(path: str, filename: str) -> List[Document]:
    """
    Carga un archivo TXT completo como un solo documento.
    El nombre del archivo se usa como fuente en los metadatos.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        return []

    # Nombre limpio del archivo para metadatos
    clean_name = filename.replace(".txt", "").replace("_", " ").title()

    return [
        Document(
            page_content=content,
            metadata={
                "source":     filename,
                "control_id": clean_name,
                "title":      clean_name,
                "domain":     "ISO/IEC 27002",
            }
        )
    ]


# ── Función de validación del dataset ────────────────────────────────────────
def validate_dataset(dataset_dir: str = "dataset/") -> dict:
    """
    Valida la estructura del dataset y retorna un reporte.
    Útil para depuración antes de indexar.
    """
    report = {
        "json_files": [],
        "txt_files":  [],
        "total_items": 0,
        "errors": [],
        "warnings": [],
    }

    if not os.path.exists(dataset_dir):
        report["errors"].append(f"Carpeta no encontrada: {dataset_dir}")
        return report

    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)

        if filename.endswith(".json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                count = len(data) if isinstance(data, list) else 1
                report["json_files"].append({"file": filename, "items": count})
                report["total_items"] += count
            except json.JSONDecodeError as e:
                report["errors"].append(f"JSON inválido en {filename}: {e}")

        elif filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            words = len(content.split())
            report["txt_files"].append({"file": filename, "words": words})
            if words < 50:
                report["warnings"].append(
                    f"{filename} tiene muy poco texto ({words} palabras)"
                )

    return report
