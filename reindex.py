"""
reindex.py
──────────
Script auxiliar para re-indexar el dataset desde cero.
Útil cuando agregas nuevos controles o modificas el dataset.

Uso:
    python reindex.py
"""

import shutil
import os
from config import CHROMA_PERSIST_DIR
from utils.dataset_loader import load_dataset, validate_dataset
from utils.rag_pipeline import RAGPipeline


def main():
    print("=" * 55)
    print("  Agente ISO/IEC 27002 — Re-indexación del dataset")
    print("=" * 55)

    # 1. Validar dataset
    print("\n📋 Validando dataset…")
    report = validate_dataset("dataset/")

    if report["errors"]:
        print("\n❌ Errores encontrados:")
        for err in report["errors"]:
            print(f"   • {err}")
        return

    if report["warnings"]:
        print("\n⚠️  Advertencias:")
        for w in report["warnings"]:
            print(f"   • {w}")

    total = report["total_items"]
    print(f"\n   JSON: {len(report['json_files'])} archivo(s)")
    print(f"   TXT:  {len(report['txt_files'])} archivo(s)")
    print(f"   Total de registros: {total}")

    if total == 0:
        print("\n❌ No hay registros en el dataset. Agrega contenido primero.")
        return

    # 2. Eliminar vectorstore anterior
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"\n🗑️  Eliminando vectorstore anterior ({CHROMA_PERSIST_DIR})…")
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print("   ✅ Eliminado")

    # 3. Cargar documentos
    print("\n📂 Cargando documentos…")
    docs = load_dataset("dataset/")

    # 4. Indexar
    print("\n🔨 Generando embeddings e indexando en ChromaDB…")
    print("   (Esto puede tomar unos minutos según el tamaño del dataset)\n")
    pipeline = RAGPipeline()
    pipeline.build_vectorstore(docs)

    count = pipeline.get_doc_count()
    print(f"\n✅ Re-indexación completada.")
    print(f"   Fragmentos indexados en ChromaDB: {count}")
    print(f"\n▶  Ahora puedes correr la app con:  streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
