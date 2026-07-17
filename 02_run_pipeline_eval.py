"""
02_run_pipeline_eval.py
------------------------
Ejecuta el RAGPipeline real sobre eval_dataset_with_ids.json,
capturando la respuesta generada (answer) y los fragmentos
recuperados (contexts + control_ids recuperados) para cada pregunta.

IMPORTANTE: esto hace llamadas reales a la API de OpenAI (279 preguntas x
2 llamadas al LLM cada una aprox., + 279 embeddings de consulta).
Verifica el costo estimado antes de correrlo completo. Puedes probar
primero con un subconjunto pequeño (ver LIMIT abajo).
"""

import json
import time
from utils.dataset_loader import load_dataset
from utils.rag_pipeline import RAGPipeline

INPUT_PATH  = "eval_dataset_with_ids.json"
OUTPUT_PATH = "eval_results_full.json"

# Pon un número pequeño (ej. 5) para probar primero, o None para correr todo
LIMIT = None

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

if LIMIT:
    eval_data = eval_data[:LIMIT]

print("Cargando dataset y construyendo/cargando vectorstore...")
docs = load_dataset("dataset/")  # ajusta la ruta si tu carpeta se llama distinto
pipeline = RAGPipeline()
pipeline.build_vectorstore(docs)

results = []

for i, item in enumerate(eval_data):
    question = item["question"]
    print(f"[{i+1}/{len(eval_data)}] {question}")

    try:
        result = pipeline.query(question, k=4, history=[])

        retrieved_control_ids = [
            doc.metadata.get("control_id", "") for doc in result["docs"]
        ]
        retrieved_contexts = [doc.page_content for doc in result["docs"]]

        results.append({
            "question": question,
            "ground_truth": item["ground_truth"],
            "expected_control_id": item.get("expected_control_id"),
            "answer": result["answer"],
            "contexts": retrieved_contexts,
            "retrieved_control_ids": retrieved_control_ids,
            "mode": result.get("mode", ""),
        })

    except Exception as e:
        print(f"  ERROR en esta pregunta: {e}")
        results.append({
            "question": question,
            "ground_truth": item["ground_truth"],
            "expected_control_id": item.get("expected_control_id"),
            "answer": None,
            "contexts": [],
            "retrieved_control_ids": [],
            "mode": "error",
            "error": str(e),
        })

    time.sleep(0.5)  # pequeña pausa para evitar rate limits

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nListo. Resultados guardados en {OUTPUT_PATH}")
print(f"Total procesado: {len(results)} preguntas")
