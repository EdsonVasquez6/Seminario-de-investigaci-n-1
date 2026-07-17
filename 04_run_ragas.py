"""
04_run_ragas.py
------------------------------
Calcula Faithfulness, Answer Relevancy y Answer Correctness (RAGAS)
a partir de eval_results_full.json (generado por 02_run_pipeline_eval.py).

Requiere: pip install ragas datasets langchain-openai --break-system-packages
"""

import json
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness

load_dotenv()

INPUT_PATH = "eval_results_full.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

# Filtra entradas que sí tuvieron respuesta (excluye errores)
valid = [r for r in results if r.get("answer") and r.get("contexts")]
print(f"Entradas válidas para RAGAS: {len(valid)} / {len(results)}")

data = {
    "question": [r["question"] for r in valid],
    "answer": [r["answer"] for r in valid],
    "contexts": [r["contexts"] for r in valid],
    "ground_truth": [r["ground_truth"] for r in valid],
}

dataset = Dataset.from_dict(data)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, answer_correctness],
)

print("\nResultados RAGAS:")
print(result)

df = result.to_pandas()
df.to_csv("ragas_results_detailed.csv", index=False, encoding="utf-8")

summary = {
    "faithfulness_mean": float(df["faithfulness"].mean()),
    "answer_relevancy_mean": float(df["answer_relevancy"].mean()),
    "answer_correctness_mean": float(df["answer_correctness"].mean()),
    "n_evaluated": len(valid),
}

with open("ragas_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nResumen guardado en ragas_summary.json")
print("Detalle por pregunta guardado en ragas_results_detailed.csv")
