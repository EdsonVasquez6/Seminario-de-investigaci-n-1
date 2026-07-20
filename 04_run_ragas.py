
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness

INPUT_PATH = "eval_results_full_expB.json"

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
df.to_csv("ragas_results_detailed_expB.csv", index=False, encoding="utf-8")

summary = {
    "faithfulness_mean": float(df["faithfulness"].mean()),
    "answer_relevancy_mean": float(df["answer_relevancy"].mean()),
    "answer_correctness_mean": float(df["answer_correctness"].mean()),
    "n_evaluated": len(valid),
}
