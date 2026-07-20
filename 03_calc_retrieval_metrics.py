

import json

INPUT_PATH = "eval_results_full_expB.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

recalls = []
reciprocal_ranks = []
sin_expected = 0

for item in results:
    expected = item.get("expected_control_id")
    retrieved = item.get("retrieved_control_ids", [])

    if not expected:
        sin_expected += 1
        continue

 
    hit = expected in retrieved
    recalls.append(1 if hit else 0)


    if hit:
        rank = retrieved.index(expected) + 1
        reciprocal_ranks.append(1 / rank)
    else:
        reciprocal_ranks.append(0)

recall_at_k = sum(recalls) / len(recalls) if recalls else 0
mrr_at_k = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

print(f"Preguntas evaluadas: {len(recalls)}")
print(f"Preguntas sin expected_control_id (excluidas): {sin_expected}")
print(f"\nRecall@k: {recall_at_k:.4f}")
print(f"MRR@k:    {mrr_at_k:.4f}")

# Guarda un resumen
with open("retrieval_metrics_summary_expB.json", "w", encoding="utf-8") as f:
    json.dump({
        "recall_at_k": recall_at_k,
        "mrr_at_k": mrr_at_k,
        "n_evaluated": len(recalls),
        "n_excluded_no_expected_id": sin_expected,
    }, f, indent=2, ensure_ascii=False)

print("\nResumen guardado en retrieval_metrics_summary_expB.json")
