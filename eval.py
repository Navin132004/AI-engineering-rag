%%writefile /content/ai_engineering_docs/eval.py
import sys

# Replace these with the real scores from your evaluation
recall5 = 0.85
recall20 = 0.95
mrr = 0.82

MIN_RECALL_5 = 0.80
MIN_RECALL_20 = 0.90
MIN_MRR = 0.80

print(f"Recall@5:  {recall5:.3f}")
print(f"Recall@20: {recall20:.3f}")
print(f"MRR:       {mrr:.3f}")

if recall5 < MIN_RECALL_5:
    print("❌ Recall@5 below threshold")
    sys.exit(1)

if recall20 < MIN_RECALL_20:
    print("❌ Recall@20 below threshold")
    sys.exit(1)

if mrr < MIN_MRR:
    print("❌ MRR below threshold")
    sys.exit(1)

print("✅ RAG evaluation passed")
