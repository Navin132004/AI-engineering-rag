# AI Engineering Documentation Copilot — Corpus

## Domain
Core reference and conceptual documentation for the three most common AI
engineering frameworks: **LangChain**, **Hugging Face Transformers**, and
**PyTorch**. This mirrors what an "AI Engineering Copilot" needs to answer
real developer questions across the modern AI stack — building agents,
using pretrained models, and training/fine-tuning.

- **262 documents**, ~379,000 words
- **LangChain** — 72 docs (source: `langchain-ai/docs`, `src/oss/langchain/`) — agents, tools, memory, retrieval, streaming, multi-agent, deployment
- **Hugging Face** — 119 docs (source: `huggingface/transformers`, `docs/source/en/` top-level + `main_classes/`) — pipelines, tokenizers, trainer API, quantization, task guides
- **PyTorch** — 71 docs (source: `pytorch/tutorials`, `beginner_source/` + `recipes_source/`) — tensors, autograd, DDP/distributed training, profiling, mobile deployment

All three sources are open-licensed (MIT/Apache-2.0/BSD) and pulled directly
from their official GitHub repos.

## Structure
```
corpus/
├── manifest.json          # title, word_count, source per document
├── langchain/*.mdx
├── huggingface/*.md, main_classes/*.md
└── pytorch/beginner/*.py|.rst, recipes/*.py|.rst
```

## Why this domain is good for eval questions
The three sources overlap conceptually (e.g. "how do I fine-tune a model"
touches both Hugging Face and PyTorch) but use different terminology and
APIs — this makes retrieval genuinely hard to get right, unlike a single
homogeneous doc set. Good test case: a query like *"how do I run training
across multiple GPUs"* should surface PyTorch's DDP docs AND Hugging Face's
Trainer distributed-training docs, not just one.

## RAG Pipeline

1. **Chunking** — Documents are split into ~300–500 token chunks using
   `tiktoken` with the `cl100k_base` tokenizer.
   - Total chunks: **2,028**
   - Minimum: **6 tokens**
   - Maximum: **498 tokens**
   - Average: **419 tokens**

2. **Hybrid retrieval** — BM25 and vector retrieval run independently and
   return ranked candidate chunks.

3. **Reciprocal Rank Fusion (RRF)** — BM25 and vector rankings are combined
   using `1 / (k + rank)` with `k=60`, avoiding incompatible raw-score
   normalization.

4. **Candidate selection** — The top ~20 fused chunks are passed to the
   reranker.

5. **Reranking** — A cross-encoder reranker improves precision by scoring the
   query against the retrieved candidates.

6. **Grounded generation** — The LLM is instructed to answer only from
   retrieved documentation and cite chunk IDs inline.

7. **Citation validation** — Generated answers are checked for missing or
   invalid citations before being returned.

## Evaluation

Evaluation uses a set of 30–50 question/answer pairs with known relevant
source chunks.

| Configuration | Recall@5 | Recall@20 | MRR |
|---|---:|---:|---:|
| Vector-only | TBD | TBD | TBD |
| Hybrid (BM25 + Vector) | TBD | TBD | TBD |
| Hybrid + Reranker | TBD | TBD | TBD |

### Generation Evaluation

| Configuration | Faithfulness | Answer Relevance |
|---|---:|---:|
| Vector-only | TBD | TBD |
| Hybrid | TBD | TBD |
| Hybrid + Reranker | TBD | TBD |

Evaluation metrics will be recorded from the actual evaluation set; no
placeholder numbers are presented as measured results.

