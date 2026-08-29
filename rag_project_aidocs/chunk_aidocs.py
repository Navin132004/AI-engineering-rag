"""
Chunk the AI Engineering Docs corpus (LangChain .mdx, Hugging Face .md,
PyTorch .py/.rst) into overlapping ~300-500 token chunks.

Token estimation: word-count heuristic (1 token ~= 0.75 words), same as the
Kubernetes-docs pipeline -- no network access to a real tokenizer's vocab
file in this sandbox. Swap in a real tokenizer if you have one available.
"""
import os, json, re

CORPUS_DIR = "corpus_aidocs"
MANIFEST_PATH = "corpus_aidocs/manifest.json"
OUT_PATH = "corpus_aidocs/chunks.jsonl"

TARGET_TOKENS = 400
OVERLAP_TOKENS = 60
WORDS_PER_TOKEN = 0.75
TARGET_WORDS = int(TARGET_TOKENS * WORDS_PER_TOKEN)
OVERLAP_WORDS = int(OVERLAP_TOKENS * WORDS_PER_TOKEN)


def clean_mdx(text):
    """LangChain docs: strip frontmatter, JSX components, import statements."""
    text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)
    text = re.sub(r'^import\s+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'<[A-Z]\w*[^>]*/?>', '', text)          # JSX components <Foo ... />
    text = re.sub(r'</[A-Z]\w*>', '', text)                 # closing JSX tags
    text = re.sub(r'\{\{[<%].*?[%>]\}\}', '', text, flags=re.DOTALL)
    return text


def clean_md(text):
    """Hugging Face docs: strip frontmatter, license header comment, and
    HF-specific admonition tags."""
    text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)
    text = re.sub(r'<!--Copyright.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<Tip>|</Tip>|<Youtube.*?/>', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\[.*?\]\]', '', text)  # HF anchor tags like [[autodoc]]
    return text


def clean_py_tutorial(text):
    """PyTorch tutorials: keep the module docstring (the actual prose) plus
    comments; drop raw code blocks to keep chunks prose-focused, since code
    without surrounding explanation chunks poorly for retrieval."""
    # Extract the leading triple-quoted docstring, which holds the tutorial prose
    m = re.search(r'^r?"""(.*?)"""', text, re.DOTALL)
    prose = m.group(1) if m else text
    # Also grab any ###### %% comment-cells used for inline explanation
    comments = re.findall(r'^# %%.*?\n((?:^#.*\n?)+)', text, re.MULTILINE)
    return prose + "\n\n" + "\n\n".join(c.replace("#", "").strip() for c in comments)


def clean_rst(text):
    """PyTorch recipes in RST: strip directive markup, keep prose."""
    text = re.sub(r'^\.\.\s+\w+::.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r':(\w+):`([^`]+)`', r'\2', text)  # :ref:`text` -> text
    return text


def clean_for_source(path, raw):
    if path.startswith("langchain/"):
        return clean_mdx(raw)
    if path.startswith("huggingface/"):
        return clean_md(raw)
    if path.endswith(".py"):
        return clean_py_tutorial(raw)
    if path.endswith(".rst"):
        return clean_rst(raw)
    return raw


def chunk_words(words, target, overlap):
    if len(words) <= target:
        yield words
        return
    step = target - overlap
    i = 0
    while i < len(words):
        window = words[i:i + target]
        if len(window) < overlap and i != 0:
            break
        yield window
        if i + target >= len(words):
            break
        i += step


def main():
    manifest = json.load(open(MANIFEST_PATH))
    chunks = []
    chunk_id = 0

    for entry in manifest:
        fpath = os.path.join(CORPUS_DIR, entry["path"])
        with open(fpath, encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        body = clean_for_source(entry["path"], raw)
        body = re.sub(r'\n{3,}', '\n\n', body).strip()
        words = body.split()
        if len(words) < 15:
            continue  # skip near-empty after cleaning

        for idx, window in enumerate(chunk_words(words, TARGET_WORDS, OVERLAP_WORDS)):
            chunk_text = " ".join(window)
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:05d}",
                "doc_path": entry["path"],
                "doc_title": entry["title"],
                "source": entry["source"],
                "chunk_index": idx,
                "word_count": len(window),
                "est_tokens": int(len(window) / WORDS_PER_TOKEN),
                "text": chunk_text,
            })
            chunk_id += 1

    with open(OUT_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"Docs processed: {len(manifest)}")
    print(f"Chunks created: {len(chunks)}")
    tok = [c["est_tokens"] for c in chunks]
    print(f"Avg est. tokens/chunk: {sum(tok)/len(tok):.0f}")
    print(f"Min/Max est. tokens: {min(tok)} / {max(tok)}")
    from collections import Counter
    print("Chunks by source:", Counter(c["source"] for c in chunks))


if __name__ == "__main__":
    main()
