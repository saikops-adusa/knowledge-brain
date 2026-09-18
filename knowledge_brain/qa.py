import math
import re
from collections import Counter
from typing import Iterable, List, MutableMapping

from pypdf import PdfReader

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")


def extract_text_from_pdfs(files: Iterable) -> str:
    texts: List[str] = []
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
    return "\n".join(texts).strip()


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")
    if not text.strip():
        return []
    words = text.split()
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def cosine_similarity(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot_product = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(count * count for count in left.values()))
    right_norm = math.sqrt(sum(count * count for count in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def best_chunks(question: str, chunks: List[str], top_k: int = 3) -> List[str]:
    question_tokens = Counter(tokenize(question))
    scored = []
    for chunk in chunks:
        score = cosine_similarity(question_tokens, Counter(tokenize(chunk)))
        scored.append((score, chunk))

    ranked = [chunk for score, chunk in sorted(scored, key=lambda pair: pair[0], reverse=True) if score > 0]
    return ranked[:top_k]


def answer_question(question: str, chunks: List[str]) -> str:
    matches = best_chunks(question, chunks)
    if not matches:
        return "I could not find a matching answer in the uploaded presentations. Try a more specific question."
    return "\n\n".join(matches)


def refresh_chunks(uploaded_files: Iterable, state: MutableMapping[str, List[str]]) -> List[str]:
    if uploaded_files:
        combined_text = extract_text_from_pdfs(uploaded_files)
        chunks = split_text(combined_text)
        state["chunks"] = chunks
        return chunks
    state["chunks"] = []
    return []
