"""Short demo of overlapping chunking for RAG-style retrieval."""

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    text: str
    start: int
    end: int


def chunk_with_overlap(text: str, chunk_size: int = 80, overlap: int = 20) -> List[Chunk]:
    """Character-based chunking with overlap."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks: List[Chunk] = []
    step = chunk_size - overlap
    start = 0
    idx = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        snippet = text[start:end].strip()
        if snippet:
            chunks.append(Chunk(f"c{idx}", snippet, start, end))
            idx += 1
        start += step

    return chunks


def naive_retrieve(chunks: List[Chunk], query: str) -> List[Chunk]:
    """Tiny stand-in for retrieval: return chunks containing the query string."""
    return [c for c in chunks if query.lower() in c.text.lower()]


def main() -> None:
    # The key phrase crosses a boundary near the middle.
    doc = (
        "RAG systems often split long documents into chunks. "
        "A tricky case happens when a key phrase like brown fox jumps "
        "is split across two chunks without overlap. "
        "Overlap helps keep boundary context so retrieval still works."
    )
    query = "brown fox jumps"

    print("=== Without overlap ===")
    no_overlap = chunk_with_overlap(doc, chunk_size=80, overlap=0)
    hits = naive_retrieve(no_overlap, query)
    print(f"chunks: {len(no_overlap)}, hits: {len(hits)}")
    for c in hits:
        print(f"{c.chunk_id}: {c.text}")

    print("\n=== With overlap ===")
    with_overlap = chunk_with_overlap(doc, chunk_size=80, overlap=20)
    hits = naive_retrieve(with_overlap, query)
    print(f"chunks: {len(with_overlap)}, hits: {len(hits)}")
    for c in hits:
        print(f"{c.chunk_id}: {c.text}")


if __name__ == "__main__":
    main()
