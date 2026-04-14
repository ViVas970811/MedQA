"""Build and verify the FAISS vector index."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medqa.config import get_settings
from medqa.data.loader import DataLoader
from medqa.models.embeddings import EmbeddingModel
from medqa.data.vectorstore import VectorStore


def main() -> None:
    settings = get_settings()
    loader = DataLoader(settings)
    embedder = EmbeddingModel(settings)
    store = VectorStore()

    corpus = loader.load_corpus()
    print(f"Loaded {len(corpus)} corpus questions")

    start = time.perf_counter()
    embeddings = embedder.encode(corpus, show_progress=True)
    store.build(corpus, embeddings)
    elapsed = time.perf_counter() - start

    print(f"Index built: {store.size} vectors in {elapsed:.1f}s")

    # Quick sanity check
    test_q = "Why does my chest feel tight when I wake up?"
    query_emb = embedder.encode_query(test_q)
    results = store.search(query_emb, k=3)

    print(f"\nTest query: {test_q}")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.question} (L2={r.score:.2f})")


if __name__ == "__main__":
    main()
