from pathlib import Path
import json
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Resolves to devai-rag-chatbot root folder
BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "artifacts" / "chunks.json"
INDEX_PATH = BASE_DIR / "artifacts" / "index.faiss"

MODEL_NAME = "all-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words for BM25 keyword matching."""
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

        if not CHUNKS_PATH.exists() or not INDEX_PATH.exists():
            raise FileNotFoundError("Run 'python src/ingest.py' first to build artifacts.")

        self.chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(INDEX_PATH))

        # Build BM25 sparse index over text content for keyword matching
        corpus = [
            tokenize(chunk.get("text", chunk.get("content", str(chunk))))
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 5, candidate_k: int = 20):
        """
        Executes a hybrid search combining FAISS dense vector search 
        and BM25 sparse keyword search using Reciprocal Rank Fusion (RRF).
        """
        # 1. Dense Search (FAISS)
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.asarray(query_embedding, dtype="float32")
        dense_scores, dense_indices = self.index.search(query_embedding, candidate_k)

        dense_ranks = []
        for index in dense_indices[0]:
            if index != -1:
                dense_ranks.append(int(index))

        # 2. Sparse Search (BM25)
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sorted_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
        
        sparse_ranks = [
            int(idx) for idx in sorted_bm25_indices if bm25_scores[idx] > 0
        ]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        rrf_k = 60  # Standard RRF constant

        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, idx in enumerate(sparse_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort combined results by highest RRF score
        fused_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        results = []
        for idx in fused_indices:
            result = self.chunks[idx].copy()
            result["score"] = float(rrf_scores[idx])
            results.append(result)

        return results