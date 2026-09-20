from pathlib import Path
import json
import pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Resolves to the devai-rag-chatbot root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Update paths to use BASE_DIR to avoid relative path errors
PDF_PATH = BASE_DIR / "data" / "agent_as_a_judge.pdf"
CHUNKS_PATH = BASE_DIR / "artifacts" / "chunks.json"
INDEX_PATH = BASE_DIR / "artifacts" / "index.faiss"

MODEL_NAME = "all-MiniLM-L6-v2"

def extract_pages(pdf_path):
    document = pymupdf.open(str(pdf_path))
    pages = []
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_number, "text": text})
    return pages

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks

def build_index_and_chunks():
    pages = extract_pages(PDF_PATH)
    all_chunks = []

    for page in pages:
        chunks = chunk_text(page["text"])
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"p{page['page']}_c{chunk_index}",
                "page": page["page"],
                "text": chunk
            })

    CHUNKS_PATH.parent.mkdir(exist_ok=True, parents=True)
    
    # Save chunks JSON
    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"Created {len(all_chunks)} chunks.")

    # Generate Embeddings & Build FAISS Index
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved FAISS index to {INDEX_PATH}")

if __name__ == "__main__":
    build_index_and_chunks()