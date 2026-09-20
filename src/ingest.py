import os
import io
import json
import base64
from pathlib import Path
import pymupdf  # PyMuPDF
import faiss
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Resolves base directory flexibly whether run from src/ or root
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR if (CURRENT_DIR / "data").exists() else CURRENT_DIR.parent

PDF_PATH = BASE_DIR / "data" / "agent_as_a_judge.pdf"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
IMAGES_DIR = ARTIFACTS_DIR / "images"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.json"
INDEX_PATH = ARTIFACTS_DIR / "index.faiss"

MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize OpenAI Client if key is available
api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=api_key) if api_key else None

# Optional OCR imports
try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

try:
    import easyocr
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
    HAS_EASYOCR = True
except Exception:
    HAS_EASYOCR = False


def encode_image_to_base64(image_path: Path) -> str:
    """Encodes a local PNG image file to base64 format for OpenAI Vision API."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def generate_vision_description(image_path: Path, page_number: int) -> str:
    """Sends page/figure image to GPT-4o-mini Vision to generate a detailed textual description."""
    if not openai_client:
        return ""
    
    try:
        base64_img = encode_image_to_base64(image_path)
        prompt = (
            f"Analyze this image from Page {page_number} of an AI research paper. "
            "Transcribe all text in diagrams, flowcharts, labels, tables, and figures verbatim. "
            "Specifically describe any comparisons (e.g., between LLM-as-a-Judge, Agent-as-a-Judge, and Human-as-a-Judge), "
            "highlighting key drawbacks, bottlenecks, manual effort, costs, and execution times mentioned."
        )
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                        }
                    ]
                }
            ],
            max_tokens=600,
            temperature=0
        )
        description = response.choices[0].message.content.strip()
        return description
    except Exception as e:
        print(f"Vision API processing skipped for page {page_number}: {str(e)}")
        return ""


def perform_ocr_on_page(page, dpi=150) -> str:
    """Fallback OCR on PDF page pixmap."""
    ocr_text = ""
    try:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        if HAS_PYTESSERACT:
            try:
                ocr_text = pytesseract.image_to_string(img).strip()
            except Exception:
                ocr_text = ""

        if not ocr_text and HAS_EASYOCR:
            try:
                img_np = np.array(img)
                results = easyocr_reader.readtext(img_np, detail=0)
                ocr_text = " ".join(results).strip()
            except Exception:
                ocr_text = ""
    except Exception as e:
        print(f"OCR failed for page: {str(e)}")

    return ocr_text


def extract_pages(pdf_path: Path):
    """Extracts text, saves page images, and generates AI Vision figure descriptions."""
    document = pymupdf.open(str(pdf_path))
    pages = []
    IMAGES_DIR.mkdir(exist_ok=True, parents=True)

    for page_number, page in enumerate(document, start=1):
        standard_text = page.get_text("text").strip()
        ocr_text = perform_ocr_on_page(page)

        # Save page image for UI display & Vision LLM
        pix = page.get_pixmap(dpi=150)
        image_filename = f"page_{page_number}.png"
        image_path = IMAGES_DIR / image_filename
        pix.save(str(image_path))

        # Generate Vision LLM description for diagrams
        vision_description = generate_vision_description(image_path, page_number)

        combined_text_parts = []
        if standard_text:
            combined_text_parts.append(standard_text)
        
        if ocr_text and ocr_text not in standard_text:
            combined_text_parts.append(f"\n--- [OCR Diagram Content Page {page_number}] ---\n{ocr_text}")

        if vision_description:
            combined_text_parts.append(f"\n--- [AI Vision Figure/Diagram Description Page {page_number}] ---\n{vision_description}")

        full_page_text = "\n\n".join(combined_text_parts).strip()

        if full_page_text:
            pages.append({
                "page": page_number,
                "text": full_page_text,
                "image_path": f"artifacts/images/{image_filename}",
                "vision_description": vision_description
            })

    return pages


def chunk_text(text, chunk_size=300, overlap=50):
    """Splits text into overlapping word chunks for vector retrieval."""
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
    print(f"Processing PDF, rendering page images, running OCR & AI Vision on {PDF_PATH.name}...")
    pages = extract_pages(PDF_PATH)
    all_chunks = []

    for page in pages:
        chunks = chunk_text(page["text"])
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"p{page['page']}_c{chunk_index}",
                "page": page["page"],
                "text": chunk,
                "image_path": page["image_path"],
                "has_vision_data": bool(page["vision_description"])
            })

    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

    # Save chunks JSON
    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"Created {len(all_chunks)} chunks with multimodal image associations.")

    # Generate Embeddings & Build FAISS Index
    print("Generating vector embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved FAISS vector index to {INDEX_PATH}")


if __name__ == "__main__":
    build_index_and_chunks()