# 🤖 DevAI Agent-as-a-Judge Multimodal RAG Chatbot

## Project Overview
This repository contains a specialized Multimodal Retrieval-Augmented Generation (RAG) chatbot engineered to evaluate and query the *Agent-as-a-Judge: Evaluate Agents with Agents* research paper. Developed by Sahil Raj (Roll No. 221MN045, National Institute of Technology Karnataka), this system implements a hybrid search architecture combining FAISS dense vector retrieval, BM25 sparse keyword matching, OCR diagram processing, and OpenAI GPT-4o-mini Vision processing.

The chatbot extracts complex tabular data, mathematical derivations, exact benchmark metrics, and visual diagrams (such as flowcharts and comparative figures) with deterministic accuracy and source chunk citations.

## System Architecture Workflow
Below is the data pipeline and multimodal RAG architecture, covering document ingestion, figure vision extraction, hybrid chunk retrieval, and evidence-grounded generation:

![Multimodal DevAI RAG Chatbot Workflow](workflow.png)

### Architectural Highlights
1. **Multimodal Document Ingestion:** Uses PyMuPDF to extract text layers, OCR (`pytesseract` / `easyocr`) to capture embedded diagram text, and `gpt-4o-mini` Vision to describe complex figures.
2. **Hybrid Retrieval (Dense + Sparse):** Integrates SentenceTransformer (`all-MiniLM-L6-v2`) vector embeddings with BM25 keyword indexing, fused via Reciprocal Rank Fusion (RRF).
3. **Guardrailed Chain-of-Thought Reasoning:** Uses a structured prompt requiring multi-step extraction, math verification, and entity comparisons before generating answers.
4. **Interactive Evidence Viewer:** Built with Streamlit, displaying latency, retrieved text chunks, RRF similarity scores, and source diagram images.

## Installation & Execution Setup

### 1. Prerequisites
Ensure you have Python 3.9 or higher installed. Activate your virtual environment:

```bash
# On Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1