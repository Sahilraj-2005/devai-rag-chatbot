import os
from dotenv import load_dotenv
from openai import OpenAI
from retriever import Retriever

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

client = OpenAI(api_key=api_key)
retriever = Retriever()


SYSTEM_PROMPT = """You are a precision AI assistant specializing in evaluating AI research papers.

Follow these strict rules when answering questions based on the provided context:
1. EXTRACT: Read all provided context chunks carefully, including standard text, OCR content, and AI Vision figure/diagram descriptions. Extract and list all relevant raw facts, diagram labels, comparisons, names, percentages, and metrics.
2. COMPARE & EVALUATE: When asked for extremes (e.g., "most expensive", "highest alignment", "total count") or specific drawbacks in diagrams (e.g., Human-as-a-Judge vs. Agent-as-a-Judge), evaluate ALL extracted information across all chunks before drawing a conclusion.
3. MATH & CONVERSIONS: State raw numbers first before calculating percentages or ratios step-by-step. Double-check that numerators and denominators are assigned correctly.
4. GROUNDING: Answer strictly using facts from the provided context (including AI Vision figure descriptions and diagram text). If a query asks about a diagram, use the Vision/OCR descriptions in the chunks. Only if there is genuinely no relevant information in the context, respond with: "I could not find sufficient evidence in the document."
5. SYNTHESIZE: Provide a clear, direct final answer."""


def build_user_prompt(question, retrieved_chunks):
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            f"[Page {chunk['page']} | ID: {chunk['chunk_id']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"""Context Information:
---------------------
{context}
---------------------

Question: {question}

Please answer the question step-by-step following your system guidelines."""


def answer_question(question, top_k=5):
    # Retrieve top_k chunks using hybrid search (FAISS + BM25)
    retrieved_chunks = retriever.search(question, top_k=top_k)
    user_prompt = build_user_prompt(question, retrieved_chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,  # Deterministic output for precision QA
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
                "text": chunk["text"],
                "image_path": chunk.get("image_path", "")
            }
            for chunk in retrieved_chunks
        ]
    }