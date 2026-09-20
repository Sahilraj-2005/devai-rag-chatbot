import time
import os
import streamlit as st
from rag import answer_question

# Page Configuration
st.set_page_config(
    page_title="DevAI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration & Preset Questions
with st.sidebar:
    st.title("⚙️ Controls & Benchmark Tests")
    st.markdown("---")
    
    top_k = st.slider("Retrieved Chunks (Top-K)", min_value=1, max_value=10, value=5)
    
    st.markdown("### 💡 Quick Test Questions")
    sample_questions = [
        "Select a sample question...",
        "What percentage of evaluation time and cost does Agent-as-a-Judge save?",
        "What was MetaGPT's exact Task Solve Rate?",
        "Which framework was the most expensive per task?",
        "What is the DevAI dataset and how many tasks does it contain?",
        "Which search algorithm configuration gave the highest alignment rate?",
        "In the diagram comparing LLM-as-a-Judge, Agent-as-a-Judge, and Human-as-a-Judge, what key drawback is highlighted for Human-as-a-Judge?"
    ]
    
    selected_sample = st.selectbox("Choose a preset question:", sample_questions)
    
    st.markdown("---")
    st.markdown("**Paper:** *Agent-as-a-Judge: Evaluate Agents with Agents*")
    st.markdown("**Benchmark:** DevAI (55 Tasks)")

# Main Title & Subtitle
st.title("🤖 DevAI Agent-as-a-Judge RAG Chatbot")
st.caption("Hybrid RAG pipeline combining FAISS vector search, BM25 keyword matching, Multimodal AI Vision, and LLM reasoning.")

# Set input field value based on sample selection
default_query = "" if selected_sample == "Select a sample question..." else selected_sample

question = st.text_input(
    "Enter your question:",
    value=default_query,
    placeholder="What percentage of evaluation time does Agent-as-a-Judge save?"
)

col_ask, col_clear = st.columns([1, 5])
with col_ask:
    submit_button = st.button("🚀 Ask", use_container_width=True)

if submit_button and question.strip():
    start_time = time.time()
    
    with st.spinner("Retrieving multimodal evidence chunks and generating answer..."):
        try:
            try:
                result = answer_question(question, top_k=top_k)
            except TypeError:
                result = answer_question(question)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.stop()
            
    elapsed_time = time.time() - start_time

    # Display Answer and Retrieved Context in Tabs
    tab_answer, tab_sources = st.tabs(["💬 Answer", "📚 Retrieved Evidence Chunks"])

    with tab_answer:
        st.caption(f"⚡ Latency: `{elapsed_time:.2f} seconds`")
        st.markdown("---")
        st.subheader("Answer")
        st.markdown(result.get("answer", "No answer returned."))

    with tab_sources:
        sources = result.get("sources", [])
        st.subheader(f"Retrieved Evidence Chunks ({len(sources)})")
        
        if not sources:
            st.info("No evidence chunks were retrieved for this query.")
        else:
            for idx, source in enumerate(sources, 1):
                page = source.get("page", source.get("metadata", {}).get("page", "N/A"))
                chunk_id = source.get("chunk_id", source.get("id", idx))
                score = source.get("score", 0.0)
                text_content = source.get("text", source.get("content", source.get("page_content", "")))
                
                # Check for associated image path from our AI Vision ingestion
                image_path = source.get("image_path", "")

                title = f"Chunk #{idx} | Page: {page} | ID: {chunk_id} | Score: {score:.4f}"
                
                with st.expander(title):
                    # If an image exists for this page, display it inside the expander
                    if image_path and os.path.exists(image_path):
                        st.image(image_path, caption=f"Source Figure - Page {page}", use_container_width=True)
                    
                    st.text_area(
                        label=f"text_{idx}",
                        value=text_content,
                        height=140,
                        disabled=True,
                        label_visibility="collapsed"
                    )