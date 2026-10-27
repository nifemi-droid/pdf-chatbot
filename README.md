# 📄 Chat with your PDF (RAG Application)

A full-stack AI application that allows users to upload a PDF and have a conversation with it. Built as a portfolio piece to demonstrate hands-on Technical AI PM skills, moving from architectural design to production-ready code.

## 🏗️ Architecture

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to ensure the AI only answers based on the uploaded document, minimizing hallucinations.

*   **Frontend:** Streamlit (Python) for a fast, interactive chat UI.
*   **Backend:** FastAPI (Python) handling REST API requests.
*   **Vector Database:** ChromaDB (in-memory) for storing and querying document embeddings.
*   **Embeddings:** `all-MiniLM-L6-v2` via HuggingFace (runs locally on CPU).
*   **LLM:** Groq API using `openai/gpt-oss-20b`.

## 🧠 The PM Decision: Why the 20B Model?

During the evaluation phase, I benchmarked `gpt-oss-120b` (the larger model) against `gpt-oss-20b` (the smaller, cheaper model) on a custom "Golden Dataset" of product queries. 

*   **Latency:** Both models averaged ~1.1 seconds on Groq's infrastructure.
*   **Quality:** Both models scored 3/3 on the evaluation harness.
*   **Cost:** The 20B model is approximately **90% cheaper** per 1M tokens.

**Decision:** Because the 20B model met the quality and speed requirements at a fraction of the cost, it was selected for production to maximize ROI.

## 🚀 How to Run Locally

### Prerequisites
*   Python 3.9+
*   A free [Groq API Key](https://console.groq.com/)

### 1. Clone the repository & setup
```bash
git clone https://github.com/nifemi-droid/pdf-chatbot.git
cd pdf-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt