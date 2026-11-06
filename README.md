# PDF Chatbot

A chatbot that answers questions about PDF documents using Retrieval-Augmented Generation (RAG).

## Features
- Extracts text from PDFs, including scanned pages, using OCR (Tesseract)
- Splits documents into chunks and embeds them with Hugging Face sentence-transformers
- Hybrid retrieval: FAISS (semantic search) combined with BM25 (keyword search)
- Generates answers with Groq-hosted LLMs through LangChain

## Tech Stack
Python, LangChain, FAISS, BM25, sentence-transformers, Tesseract, Groq

## Setup
1. Clone the repo and create a virtual environment
2. `pip install -r requirements.txt`
3. Install Tesseract OCR
4. Create a `.env` file with `GROQ_API_KEY=your_key_here`
5. Run `python task_pdf.py`

## Live Demo
Link will be added after deployment.
