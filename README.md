# RAG API with Chat History & Streaming

This project implements a **Retrieval-Augmented Generation (RAG)** API using **FastAPI**, **LangChain**, and **Chroma** vector store. It allows you to load or fetch documents, split them into chunks, store embeddings, and query them using a language model (LLM) with session-based chat history. The API also supports **streaming responses** for interactive chat applications.

---

## Features

- Load documents from:
  - Local files (`.pdf` or `.docx`)  
  - URLs
- Split documents into manageable chunks using `RecursiveCharacterTextSplitter`.
- Store vector embeddings in a local **Chroma** vector store.
- Query documents using **LLM** (`Gemini-2.5-Flash`) with context retrieval.
- Session-based chat history for conversational queries.
- Non-streaming and streaming endpoints via **FastAPI**.
- Ngrok integration for easy public access (ideal for Colab or local testing).

---

## Requirements

- Python 3.10+

## Install dependencies:
```bash
pip install -r requirements.txt
```

## Required environment variables:

- GOOGLE_API_KEY — Your Google API Key (needed to use the LLM)
- NGROK_AUTHTOKEN — Ngrok authentication token (needed to expose the API publicly)

## How It Works

### 1. Document Loading
- Load files from a local folder or URL.
- Supports `.pdf` and `.docx` files.

### 2. Document Splitting
- Splits documents into chunks of `1000` characters with `200` character overlap.
- Improves semantic search and context handling.

### 3. Vector Store
- Chunks are embedded (via `sentence_transformer_embeddings`) and stored in **Chroma**.
- Local persistence in `./chroma_langchain_db`.

### 4. RAG Chain Setup
- Retrieves top-k relevant chunks from the vector store.
- Formats context, chat history, and question with a **prompt template**.
- Generates answer with **LLM**.
- Session-based chat history using `InMemoryChatMessageHistory`.

### 5. FastAPI Endpoints
- `/` - Health check.
- `/query` - Non-streaming query endpoint.
- `/query_stream` - Streaming endpoint for interactive chat.

### 6. Streaming with Async
- Uses `asyncio` and `StreamingResponse` to stream answers in real-time.

### 7. Ngrok Integration
- Exposes local FastAPI server publicly for Colab or local development.
