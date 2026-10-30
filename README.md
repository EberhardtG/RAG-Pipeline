
# 🧠 Retrieval-Augmented Generation (RAG) Pipeline  
A lightweight RAG system built with **ChromaDB**, **Ollama**, and a custom Python pipeline. This project implements ingestion, retrieval, prompt construction, grounded generation, and testing over a local document collection.

---

## 📌 Overview

This project demonstrates a complete RAG workflow:

1. **Ingest** a folder of text documents  
2. **Chunk** them by paragraph  
3. **Store** them in a persistent ChromaDB collection  
4. **Retrieve** the most relevant chunks for a user query  
5. **Generate** an answer using Ollama, grounded strictly in retrieved context  
6. **Cite sources** for every fact used  
7. **Gracefully refuse** when the answer is not in the documents  
8. **Run three required test queries** to validate behavior  

This pipeline forms the foundation for the module’s larger project (FastAPI + Streamlit + Docker).

---

## 📁 Project Structure

```
my_rag.py
docs/
    api_basics.txt
    python_functions.txt
    fastapi_intro.txt
    sql_databases.txt
    embeddings_overview.txt
    llm_behavior.txt
rag_db/   (auto-created by ChromaDB)
```

---

## ⚙️ Features

### **Document Ingestion**
- Loads `.txt` and `.md` files  
- Splits content by paragraphs  
- Stores chunks with metadata (filename + chunk index)  
- Persists embeddings in ChromaDB  

### **Semantic Retrieval**
- Queries ChromaDB for top‑N relevant chunks  
- Sorts results by distance  
- Displays retrieved chunks before answering  

### **RAG Prompt Construction**
- Injects retrieved context into the system prompt  
- Includes strict grounding rules:
  - Use ONLY retrieved context  
  - Say *“I don’t know”* if answer is not present  
  - Cite filename + chunk index  
  - No outside knowledge  
  - No invented citations  

### **Grounded Generation**
- Calls Ollama’s `/api/chat` endpoint  
- Supports streaming output  
- Handles:
  - Connection errors  
  - Timeouts  
  - Malformed responses  
  - Non‑200 status codes  

### **Interactive Query Function**
- Runs retrieval → prompt building → generation  
- Prints timing  
- Returns grounded answer  

---

## 🧪 Testing (Required by Assignment)

The project includes three test queries demonstrating correct RAG behavior:

### **1. Answerable Question**
**“What does it mean for an API to be stateless?”**  
Expected:  
- Correct grounded answer  
- Citations from `api_basics.txt`  

### **2. Related but Not Directly Answered**
**“Does FastAPI automatically connect to a SQL database?”**  
Expected:  
- Retrieved chunks from FastAPI + SQL docs  
- Model responds: **“I don’t know.”**  
- Citations included  

### **3. Outside the Scope**
**“How do airplanes generate lift?”**  
Expected:  
- Irrelevant chunks retrieved  
- Model responds: **“I don’t know.”**  
- Citations included  

These tests confirm that the pipeline handles:

- In-domain answers  
- Partial-domain questions  
- Out-of-domain questions  

Exactly as a proper RAG system should.

---

## 🚀 Running the Pipeline

1. Start Ollama:

```
ollama serve
```

2. Run the script:

```
python my_rag.py
```

3. View the printed results for all three test queries.

---

## 🎯 Purpose

This project builds the core RAG logic you will later wrap with:

- **FastAPI** (backend API)  
- **Streamlit** (frontend UI)  
- **Docker** (deployment)  

