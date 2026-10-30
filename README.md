
# 🧠 Guardrailed Retrieval-Augmented Generation (RAG) Pipeline  
A reliable, production‑ready RAG system built with **ChromaDB**, **Ollama**, and Python.  
This version includes **all four required guardrails**: distance filtering, confidence scoring, strengthened system prompt, and structured responses.

---

## 📌 Overview

This project extends the basic RAG pipeline from Lesson 3 by adding **robust guardrails** that prevent hallucinations, enforce grounding, and provide confidence scoring. These guardrails are required for the module’s final project (Containerized RAG Assistant).

The pipeline performs:

1. **Document ingestion**  
2. **Paragraph‑level chunking**  
3. **Embedding storage in ChromaDB**  
4. **Semantic retrieval with distance filtering**  
5. **Guardrailed generation using Ollama**  
6. **Structured response output**  
7. **Confidence scoring**  
8. **Testing across four query types**

---

## 🛡️ Guardrails Implemented

### **Guardrail 1 — Distance Threshold Filtering**
- Only chunks with distance ≤ `dt_threshold` (default: **1.0**) are used.  
- If no chunks pass the filter, the pipeline returns:

```json
{
  "answer": "I don’t know.",
  "sources": [],
  "confidence": "low",
  "chunks_retrieved": []
}
```

This prevents irrelevant context from influencing the model.

---

### **Guardrail 2 — Confidence Levels**
Confidence is computed from the **best (lowest) distance**:

| Best Distance | Confidence |
|--------------|------------|
| `< 0.5`      | **high**   |
| `< 1.0`      | **medium** |
| `>= 1.0`     | **low**    |

This gives downstream systems (FastAPI, Streamlit) a way to judge answer reliability.

---

### **Guardrail 3 — Strengthened System Prompt**
The system prompt explicitly instructs the model to:

- Use **only** retrieved context  
- **Never** make up information  
- Say **“I don’t know”** when unsure  
- Always cite filename + chunk index  
- Never invent citations  
- Keep answers concise  
- Produce output compatible with structured responses  

This prevents hallucinations and enforces strict grounding.

---

### **Guardrail 4 — Structured Response Dictionary**
Every call to `rag_query()` returns:

```json
{
  "answer": "...",
  "sources": [...],
  "confidence": "...",
  "chunks_retrieved": [...]
}
```

This makes the pipeline easy to integrate into:

- FastAPI endpoints  
- Streamlit UIs  
- Dockerized microservices  

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
rag_db/   (persistent ChromaDB storage)
```

---

## ⚙️ How the Pipeline Works

### **1. Ingestion**
- Loads `.txt` and `.md` files  
- Splits into paragraphs  
- Stores chunks with metadata  
- Embeds and persists them in ChromaDB  

### **2. Retrieval**
- Queries ChromaDB using cosine similarity  
- Sorts by distance  
- Applies threshold filtering  

### **3. Prompt Construction**
- Injects retrieved chunks into the system prompt  
- Includes guardrail instructions  
- Adds source metadata  

### **4. Guardrailed Generation**
- Calls Ollama’s chat API  
- Produces grounded answers  
- Returns structured response  

---

## 🧪 Testing (All 4 Required Queries)

The pipeline is tested with:

### **1. In‑scope**
A question directly answered by the documents.

### **2. Partially in‑scope**
Related to the domain but not explicitly answered.

### **3. Out‑of‑scope**
Completely unrelated to the documents.

### **4. Ambiguous**
Lacks clear meaning or context.

Each test prints the **structured response dictionary**, including:

- Answer  
- Sources  
- Confidence  
- Retrieved chunks  

---

## 🚀 Running the Pipeline

Start Ollama:

```
ollama serve
```

Run the script:

```
python my_rag.py
```

View the structured responses printed for all four test queries.

---

## 🎯 Purpose

This guardrailed RAG pipeline is designed to be:

- **Reliable**  
- **Grounded**  
- **Non‑hallucinatory**  
- **Production‑ready**

It forms the foundation for your upcoming:

- FastAPI backend  
- Streamlit frontend  
- Docker deployment  

By implementing guardrails now, your final project becomes significantly easier.

