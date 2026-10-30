import chromadb
import requests
import json
import os
import time

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"  # Adjust based on your downloaded model
client = chromadb.PersistentClient(path="rag_db")
collection = client.get_or_create_collection(
    "my_rag_collection",
    configuration={"hnsw": {"space": "cosine"}}
)


# --- Step 1: Ingestion ---

def load_documents(directory):
    """Load all .txt and .md files from a directory."""
    docs = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(('.txt', '.md')):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                content = f.read()
            # Split by paragraphs for chunking
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            for i, para in enumerate(paragraphs):
                docs.append({
                    "text": para,
                    "id": f"{filename}_{i}",
                    "metadata": {"source": filename, "chunk_index": str(i)}
                })
    return docs

def ingest(collection, docs_directory):
    """Load, chunk, and store documents in ChromaDB."""
    chunks = load_documents(docs_directory)
    if not chunks:
        print("No documents found!")
        return 0

    collection.upsert(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[c["id"] for c in chunks]
    )
    return len(chunks)

# --- Step 2: Retrieval ---

def retrieve(collection, query, n_results=3):
    """Search ChromaDB for relevant document chunks."""
    # Guard: no documents ingested
    if collection.count() == 0:
        print("No documents in the collection. Did you run ingestion?")
        return []
    
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count())
    )

    chunks = []
    for i in range(len(results['documents'][0])):
        chunks.append({
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i]
        })
    # Sort once, after the list is complete
    chunks.sort(key=lambda c: c["distance"])

    return chunks


# --- Step 3: Prompt Building ---

SYSTEM_PROMPT = """You are a guardrailed retrieval‑augmented assistant. Follow these rules exactly:

1. Use ONLY the retrieved context provided to you. Never rely on outside knowledge.
2. If the answer is not fully supported by the context, say “I don’t know.”
3. Never make up information, details, explanations, or assumptions.
4. Cite your sources by listing the filename and chunk index for every fact you use.
5. Do NOT invent citations. Only cite chunks that were actually retrieved.
6. Keep your answer concise, factual, and grounded.
7. Your answer will be placed into a structured response dictionary. Do NOT include extra commentary.
8. After your answer, include a “Sources:” section listing the chunks you used.
9. If no chunks are relevant, respond only with: “I don’t know.” and an empty Sources list.
"""


def build_messages(question, retrieved_chunks):
    """Build the RAG prompt with retrieved context."""
    context_parts = []
    for chunk in retrieved_chunks:
        source = chunk['metadata']['source']
        idx = chunk['metadata']['chunk_index']
        context_parts.append(f"[Source: {source}, Chunk {idx}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)
    system_with_context = f"{SYSTEM_PROMPT}\nCONTEXT:\n{context}"

    return [
        {"role": "system", "content": system_with_context},
        {"role": "user", "content": question}
    ]

# --- Step 4: Generation ---

def generate(messages, stream=False):
    """Call Ollama to generate a response."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": MODEL, "messages": messages, "stream": stream},
            stream=stream,
            timeout=30
        )

        if response.status_code != 200:
            return f"ERROR: Ollama returned status {response.status_code}: {response.text}"

        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        full_response += token
            print()
            return full_response

        # Non-streaming mode
        try:
            return response.json()["message"]["content"]
        except (KeyError, TypeError):
            return "ERROR: Unexpected response format from Ollama."

    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Is it running? (ollama serve)"
# --- Step 5: The RAG Function ---

def rag_query(collection, question, n_results=3, stream=True, dt_threshold=1.0):
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # Retrieve
    retrieved = retrieve(collection, question, n_results)
    chunks = [c for c in retrieved if c["distance"] <= dt_threshold]

    print(f"\nRetrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] (dist: {chunk['distance']:.4f}) {chunk['metadata']['source']}")

    # Guardrail: no relevant chunks
    if len(chunks) == 0:
        return {
            "answer": "I dont know.",
            "sources": [],
            "confidence": "low",
            "chunks_retrieved": []
        }

    # Build prompt
    messages = build_messages(question, chunks)

    # Generate
    print(f"\nAnswer (streaming):\n")
    start = time.time()
    answer = generate(messages, stream=stream)
    elapsed = time.time() - start
    print(f"\n[Generated in {elapsed:.1f}s]")

    # Confidence scoring
    best_dist = chunks[0]["distance"]
    if best_dist < 0.5:
        confidence = "high"
    elif best_dist < 1.0:
        confidence = "medium"
    else:
        confidence = "low"

    # Structured response
    return {
        "answer": answer,
        "sources": [c["metadata"] for c in chunks],
        "confidence": confidence,
        "chunks_retrieved": chunks
    }

#---Testing Questions--- 



# --- Ingest documents ---
num_chunks = ingest(collection, "docs")
print(f"Ingested {num_chunks} chunks.")

# --- Testing Guardrailed RAG Pipeline ---

test_queries = [
    ("In-scope", "What does it mean for an API to be stateless?"),
    ("Partially in-scope", "Does FastAPI automatically connect to a SQL database?"),
    ("Out-of-scope", "How do airplanes generate lift?"),
    ("Ambiguous", "Can you tell me more about that thing we talked about earlier?")
]

print("\n" + "="*60)
print("RUNNING 4 GUARDRAILED RAG TESTS")
print("="*60)

for label, query in test_queries:
    print("\n" + "-"*60)
    print(f"TEST: {label}")
    print("-"*60)

    result = rag_query(collection, query, n_results=3, stream=False, dt_threshold=1.0)

    # Print structured response dictionary
    print("\nStructured Response:")
    print(json.dumps(result, indent=4))
