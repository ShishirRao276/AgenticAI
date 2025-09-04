
---

# Ollama + LangChain Labs

This repo contains **hands-on labs** and **Mini-Projects** using [LangChain](https://www.langchain.com/), [Ollama](https://ollama.com/), and [Chroma](https://www.trychroma.com/) to run **local LLM-powered Q\&A** over your own documents.

Each week introduces new concepts and builds towards creating a **Knowledge Base (KB) Assistant**.

---

## Project Structure

```
ollama-langchain-labs/
│
├── week1/
│   ├── lab_prompt.py          # Simple LangChain prompt → JSON output
│   ├── Mini_ProjectWeek1.py   # Mini Project #1: Q&A over PDF with FAISS + Ollama embeddings
│   └── Report1.pdf            # Example file
│
├── week2/
│   ├── lab_rag_simple.py      # Lab: Simple RAG over a text file
│   ├── Mini_ProjectWeek2.py   # Mini Project #2: KB Assistant with Chroma + HuggingFace + Ollama
│   ├── inspect_chroma.py      # Inspect Chroma store (embeddings, docs, similarity search)
│   └── docs/                  # Folder for your .txt/.pdf documents
│
└── README.md
```

---

## Features

* **LangChain + Ollama integration**
* **Prompt Library** for structured outputs (`qa`, `summarize`, `json_qa`)
* **Local Q\&A over documents**

  * Week 1 → FAISS + Ollama embeddings (slower, simple intro)
  * Week 2 → Chroma + HuggingFace embeddings (faster, CPU-friendly, persistent)
* **Knowledge Base Assistant** with persistent **ChromaDB collections**
* **Inspection tool** to check stored embeddings and preview chunks
* **Interactive Q\&A loop** for natural language queries

---

## Setup

### 1. Clone repo

```bash
git clone https://github.com/ShishirRao276/AgenticAI.git
cd AgenticAI
```

### 2. Create & activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama & pull models

```bash
# Download Ollama from https://ollama.com/download
ollama pull llama3.2:3b   # Small, faster
# or
ollama pull llama3.1:8b   # Larger, more accurate
```

---

## Usage

### Week 1 – Prompting & FAISS Q\&A

#### 1. Lab – Prompt Template → JSON output

```bash
cd week1
python lab_prompt.py
```

Example:

```
Enter your question: What is the capital of France?
```

Output:

```json
{
  "question": "What is the capital of France?",
  "answer": "Paris"
}
```

---

#### 2. Mini Project #1 – Local PDF Q\&A with FAISS

```bash
cd week1
python Mini_ProjectWeek1.py
```

Example:

```
Enter your question: Summarize the file
```

Logs (truncated):

```
[INFO] Loading PDF: Report1.pdf
[INFO] Extracted page 1 with 5195 chars
[INFO] Created 49 chunks
[INFO] Creating embeddings with Ollama...
[INFO] Embeddings + FAISS index built in 497.11 sec
[INFO] Querying LLM with: Summarize the file
[INFO] LLM answered in 44.99 sec
```

Final Answer:

```json
{
  "question": "Summarize the file",
  "answer": "The file discusses challenges in detecting hate speech in memes, highlighting the need for more sophisticated models..."
}
```

---

### Week 2 – RAG with Chroma + HuggingFace

In Week 1, we used **Ollama embeddings**. That worked, but it was **extremely slow**, since Ollama had to embed every chunk through its local LLM endpoint.

In Week 2, we switched to **HuggingFace Sentence Transformers** (`all-MiniLM-L6-v2`) for embeddings because:

* **Much faster** on CPU (no GPU required)
* Models are optimized for **semantic similarity tasks**
* **Embeddings are cached** locally and reused
* **Privacy-friendly** → no cloud calls

Now, HuggingFace handles embeddings, Chroma stores them, and Ollama is used purely for **LLM inference (answers)**.

---

#### 1. Lab – Simple RAG (`lab_rag.py`)

Reads a single `.txt` file, splits into chunks, embeds with HuggingFace, stores in Chroma, retrieves via MMR reranking, and answers with Ollama.

Run:

```bash
cd week2
python lab_rag.py
```

Example:

```
Ask a question (or 'quit'): What is Chroma?
[Retriever pulled 2 chunks]
Chunk 1: Chroma is a vector database used for storing embeddings...
Chunk 2: LangChain makes it easy to build applications...
Answer: Chroma is a vector database for storing embeddings and enabling semantic search.
```

---

#### 2. Mini Project #2 – KB Assistant (`Mini_ProjectWeek2.py`)

Handles multiple `.pdf` and `.txt` files in `docs/`.
Builds a **persistent Chroma store** that survives across runs.

Run:

```bash
cd week2
python Mini_ProjectWeek2.py
```

Logs (example):

```
[INFO] Found PDF file: Report1.pdf
[INFO] Page 1/7 of Report1.pdf: 5195 characters
...
[INFO] No embeddings found, rebuilding collection...
[INFO] Total chunks: 23
[INFO] Built new Chroma collection with 23 embeddings.
[INFO] User question: Who are the authors?
[INFO] Retriever returned 3 chunks
```

Answer:

```
The authors are Sai Shishir Ailneni, Priyaanka Reddy Boothkuri, and Manogna Tummanepally.
```

---

#### 3. Inspect Stored Embeddings (`inspect_chroma.py`)

Check how many embeddings are stored, preview documents, and run direct similarity search.

Run:

```bash
python inspect_chroma.py
```

Example:

```
Total embeddings in collection 'kb_docs': 23

Sample docs:
Doc 1: DETECTION OF HATE SPEECH IN MEMES ...
Doc 2: I. INTRODUCTION ...
Doc 3: ...

Query: What is this document about?
Result 1: The dataset comprises several thousand memes...
Result 2: pre-trained models with advanced text processing...
Result 3: deep learning transformers capturing contextual...
```

---