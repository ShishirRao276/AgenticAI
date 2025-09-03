---

# Ollama + LangChain Labs

This repo contains hands-on labs and a **Mini-Project** using [LangChain](https://www.langchain.com/) with [Ollama](https://ollama.com/) to run **local LLMs** for Q\&A over files.

---

## Project Structure

```
ollama-langchain-labs/
│
├── week1/
│   ├── lab_prompt.py          # Simple LangChain prompt → JSON output
│   ├── Mini_ProjectWeek1.py   # Local Q&A over a PDF with configurable prompts
│   └── Report1.pdf            # Example file
└── README.md
```

---

## Features

* **LangChain + Ollama integration**
* **Configurable Prompt Library** (`qa`, `summarize`, `json_qa`)
* **Local PDF Q\&A** using embeddings + FAISS
* **Structured JSON answers**

---

## Setup

### 1. Clone repo

```bash
git clone https://github.com/ShishirRao276/AgenticAI.git
cd ollama-langchain-labs/week1
```

### 2. Create & activate venv

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install langchain langchain-ollama faiss-cpu pypdf
```

### 4. Install Ollama & pull a model

```bash
# Install Ollama from https://ollama.com/download
ollama pull llama3.2:3b   # small, faster
# or
ollama pull llama3.1:8b   # bigger, more accurate
```

---

## Usage

### Lab Prompt Example

```bash
python lab_prompt.py
```

**Input:**

```
Enter your question: What is the capital of France?
```

**Output:**

```json
{
  "question": "What is the capital of France?",
  "answer": "Paris"
}
```

---

### Mini Project – PDF Q\&A

```bash
python Mini_ProjectWeek1.py
```

**Input:**

```
Enter your question: Summarize the file
```

**Logs (truncated):**

```
[INFO] Starting Q&A pipeline...
[INFO] Running QA with prompt type: json_qa
[INFO] Loading PDF: Report1.pdf
[INFO] Extracted page 1 with 5195 chars
[INFO] Extracted page 2 with 5126 chars
[INFO] Extracted page 3 with 3590 chars
[INFO] Extracted page 4 with 4304 chars
[INFO] Extracted page 5 with 1388 chars
[INFO] Extracted page 6 with 3028 chars
[INFO] Extracted page 7 with 2843 chars
[INFO] Finished PDF extraction in 0.26 sec
[INFO] Splitting text into chunks...
[INFO] Created 49 chunks
[INFO] Creating embeddings with Ollama...
[INFO] HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
[INFO] Loading faiss with AVX2 support.
[INFO] Successfully loaded faiss with AVX2 support.
[INFO] Embeddings + FAISS index built in 497.11 sec
[INFO] Building RetrievalQA chain...
[INFO] Querying LLM with: Who are the authors of the file?
[INFO] HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
[INFO] HTTP Request: POST http://localhost:11434/api/generate "HTTP/1.1 200 OK"
[INFO] LLM answered in 44.99 sec
```

**Final Answer:**

```json
{
  "question": "Summarize the file",
  "answer": "The file discusses challenges in detecting hate speech in memes, highlighting the need for more sophisticated models that can interpret nuanced layers of communication. It presents a comparison between different models, including image-based CNNs and autoencoders, and their utility in evaluating algorithm effectiveness."
}
```

---
