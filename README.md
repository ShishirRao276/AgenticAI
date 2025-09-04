---

# Ollama + LangChain Labs

This repo contains hands-on labs and **Mini-Projects** using [LangChain](https://www.langchain.com/) with [Ollama](https://ollama.com/) and [Chroma](https://www.trychroma.com/) to run **local LLM-powered Q\&A** over files.

---

## Project Structure

```
ollama-langchain-labs/
│
├── week1/
│   ├── lab_prompt.py          # Simple LangChain prompt → JSON output
│   ├── Mini_ProjectWeek1.py   # Local Q&A over a PDF with FAISS + Ollama embeddings
│   └── Report1.pdf            # Example file
│
├── week2/
│   ├── lab_rag.py             # RAG pipeline: Chroma + HuggingFace embeddings + Ollama LLM
│   ├── inspect_chroma.py      # Inspect Chroma store (docs, embeddings, similarity search)
│   └── docs/                  # Folder for your .txt/.pdf documents
│
└── README.md
```

---

## Features

* **LangChain + Ollama integration**
* **Prompt Library** (`qa`, `summarize`, `json_qa`)
* **Local PDF/Text Q\&A** using FAISS (Week 1) or Chroma (Week 2)
* **Embeddings**:

  * Week 1: via Ollama (slower)
  * Week 2: via HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (fast, CPU)
* **Knowledge Base Assistant** with persisted **ChromaDB** collections
* **Inspect tool** to check how many embeddings are stored and preview chunks
* **Interactive Q\&A** loop for your own questions

---

## Setup

### 1. Clone repo

```bash
git clone https://github.com/ShishirRao276/AgenticAI.git
cd ollama-langchain-labs
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

### 4. Install Ollama & pull models

```bash
# Install Ollama from https://ollama.com/download
ollama pull llama3.2:3b          # LLM for answering
```

---

## Usage

### Week 1 – Simple Prompt & FAISS Q\&A

```bash
cd week1
python lab_prompt.py
python Mini_ProjectWeek1.py
```

Example:

```text
Enter your question: Summarize the file
```

Output:

```json
{
  "question": "Summarize the file",
  "answer": "The file discusses challenges in detecting hate speech in memes, highlighting the need for more sophisticated models..."
}
```

---

### Week 2 – RAG with Chroma + HuggingFace

#### 1. Put your docs

Drop `.pdf` or `.txt` files into `week2/docs/`.

#### 2. Run the RAG pipeline

```bash
cd week2
python lab_rag.py
```

Example:

```text
Ask about your docs (or 'quit'): Who are the authors?
```

Logs will show:

```
[INFO] Found PDF file: Report1.pdf
[INFO] Page 1/7 of Report1.pdf: 5195 characters
...
[INFO] No embeddings found, rebuilding collection...
[INFO] Total chunks: 23
[INFO] Built new Chroma collection with 23 embeddings.
```

And you’ll get an answer grounded in your document.

#### 3. Inspect Chroma

Use the inspection tool to see stored embeddings and test similarity search directly:

```bash
python inspect_chroma.py
```

Example output:

```
Total embeddings in collection 'kb_docs': 23

Sample docs:
Doc 1: DETECTION OF HATE SPEECH IN MEMES ...
Doc 2: I. INTRODUCTION ...
Doc 3: text and image that often characterizes memes...

Query: What is this document about?
Result 1: The dataset comprises several thousand memes...
Result 2: pre-trained models with advanced text processing...
Result 3: deep learning transformers capturing contextual...
```

---