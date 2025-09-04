import os, logging
from typing import List
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

DOCS_PATH = "docs"   
PERSIST_DIR = "chroma_store"  
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  
LLM_MODEL = "llama3.2:3b"  
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 120
TOP_K = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag-hf")

def load_folder(folder: str) -> List[str]:
    texts = []
    log.info(f"Scanning folder: {folder}")
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if name.lower().endswith(".txt"):
            log.info(f"Found TXT file: {name}")
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
                log.info(f"Loaded {len(txt)} characters from {name}")
                if txt.strip():
                    texts.append(txt)
        elif name.lower().endswith(".pdf"):
            log.info(f"Found PDF file: {name}")
            reader = PdfReader(path)
            pages = []
            for i, p in enumerate(reader.pages):
                page_txt = p.extract_text() or ""
                log.info(f"Page {i+1}/{len(reader.pages)} of {name}: {len(page_txt)} characters")
                pages.append(page_txt)
            txt = "\n".join(pages)
            if txt.strip():
                texts.append(txt)
        else:
            log.warning(f"Skipping unsupported file type: {name}")
    log.info(f"Loaded {len(texts)} documents from folder {folder}")
    return texts

def build_or_load_vectorstore(texts: List[str]) -> Chroma:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vs = Chroma(
        collection_name="kb_docs",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    count = vs._collection.count()
    if count > 0:
        log.info(f"Loaded existing Chroma collection with {count} embeddings.")
        return vs
    else:
        log.info("No embeddings found, rebuilding collection...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for i, t in enumerate(texts):
        chunks.extend(splitter.split_text(t))
    log.info(f"Total chunks: {len(chunks)}")

    vs = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="kb_docs",
        persist_directory=PERSIST_DIR,
    )
    log.info(f"Built and persisted new Chroma collection with {vs._collection.count()} embeddings.")
    return vs

if __name__ == "__main__":
    if not os.path.isdir(DOCS_PATH):
        raise SystemExit(f"Folder not found: {DOCS_PATH}. Create it and add .txt or .pdf files.")
    docs = load_folder(DOCS_PATH)
    if not docs:
        raise SystemExit(f"No docs found in {DOCS_PATH} (need .txt or .pdf).")

    vectorstore = build_or_load_vectorstore(docs)

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = OllamaLLM(model=LLM_MODEL, base_url="http://localhost:11434", temperature=0)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    while True:
        q = input("\nAsk about your docs (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit"):
            print("bye")
            break

        log.info(f"User question: {q}")
        docs = retriever.get_relevant_documents(q)
        log.info(f"Retriever returned {len(docs)} chunks")
        for i, d in enumerate(docs):
            log.debug(f"Chunk {i+1} (len={len(d.page_content)}): {d.page_content[:120]}...")

        ans = qa.invoke({"query": q})
        print("\nAnswer:", ans)