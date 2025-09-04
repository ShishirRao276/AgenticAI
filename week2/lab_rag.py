import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

DOC_PATH = "docs/simple_rag.txt"  
PERSIST_DIR = "week2/chroma_store_simple"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

if not os.path.exists(DOC_PATH):
    raise SystemExit(f"File not found: {DOC_PATH}")
with open(DOC_PATH, "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"Loaded {len(chunks)} chunks from {DOC_PATH}")

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="lab_simple",
    persist_directory=PERSIST_DIR
)

retriever = vectorstore.as_retriever(
    search_type="mmr",           
    search_kwargs={"k": 2, "fetch_k": 5}
)

llm = OllamaLLM(model=LLM_MODEL, base_url="http://localhost:11434", temperature=0)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

while True:
    q = input("\nAsk a question (or 'quit'): ").strip()
    if q.lower() in ("quit", "exit"):
        print("bye")
        break

    docs = retriever.invoke(q)
    print(f"\n[Retriever pulled {len(docs)} chunks]")
    for i, d in enumerate(docs, 1):
        print(f"Chunk {i}: {d.page_content[:150]}...")

    ans = qa.invoke({"query": q})
    print("\nAnswer:", ans)
