from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "chroma_store"
COLLECTION_NAME = "kb_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load embeddings + Chroma collection
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vs = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

# 1. Count total embeddings
print(f"Total embeddings in collection '{COLLECTION_NAME}': {vs._collection.count()}")

# 2. Show first 3 docs
res = vs._collection.get(limit=3)
print("\nSample docs:")
for i, doc in enumerate(res["documents"], 1):
    print(f"Doc {i}: {doc[:200]}...")

# 3. Try a similarity search
query = "What is this document about?"
results = vs.similarity_search(query, k=3)
print(f"\nQuery: {query}")
for i, r in enumerate(results, 1):
    print(f"Result {i}: {r.page_content[:200]}...")
