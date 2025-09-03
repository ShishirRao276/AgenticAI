import logging
import time
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM,OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pypdf import PdfReader

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- Prompt Library ----------------
PROMPTS = {
    "qa": PromptTemplate.from_template(
        "Use the context to answer the question.\n\nContext:\n{context}\n\nQ: {question}\nA:"
    ),
    "summarize": PromptTemplate.from_template(
        "Summarize the following text:\n\n{context}\n\nSummary:"
    ),
    "json_qa": PromptTemplate.from_template(
        """
        Context:
        {context}

        Question: {question}

        Respond ONLY in valid JSON:
        {{
          "question": "{question}",
          "answer": "<your answer here>"
        }}
        """
    ),
}


# ---------------- Helper Functions ----------------
def load_pdf(path="Report1.pdf"):
    logger.info(f"Loading PDF: {path}")
    start = time.time()
    reader = PdfReader(path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        logger.info(f"Extracted page {i+1} with {len(page_text)} chars")
        text += page_text + "\n"
    logger.info(f"Finished PDF extraction in {time.time()-start:.2f} sec")
    return text


def build_retriever(text: str):
    logger.info("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_text(text)
    logger.info(f"Created {len(chunks)} chunks")

    logger.info("Creating embeddings with Ollama...")
    start = time.time()
    embeddings = OllamaEmbeddings(model="llama3.2:3b", base_url="http://localhost:11434")
    vs = FAISS.from_texts(chunks, embeddings)
    logger.info(f"Embeddings + FAISS index built in {time.time()-start:.2f} sec")

    return vs.as_retriever(search_kwargs={"k": 3})


def ask_file(question: str, prompt_type="qa", file_path="Report1.pdf"):
    logger.info(f"Running QA with prompt type: {prompt_type}")
    text = load_pdf(file_path)
    retriever = build_retriever(text)
    llm = OllamaLLM(model="llama3.2:3b", base_url="http://localhost:11434", temperature=0)

    prompt = PROMPTS[prompt_type]

    logger.info("Building RetrievalQA chain...")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

    logger.info(f"Querying LLM with: {question}")
    start = time.time()
    answer = chain.invoke(question)
    logger.info(f"LLM answered in {time.time()-start:.2f} sec")

    return answer


# ---------------- Main ----------------
if __name__ == "__main__":
    q = input("Enter your question: ")
    logger.info("Starting Q&A pipeline...")
    answer = ask_file(q, prompt_type="json_qa", file_path="Report1.pdf")
    print("\nFinal Answer:\n", answer)
