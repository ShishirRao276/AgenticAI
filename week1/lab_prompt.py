from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama3.1",
    base_url="http://localhost:11434",
    temperature=0
)

template = """
You are a JSON-only bot.

Question: {question}

Respond ONLY in valid JSON with this structure:
{{
  "question": "{question}",
  "answer": "<your answer here>"
}}
"""

prompt = PromptTemplate.from_template(template)

if __name__ == "__main__":
    user_question = input("Enter your question: ")
    final = prompt.format(question=user_question)
    out = llm.invoke(final)
    print(out)
