import os
import json
import requests
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

DB_DIR = "./vector_db"
ANSWERS_DIR = "answers"
os.makedirs(ANSWERS_DIR, exist_ok=True)

embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings_model)

queries = [
    "Як штучний інтелект допомагає оптимізувати рутинні завдання викладача ЗВО?",
    "Які існують ґрунтовні відмінності між штучним інтелектом та машинним навчання?",
    "Які особливості використання штучного інтелекту в діяльності підприємств?",
    "Які правові та етичні аспекти використання ШІ в контексті міжнародного врегулювання?",
    "Які виклики чи бар'єри виникають при інтеграції ШІ у навчальний процес та бізнес?"
]

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={API_KEY}"

for i, query in enumerate(queries, 1):
    query_vector = embeddings_model.embed_query(query)
    relevant_docs = vectorstore.similarity_search_by_vector(query_vector, k=2)
    retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt_text = f"""Використовуючи наведений контекст, дай відповідь на запитання користувача.

Контекст:
{retrieved_context}

Запитання:
{query}

Якщо відповіді на запитання немає в контексті, повідом про це."""

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }]
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    result_data = response.json()
    
    try:
        answer = result_data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        answer = f"Помилка отримання відповіді: {result_data}"

    file_content = f"Запит: {query}\n\nКонтекст:\n{retrieved_context}\n\n" + "="*50 + f"\n\nВідповідь LLM:\n{answer}"
    output_path = os.path.join(ANSWERS_DIR, f"answer_{i}.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(file_content)