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
    "Які основні напрями та переваги застосування систем штучного інтелекту в освітньому процесі вищих навчальних закладів?",
    "У чому полягають ключові відмінності між класичними алгоритмами машинного навчання та сучасними рішеннями на базі штучного інтелекту?",
    "Яким чином інтеграція штучного інтелекту впливає на ефективність бізнес-процесів та прийняття управлінських рішень на підприємствах?",
    "Які етичні норми, ризики та правові обмеження регулюють використання технологій ШІ на міжнародному рівні?",
    "З якими основними технічними та організаційними бар'єрами стикаються організації під час впровадження рішень штучного інтелекту?"
]

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={API_KEY}"

for i, query in enumerate(queries, 1):
    query_vector = embeddings_model.embed_query(query)
    relevant_docs = vectorstore.similarity_search_by_vector(query_vector, k=2)
    
    fragments = []
    sources = set()
    for doc in relevant_docs:
        fragments.append(doc.page_content)
        source_name = doc.metadata.get("source", "Невідомо")
        sources.add(os.path.basename(source_name))
        
    retrieved_context = "\n\n".join(fragments)
    source_str = ", ".join(sources)
    
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
        answer = result_data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        answer = f"Помилка отримання відповіді: {result_data}"


    print(f"ЗАПИТ №{i}")
    print(f"Запит користувача: {query}")
    print(f"Джерело інформації: {source_str}")
    print(f"Знайдений фрагмент:\n{retrieved_context}")
    print(f"Відповідь LLM:\n{answer}")


    file_content = f"Запит: {query}\n\nДжерело: {source_str}\n\nКонтекст:\n{retrieved_context}\n\n" + "="*50 + f"\n\nВідповідь LLM:\n{answer}"
    output_path = os.path.join(ANSWERS_DIR, f"task7_answer_{i}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(file_content)