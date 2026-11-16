import os
from dotenv import load_dotenv
import requests
import chromadb
from pypdf import PdfReader

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={API_KEY}"

files_data = [
    {"filename": "documents/doc1.pdf", "category": "програмування"},
    {"filename": "documents/doc2.pdf", "category": "штучний інтелект"},
    {"filename": "documents/doc3.pdf", "category": "бази даних"},
    {"filename": "documents/doc4.pdf", "category": "комп'ютерні мережі"},
]

documents_text = []
metadatas = []
ids = []
chunk_counter = 1

for item in files_data:
  reader = PdfReader(item["filename"])
  full_text = ""
  for page in reader.pages:
    full_text += page.extract_text() + "\n"

  chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
  if not chunks:
    chunks = [full_text.strip()]

  for chunk in chunks:
    documents_text.append(chunk)
    metadatas.append({"category": item["category"], "source": item["filename"]})
    ids.append(f"chunk_{chunk_counter}")
    chunk_counter += 1

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="rag_lab_collection")

collection.add(
    documents=documents_text,
    metadatas=metadatas,
    ids=ids
)

test_queries = [
    "Алгоритм зворотного поширення помилки використовується для оптимізації ваг у нейронних мережах.",
    "Мова програмування C++ підтримує об'єктно-орієнтоване програмування через класи та віртуальні функції.",
    "Реляційні СУБД використовують мову SQL для виконання запитів та управління транзакціями ACID.",
    "Маршрутизація пакетів у мережі Інтернет здійснюється за протоколом IP на основі таблиць маршрутизації.",
    "Згорткові нейронні мережі (CNN) є стандартним архітектурним рішенням для обробки та класифікації зображень.",
    "Індексація полів у таблицях PostgreSQL дозволяє значно прискорити виконання пошукових запитів.",
    "Функції вищого порядку в JavaScript приймають інші функції як аргументи або повертають їх."
]

categories_list = "програмування, штучний інтелект, бази даних, комп'ютерні мережі"

print("Виконання Завдання 3: Повний RAG-пайплайн для класифікації текстів")

for i, query in enumerate(test_queries, 1):
  results = collection.query(query_texts=[query], n_results=1)
  
  retrieved_context = results["documents"][0][0]
  retrieved_source_file = results["metadatas"][0][0]["source"]
  retrieved_category = results["metadatas"][0][0]["category"]

  prompt = f"""Використовуючи наведений контекст, визнач категорію запиту.
Можливі категорії: {categories_list}.
Поверни лише назву категорії.

Контекст:
{retrieved_context}

Запит:
{query}
Категорія:"""

  payload = {
      "contents": [{
          "parts": [{"text": prompt}]
      }]
  }
  headers = {"Content-Type": "application/json"}

  response = requests.post(url, json=payload, headers=headers)

  print(f"\nПриклад #{i}")
  print(f"Запит користувача: {query}")
  print(f"Знайдений контекст [Файл: {retrieved_source_file}, Категорія джерела: {retrieved_category}]")

  if response.status_code == 200:
    res_data = response.json()
    try:
      model_answer = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
      print(f"Відповідь LLM (Результат): {model_answer}")
    except (KeyError, IndexError):
      print(f"Помилка парсингу відповіді: {res_data}")
  else:
    print(f"Помилка запиту ({response.status_code}): {response.text}")