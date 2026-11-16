import os
from dotenv import load_dotenv
import requests

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={API_KEY}"

few_shot_examples = [
    {
        "text": "Python є мовою програмування високого рівня.",
        "category": "програмування"
    },
    {
        "text": "Нейронна мережа складається з набору взаємопов'язаних нейронів.",
        "category": "штучний інтелект"
    },
    {
        "text": "Реляційна база даних зберігає інформацію у вигляді таблиць.",
        "category": "бази даних"
    },
    {
        "text": "Маршрутизація пакетів у мережі Інтернет здійснюється за протоколом IP.",
        "category": "комп'ютерні мережі"
    }
]

examples_text = ""
for ex in few_shot_examples:
  examples_text += f"Текст:\n{ex['text']}\n\nКатегорія:\n{ex['category']}\n\n"

test_queries = [
    "Алгоритм зворотного поширення помилки використовується для оптимізації ваг у нейронних мережах.",
    "Мова програмування C++ підтримує об'єктно-орієнтоване програмування через класи та віртуальні функції.",
    "Реляційні СУБД використовують мову SQL для виконання запитів та управління транзакціями ACID.",
    "Маршрутизація пакетів у мережі Інтернет здійснюється за протоколом IP на основі таблиць маршрутизації.",
    "Згорткові нейронні мережі (CNN) є стандартним архітектурним рішенням для обробки та класифікації зображень.",
    "Індексація полів у таблицях PostgreSQL дозволяє значно прискорити виконання пошукових запитів.",
    "Функції вищого порядку в JavaScript приймають інші функції як аргументи або повертають їх."
]

print("Виконання Завдання 4: Імітація Fine-tuning через набір прикладів")

for i, query in enumerate(test_queries, 1):
  prompt = f"""Виконуй класифікацію текстів на основі наведених нижче прикладів.

{examples_text}Текст:
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
  print(f"Запит: {query}")

  if response.status_code == 200:
    res_data = response.json()
    try:
      model_answer = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
      print(f"Результат від LLM (Категорія): {model_answer}")
    except (KeyError, IndexError):
      print(f"Помилка парсингу: {res_data}")
  else:
    print(f"Помилка запиту ({response.status_code}): {response.text}")