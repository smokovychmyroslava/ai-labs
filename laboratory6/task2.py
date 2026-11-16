import os
from dotenv import load_dotenv
import requests

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={API_KEY}"

test_texts = [
    (
        "Алгоритм зворотного поширення помилки використовується для оптимізації"
        " ваг у нейронних мережах."
    ),
    (
        "Мова програмування C++ підтримує об'єктно-орієнтоване програмування через"
        " класи та віртуальні функції."
    ),
    (
        "Реляційні СУБД використовують мову SQL для виконання запитів та"
        " управління транзакціями ACID."
    ),
    (
        "Маршрутизація пакетів у мережі Інтернет здійснюється за протоколом IP"
        " на основі таблиць маршрутизації."
    ),
    (
        "Згорткові нейронні мережі (CNN) є стандартним архітектурним рішенням для"
        " обробки та класифікації зображень."
    ),
    (
        "Індексація полів у таблицях PostgreSQL дозволяє значно прискорити"
        " виконання пошукових запитів."
    ),
    (
        "Функції вищого порядку в JavaScript приймають інші функції як"
        " аргументи або повертають їх."
    ),
]

categories_list = (
    "програмування, штучний інтелект, бази даних, комп'ютерні мережі"
)

print("Виконання Завдання 2: Prompt Engineering (тестування на 7 прикладах)")

for i, text in enumerate(test_texts, 1):
  prompt = f"""Визнач категорію наведеного тексту.
Можливі категорії: {categories_list}.
Поверни лише назву категорії.

Текст: {text}
Категорія:"""

  payload = {"contents": [{"parts": [{"text": prompt}]}]}

  headers = {"Content-Type": "application/json"}

  response = requests.post(url, json=payload, headers=headers)

  print(f"\nПриклад #{i}")
  print(f"Текст: {text}")

  if response.status_code == 200:
    res_data = response.json()
    try:
      model_answer = (
          res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
      )
      print(f"Результат від LLM: {model_answer}")
    except (KeyError, IndexError):
      print(f"Помилка парсингу відповіді: {res_data}")
  else:
    print(
        f"Помилка запиту ({response.status_code}): {response.text}"
    )
