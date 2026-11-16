from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
MODEL = "gemini-3.5-flash-lite"

def ask_gemini(prompt):
    response = client.interactions.create(
        model=MODEL,
        input=prompt
    )
    return response.output_text


def save_result(filename, prompt, result):
    with open(filename, "w", encoding="utf-8") as file:
        file.write("PROMPT:\n")
        file.write(prompt.strip())
        file.write("\n\nRESULT:\n")

        lines = result.splitlines()
        result = "\n".join(line.strip() for line in lines if line.strip())
        file.write(result)

prompts = [
    {
        "name": "01_explanation",
        "type": "Пояснення",
        "prompt": """
Поясни простими словами, що таке API.
Розкажи, для чого воно використовується
та наведи один приклад із повсякденного життя.
Відповідь має бути зрозумілою для початківця.
"""
    },

    {
        "name": "02_text_generation",
        "type": "Генерація тексту",
        "prompt": """
Напиши невеликий текст на тему
«Роль мови програмування Python у сучасних IT-технологіях».

Розкажи про основні сфери її використання,
переваги та популярність.
Обсяг — 100–150 слів.
"""
    },

    {
        "name": "03_translation",
        "type": "Переклад",
        "prompt": """
Переклади наведений текст з англійської
мови на українську:

Python is a popular programming language
used in web development, data analysis,
artificial intelligence and automation.
It has a simple syntax, which makes it
a good choice for beginners.
"""
    },

    {
        "name": "04_summarization",
        "type": "Узагальнення",
        "prompt": """
Скороти та узагальни наведений текст,
зберігши його основну думку.

Текст:
Інтернет став невід’ємною частиною нашого життя. 
Ми використовуємо його для навчання, спілкування з друзями,
розвитку хобі та розваг. Але чим більше нашого життя переходить
в онлайн, тим ціннішою стає інформація, яку ми там залишаємо,
і тим привабливішою вона є для зловмисників. Саме тому важливо 
розуміти, що таке кібербезпека.
Кібербезпека — це набір практик та технологій, призначених для
захисту наших пристроїв (комп’ютерів, смартфонів), мереж та даних
від несанкціонованого доступу, атак та використання у злочинних цілях.
Простими словами, це набір правил і технологій, які гарантують, що 
доступ до вашої інформації маєте тільки ви, що її ніхто не змінить без
вашого відома, і що вона буде доступна вам, коли це потрібно.
"""
    },

    {
        "name": "05_list_generation",
        "type": "Генерація списку",
        "prompt": """
Склади список із 7 корисних інструментів
для початківця у програмуванні.

Для кожного інструмента вкажи:
1. Назву.
2. Для чого він використовується.
3. Його основну перевагу.

Оформи відповідь нумерованим списком.
"""
    }
]


for item in prompts:
    print("\n" + "=" * 40)
    print("Завдання:", item["type"])
    print("=" * 40)

    try:
        result = ask_gemini(item["prompt"])
        print(result)
        filename = "results/" + item["name"] + ".txt"
        save_result(filename, item["prompt"], result)
        print("\nРезультат збережено у:", filename)

    except Exception as error:
        print("Виникла помилка:", error)

print("\nЗавдання 1 виконано!")


zero_shot_prompt = """
Визнач емоційне забарвлення повідомлення.

Можливі категорії:
- позитивне;
- негативне;
- нейтральне.

Повідомлення:
"Програма працює дуже повільно,
але загалом результат мене влаштовує."

Вкажи категорію та коротко поясни свій вибір.
"""


few_shot_prompt = """
Визнач емоційне забарвлення повідомлень.

Приклади:

Повідомлення: "Я дуже задоволений результатом!"
Категорія: позитивне.

Повідомлення: "Програма постійно видає помилки."
Категорія: негативне.

Повідомлення: "Сьогодні я встановив нову програму."
Категорія: нейтральне.

Тепер класифікуй нове повідомлення:

"Програма працює дуже повільно,
але загалом результат мене влаштовує."

Вкажи категорію та коротко поясни свій вибір.
Дотримуйся формату наведених прикладів.
"""


print("\n ZERO-SHOT ")

zero_result = ask_gemini(zero_shot_prompt)

print(zero_result)

save_result(
    "results/06_zero_shot.txt",
    zero_shot_prompt,
    zero_result
)


print("\n FEW-SHOT ")

few_result = ask_gemini(few_shot_prompt)

print(few_result)

save_result(
    "results/07_few_shot.txt",
    few_shot_prompt,
    few_result
)

print("\nЗавдання 2 виконано!")


simple_prompt = "Розкажи про Python."


improved_prompt = """
Ти — викладач програмування для початківців.

Поясни, що таке мова програмування Python
та для чого вона використовується.

Розкажи про:
1. Основні сфери застосування Python.
2. Переваги цієї мови.
3. Чому вона підходить для початківців.

Використовуй просту та зрозумілу мову.
Складні технічні терміни пояснюй.

Обсяг відповіді — 150–200 слів.
Оформи відповідь із заголовками та списком.

Цільова аудиторія — студенти-початківці,
які тільки починають вивчати програмування.
"""


print("\n ПОЧАТКОВИЙ PROMPT")

simple_result = ask_gemini(simple_prompt)

print(simple_result)

save_result(
    "results/08_simple_prompt.txt",
    simple_prompt,
    simple_result
)


print("\n УДОСКОНАЛЕНИЙ PROMPT")

improved_result = ask_gemini(improved_prompt)

print(improved_result)

save_result(
    "results/09_improved_prompt.txt",
    improved_prompt,
    improved_result
)

print("\nЗавдання 3 виконано!")

roles = [
    {
        "name": "10_role_teacher",
        "role": "Викладач",
        "prompt": """
Ти — викладач інформатики.

Поясни студенту-початківцю, що таке API.
Використовуй прості слова та аналогії
з повсякденного життя.

Наведи один приклад.
"""
    },

    {
        "name": "11_role_programmer",
        "role": "Програміст",
        "prompt": """
Ти — досвідчений програміст.

Поясни, що таке API з технічної точки зору.
Розкажи, як програми взаємодіють між собою
за допомогою API.

Наведи практичний приклад використання API
у розробці програмного забезпечення.
"""
    },

    {
        "name": "12_role_scientist",
        "role": "Науковець",
        "prompt": """
Ти — науковець у галузі інформаційних технологій.

Поясни поняття API, його призначення
та роль у сучасних інформаційних системах.

Використовуй точні терміни
та науковий стиль викладу.
"""
    }
]


for item in roles:
    print("\n" + "=" * 40)
    print("СИСТЕМНА РОЛЬ:", item["role"])
    print("=" * 40)

    try:
        result = ask_gemini(item["prompt"])
        print(result)

        save_result(
            "results/" + item["name"] + ".txt",
            item["prompt"],
            result
        )

        print("\nРезультат збережено!")

    except Exception as error:
        print("Виникла помилка:", error)

print("\nЗавдання 4 виконано!")
