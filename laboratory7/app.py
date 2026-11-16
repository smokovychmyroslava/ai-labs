import os
import requests
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={API_KEY}"

chat_history = []


def calculator(expression: str) -> str:
    try:
        allowed_chars = "0123456789+-*/().% "
        if not all(c in allowed_chars for c in expression):
            return "Помилка: недопустимі символи у виразі."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Помилка обчислення: {e}"


def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=2)]
            if results:
                return "\n".join(results)
            return "Нічого не знайдено в інтернеті."
    except Exception as e:
        return f"Помилка пошуку: {e}"


tools_schema = [{
    "function_declarations": [
        {
            "name": "calculator",
            "description": "Виконує математичні обчислення. Викликай, коли потрібно порахувати числа або розв'язати приклад.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "expression": {
                        "type": "STRING",
                        "description": "Математичний вираз для обчислення, наприклад '128 * 16 + 450'"
                    }
                },
                "required": ["expression"]
            }
        },
        {
            "name": "search_web",
            "description": "Шукає актуальну інформацію, визначення чи навчальні матеріали в інтернеті.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "Пошуковий запит для пошукової системи"
                    }
                },
                "required": ["query"]
            }
        }
    ]
}]


def call_gemini(prompt: str) -> str:
    full_prompt = (
        "Ти — корисний AI-агент, студентський помічник. "
        "Ти вмієш пояснювати навчальні поняття, формувати плани підготовки, "
        "створювати тестові запитання та виконувати обчислення.\n"
    )
    
    for message in chat_history:
        full_prompt += f"{message['role']}: {message['content']}\n"
    
    full_prompt += f"Користувач: {prompt}\nАгент:"

    payload = {
        "contents": [{
            "parts": [{"text": full_prompt}]
        }]
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        res_data = response.json()
        try:
            return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            return "Помилка парсингу відповіді API."
    else:
        return f"Помилка API ({response.status_code}): {response.text}"


def run_agent(user_query: str):
    print(f"\n[Запит користувача]: {user_query}")
    
    plan_prompt = (
        f"Сформуй детальний план дій з 4 пунктів для виконання завдання: '{user_query}'. "
        "Напиши 4 нумеровані пункти плану."
    )
    plan = call_gemini(plan_prompt)
    print(f"[План агента (4 кроки)]:\n{plan}")

    api_contents = []
    for msg in chat_history:
        role = "user" if msg["role"] == "Користувач" else "model"
        api_contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })
    
    api_contents.append({
        "role": "user",
        "parts": [{"text": user_query}]
    })

    payload = {
        "system_instruction": {
            "parts": [{"text": "Ти — корисний AI-агент, студентський помічник. Відповідай чітко українською мовою."}]
        },
        "contents": api_contents,
        "tools": tools_schema
    }

    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    res_data = response.json()

    tool_name = "None"
    tool_result = ""
    answer = ""

    if "candidates" in res_data:
        candidate_content = res_data["candidates"][0]["content"]
        part = candidate_content["parts"][0]

        if "functionCall" in part:
            func_call = part["functionCall"]
            tool_name = func_call["name"]
            func_args = func_call["args"]

            if tool_name == "calculator":
                tool_result = calculator(func_args.get("expression", ""))
            elif tool_name == "search_web":
                tool_result = search_web(func_args.get("query", ""))
            else:
                tool_result = "Невідомий інструмент"

            print(f"[Використаний інструмент]: {tool_name} -> Результат: {tool_result}")

            api_contents.append(candidate_content)
            api_contents.append({
                "role": "function",
                "parts": [{
                    "functionResponse": {
                        "name": tool_name,
                        "response": {"result": tool_result}
                    }
                }]
            })

            final_payload = {
                "contents": api_contents,
                "tools": tools_schema
            }
            final_response = requests.post(url, json=final_payload, headers={"Content-Type": "application/json"})
            final_data = final_response.json()
            
            if "candidates" in final_data:
                answer = final_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                answer = "Помилка отримання фінальної відповіді від API."
        else:
            tool_name = "None"
            print("[Використаний інструмент]: Не потрібен (пряма відповідь моделі)")
            if "text" in part:
                answer = part["text"].strip()
            else:
                answer = "Помилка парсингу відповіді."
    else:
        answer = f"Помилка API: {res_data}"

    print(f"[Відповідь агента]: {answer}")

    chat_history.append({"role": "Користувач", "content": user_query})
    chat_history.append({"role": "Агент", "content": answer})
    
    return plan, tool_name, answer


if __name__ == "__main__":
    print("Запуск тестування AI-агента (Помічник студента)")
    
    print("\n Перевірка пам'яті (Завдання 5)")
    chat_history.clear()
    
    run_agent("Знайди мені корисні ресурси для вивчення цієї теми 'Реляційні бази даних")
    run_agent("Створи для мене 4 тестових запитання для перевірки знань з цієї теми.")
    run_agent("А тепер склади зручний план підготовки до іспиту з урахуванням цих завдань.")
    
    print("\nТестування агента (Завдання 6)")
    chat_history.clear()
    
    test_queries = [
        "Поясни простими словами, що таке об'єктно-орієнтоване програмування.",   
        "Сформуй короткий план підготовки до сесійного тижня з програмування.",   
        "Створи 3 тестових запитання з варіантами відповідей на тему ООП.",       
        "Обчисли значення виразу: 128 * 16 + 450",                               
        "Який результат вийшов у попередньому математичному обчисленні?",          
        "Скільки буде 100 / 0?",
        "Обчисли синус 30"  
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\nТест №{i}")
        run_agent(query)