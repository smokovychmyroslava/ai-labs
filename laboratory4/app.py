import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"   
OUT = "results_lab4"
DELAY = 7                         
TASKS = {1, 2, 3, 4, 5, 6}        
RUNS = 3                          

os.makedirs(OUT, exist_ok=True)

TEXT = """
Основні поняття
Кібербезпека або cyber security — це заходи, які вживають для захисту даних або пристроїв, підключених до мережі, від несанкціонованого доступу та 
використання у злочинних цілях. Кібербезпека це те, що забезпечує конфіденційність, цілісність і доступність даних протягом їх всього життєвого циклу.
Якщо фізична безпека спрямована на захист від злочинної діяльності або випадкового пошкодження людей та фізичного майна, то кібербезпека захищає 
комп’ютерні та серверні системи, програми користувачів та дані, які вони в них зберігають. Кібербезпека може захистити все: від особистої інформації 
до складних державних систем.
Кіберзахист — здатність запобігати кібератакам. Це вжиття активних заходів для прогнозування кібератак з боку зловмисника та протидії вторгненням. 
Усі стратегії та тактики кіберзахисту мають спільну мету — запобігати кіберзагрозам, припиняти та реагувати на них.

Чому забезпечення кібербезпеки сьогодні важливе?
Сьогодні у кіберзлочинців набагато більше можливостей та мотивації, щоб здійснювати кібератаки: через фінансову вигоду, політичні чи соціальні мотиви. 
Чому? Як ми вже говорили вище, критична інфраструктура, включаючи охорону здоров’я, фінансові установи, уряди та виробництво, використовує комп’ютери або
інтелектуальні пристрої як основну частину своєї діяльності
За останні два десятиліття кібератаки були здійснені проти критично важливої інфраструктури у всіх розвинених країнах, і незліченна кількість підприємств 
зазнали великих збитків. Щорічно у світі відбувається понад 2000 підтверджених джерел даних, кожен з яких коштує в середньому понад 3,9 мільйона доларів. 
З 2000 року кіберзлочинці вкрали особисту інформацію понад 3,5 мільярда людей, що становить половину населення світу, тому сьогодні фахівці з кібербезпеки 
дуже затребувані, як і знання запобіжних заходів, про які ми говоритимемо нижче.
"""

def with_text(instruction, text=None):
    return f"{instruction.strip()}\n\nТекст:\n{(text or TEXT).strip()}"


def ask(prompt, system=None, temperature=None, top_p=None, max_tokens=None, json_mode=False):
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        response_mime_type="application/json" if json_mode else None,
    )
    for attempt in range(4):
        try:
            r = client.models.generate_content(model=MODEL, contents=prompt, config=config)
            time.sleep(DELAY)
            return (r.text or "").strip()
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"Помилка: {e}\nПовтор через {wait} с...")
            time.sleep(wait)
    raise RuntimeError("Не вдалося отримати відповідь від моделі")


def save(name, prompt, result):
    with open(f"{OUT}/{name}.txt", "w", encoding="utf-8") as f:
        f.write("PROMPT:\n" + prompt.strip() + "\n\nRESULT:\n" + result)
    print(f"\n=== {name} ===\n{result}\n")


if 1 in TASKS:
    plain = with_text("Проаналізуй наведений текст та визнач його основні ідеї.")
    save("01_plain_prompt", plain, ask(plain))

    advanced = with_text("""Ти — викладач університету, який готує конспект для студентів-початківців.
Контекст: наведений текст — вступ до лекції з кібербезпеки.

Виконай завдання поетапно:
1. Визнач основну тему.
2. Виділи 3–5 ключових ідей.
3. Визнач основні поняття та коротко поясни кожне.
4. Сформулюй короткий висновок (одне речення).

Обмеження: використовуй лише інформацію з тексту, не додавай власних фактів; не більше 150 слів.
Формат: заголовки «Тема», «Ключові ідеї» (нумерований список), «Поняття» (маркований список), «Висновок».""")
    save("02_advanced_prompt", advanced, ask(advanced))


if 2 in TASKS:
    p1 = with_text("Визнач 5–7 основних понять тексту (терміни, які в ньому визначаються або є ключовими). "
                   "Для кожного дай коротке визначення за текстом. "
                   "Виведи маркований список у форматі «Термін — визначення».")
    s1 = ask(p1)
    save("03_step1_concepts", p1, s1)

    p2 = with_text("Визнач ключові факти тексту: лише перевірювані твердження (числа, причини, події, порівняння), "
                   "без визначень термінів. Переказуй близько до тексту, нічого не додавай і не виправляй. "
                   "Виведи маркований список.")
    s2 = ask(p2)
    save("03_step2_facts", p2, s2)

    p3 = ("Сформуй 5 запитань за текстом. Вимоги: кожне запитання перевіряє одне поняття або один факт "
          "зі списків нижче; відповідь має прямо міститися в тексті; запитання сформульовані чітко і граматично "
          "правильно. Виведи лише нумерований список запитань.\n\n"
          f"Поняття:\n{s1}\n\nФакти:\n{s2}\n\nТекст:\n{TEXT.strip()}")
    s3 = ask(p3)
    save("03_step3_questions", p3, s3)

    p4 = ("Для кожного запитання нижче сформуй 4 варіанти відповіді (A, B, C, D): один правильний і три правдоподібні "
          "неправильні. Виведи запитання разом із варіантами.\n\n"
          f"Запитання:\n{s3}\n\nТекст:\n{TEXT.strip()}")
    s4 = ask(p4)
    save("03_step4_options", p4, s4)

    p5 = ("Для кожного запитання тесту нижче вкажи правильну відповідь (літеру) і коротко обґрунтуй її за текстом.\n\n"
          f"Тест:\n{s4}\n\nТекст:\n{TEXT.strip()}")
    s5 = ask(p5)
    save("03_step5_answers", p5, s5)

    single = with_text("Створи тест із 5 запитань за наведеним текстом. Для кожного запитання дай 4 варіанти "
                       "відповіді (A–D) і вкажи правильну відповідь.")
    save("04_single_prompt_test", single, ask(single))


if 3 in TASKS:
    react = with_text("""Проаналізуй текст та визнач, які твердження є фактами, а які — висновками автора.

Дій за такою послідовністю і підписуй кожен крок:
ЗАДАЧА: сформулюй своїми словами, що потрібно зробити.
ПЛАН: склади нумерований план аналізу.
ВИКОНАННЯ: виконай план крок за кроком.
ПЕРЕВІРКА: перевір, чи кожне твердження віднесено правильно, виправ помилки.
РЕЗУЛЬТАТ: таблиця «Твердження | Факт/Висновок | Обґрунтування».""")
    save("05_react_imitation", react, ask(react))


SCHEMA = """{
  "topic": "...",
  "key_ideas": ["...", "...", "..."],
  "keywords": ["...", "..."],
  "conclusion": "..."
}"""


def check_json(raw):
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        return f"Помилка розбору JSON: {e}"
    if not isinstance(data, dict):
        return "JSON розібрано, але це не об'єкт."
    problems = []
    missing = {"topic", "key_ideas", "keywords", "conclusion"} - set(data.keys())
    if missing:
        problems.append(f"бракує полів: {sorted(missing)}")
    for key in ("key_ideas", "keywords"):
        if key in data and not isinstance(data[key], list):
            problems.append(f"поле {key} має бути списком")
    if problems:
        return "JSON розібрано, але структура неправильна: " + "; ".join(problems)
    return (f"JSON розібрано успішно, структура коректна. "
            f"Тема: {data['topic']}; ідей: {len(data['key_ideas'])}; ключових слів: {len(data['keywords'])}.")


if 4 in TASKS:
    jprompt = with_text("Проаналізуй текст і поверни результат строго у форматі JSON без жодного додаткового "
                        "тексту та без блоків ```. Використай таку структуру:\n" + SCHEMA)

    raw_a = ask(jprompt)
    save("06_json_prompt_only", jprompt, raw_a + "\n\n--- Перевірка ---\n" + check_json(raw_a))

    raw_b = ask(jprompt, json_mode=True)  
    save("06_json_api_mode", jprompt, raw_b + "\n\n--- Перевірка ---\n" + check_json(raw_b))


TASK5 = "Склади узагальнення документа (до 200 слів) та виділи 5 ключових ідей."


def chunk_text(text, size=3000, overlap=0):
    """Ділить текст на фрагменти ~size символів, по можливості по межі речення/абзацу.
    overlap > 0 — sliding window (кожен наступний фрагмент починається з перекриттям)."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def summarize_chunked(chunks, tag):
    summaries = []
    for i, c in enumerate(chunks, 1):
        p = f"Стисло узагальни фрагмент документа (3–4 речення), збережи ключові факти.\n\nФрагмент {i} з {len(chunks)}:\n{c}"
        s = ask(p)
        summaries.append(s)
        save(f"{tag}_chunk{i:02d}", p, s)
    joined = "\n\n".join(f"Узагальнення {i}:\n{s}" for i, s in enumerate(summaries, 1))
    final_p = ("Нижче наведено узагальнення послідовних частин одного документа. "
               f"На їх основі виконай завдання для всього документа: {TASK5}\n\n" + joined)
    final = ask(final_p)
    save(f"{tag}_final", final_p, final)
    return final


if 5 in TASKS:
    if not os.path.exists("big_text.txt"):
        print("\nЗавдання 5 пропущено: покладіть великий текст у файл big_text.txt поруч зі скриптом.")
    else:
        with open("big_text.txt", encoding="utf-8") as f:
            big = f.read()
        print(f"\nДовжина тексту: {len(big)} символів")

        p = f"{TASK5}\n\nДокумент:\n{big}"
        save("07_no_chunking", p, ask(p))

        chunks = chunk_text(big, size=3000)
        print(f"Chunking: {len(chunks)} фрагментів")
        summarize_chunked(chunks, "08_chunking")

        windows = chunk_text(big, size=3000, overlap=500)
        print(f"Sliding window: {len(windows)} фрагментів")
        summarize_chunked(windows, "09_sliding")

        with open(f"{OUT}/10_task5_stats.txt", "w", encoding="utf-8") as f:
            f.write(f"Довжина тексту: {len(big)} символів\n")
            f.write(f"Chunking (size=3000): {len(chunks)} фрагментів, довжини: {[len(c) for c in chunks]}\n")
            f.write(f"Sliding window (size=3000, overlap=500): {len(windows)} фрагментів, "
                    f"довжини: {[len(c) for c in windows]}\n")


if 6 in TASKS:
    task6 = ("Придумай рекламний опис (3–4 речення) для мобільного застосунку, "
             "який допомагає студентам планувати навчання.")

    for t in (0, 0.5, 1):
        for run in range(1, RUNS + 1):
            save(f"11_temp_{t}_run{run}", f"{task6}\n\n[temperature={t}]", ask(task6, temperature=t))

    for tp in (0.5, 0.8, 1.0):
        for run in range(1, RUNS + 1):
            save(f"12_topp_{tp}_run{run}", f"{task6}\n\n[temperature=1.0, top_p={tp}]",
                 ask(task6, temperature=1.0, top_p=tp))


print("\nГотово! Результати в папці", OUT)