import json
import os
import re
import time
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={API_KEY}"

QUERY_PROMPT = (
    "Ти агент фактчекінгу. Отримуєш твердження і самостійно формуєш 2-3 короткі "
    "пошукові запити для його перевірки в інтернеті. "
    "Відповідай суворо у форматі JSON: {\"queries\": [\"запит 1\", \"запит 2\"]}"
)

VERIFY_PROMPT = (
    "Ти агент фактчекінгу. Отримуєш твердження та результати пошуку з незалежних джерел. "
    "Зроби детальний розбір у структурованому вигляді:\n"
    "1. Перевірка ключових тез: розбери кожне твердження з тексту окремо, вкажи, що підтверджується, "
    "а що спростовується знайденими джерелами (з посиланням на домени).\n"
    "2. Помилки та маніпуляції: чітко поясни, де є фактичні помилки або викривлення.\n"
    "3. Правдивість інформації: вкажи відсоток (0-100%) з коротким поясненням.\n"
    "4. Фінальний вердикт: фейк, правда або маніпуляція."
)


def get_domain(url):
    netloc = urlparse(url).netloc.lower()
    return netloc.removeprefix("www.")


def search_sources(query, min_sources=3):
    picked = []
    seen = set()
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=8):
                if not isinstance(r, dict):
                    continue
                href = str(r.get("href", "")).strip()
                title = str(r.get("title", "")).strip()
                body = str(r.get("body", "")).strip()
                if not (title or body):
                    continue
                domain = get_domain(href) if href else f"source_{len(picked) + 1}"
                if domain in seen:
                    continue
                seen.add(domain)
                picked.append(f"- [{domain}] {title}: {body}")
                if len(picked) >= min_sources:
                    break
    except Exception as e:
        return f"Помилка пошуку: {e}"
    return "\n".join(picked) if picked else "Незалежних джерел не знайдено."


def get_retry_sec(text):
    m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", text)
    return int(float(m.group(1))) + 2 if m else 20


def call_gemini(system_prompt, text):
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0.2},
    }
    for attempt in range(5):
        try:
            resp = requests.post(URL, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        except requests.RequestException as e:
            print(f"Збій з'єднання: {e}. Повтор через 3 с...")
            time.sleep(3)
            continue
        if resp.status_code == 429:
            wait = get_retry_sec(resp.text)
            print(f"Ліміт запитів, чекаю {wait} с...")
            time.sleep(wait)
            continue
        if resp.status_code == 503:
            wait = 3 * (attempt + 1)
            print(f"Модель перевантажена (503), чекаю {wait} с...")
            time.sleep(wait)
            continue
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
            raise RuntimeError("Некоректна відповідь API")
        return data
    raise RuntimeError("Запит не виконано після кількох спроб.")


def get_text(data):
    candidates = data.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)).strip()


def ask_agent(prompt, text, stats):
    stats["steps"] += 1
    out = get_text(call_gemini(prompt, text))
    return out if out else "Відповідь не сформовано."


def build_queries(claim, stats):
    raw = ask_agent(QUERY_PROMPT, claim, stats).strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
        items = parsed.get("queries", []) if isinstance(parsed, dict) else []
        queries = [str(q).strip() for q in items if isinstance(q, str) and q.strip()][:3]
        if queries:
            return queries
    except Exception:
        pass
    return [" ".join(claim.split()[:6])]


def run_single_agent(claim, stats):
    queries = build_queries(claim, stats)
    blocks = []
    for idx, q in enumerate(queries, 1):
        found = search_sources(q, min_sources=3)
        blocks.append(f"Запит {idx} ({q}):\n{found}")
    sources_block = "\n\n".join(blocks)
    prompt = f"Твердження: {claim}\n\nНезалежні джерела:\n{sources_block}"
    verdict = ask_agent(VERIFY_PROMPT, prompt, stats)
    return queries, sources_block, verdict


def main():
    claims = [
        "Офіційно першим програмістом в історії вважається Ада Лавлейс, яка у 1843 році написала перший комп'ютерний вірус на мові Python для аналітичної машини Чарльза Беббіджа, через що уряд Великої Британії заборонив цю машину до 1950 року.",
        "Після ухвалення Європарламентом закону AI Act у 2024 році, на території всього Євросоюзу повністю заборонили використання генеративних нейромереж для студентів і розробників під загрозою штрафу до 35 мільйонів євро.",
        "Міжнародна космічна станція літає на висоті 400 км у зоні повної відсутності земної гравітації, тому космонавти перебувають у невагомості, а сама станція залишається на орбіті лише завдяки постійній роботі реактивних двигунів.",
    ]

    for i, claim in enumerate(claims, 1):
        t0 = time.perf_counter()
        stats = {"steps": 0}
        queries, sources_block, verdict = run_single_agent(claim, stats)
        elapsed = time.perf_counter() - t0

        print(f"Задача {i}: {claim}")
        print(f"Запити агента: {queries}")
        print("Незалежні джерела:")
        print(sources_block)
        print("\nВердикт агента:")
        print(verdict)
        print(f"Кроків: {stats['steps']} | Час: {elapsed:.2f} с\n")


if __name__ == "__main__":
    main()