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

PLANNER_PROMPT = (
    "Ти агент-планувальник фактчекінгу. Отримуєш інформацію для перевірки, "
    "складаєш план роботи: визначаєш порядок кроків (step: 1, 2, 3) та формуєш "
    "основні короткі пошукові запити, які виконавець має перевірити у кількох "
    "незалежних джерелах. Відповідай у JSON: "
    "{\"items\": [{\"step\": 1, \"query\": \"...\", \"sources_needed\": 3}]}"
)

EXECUTOR_PROMPT = (
    "Ти агент-виконавець фактчекінгу. Виконуєш пошуковий запит за планом "
    "планувальника, аналізуєш знайдені незалежні джерела та повертаєш результат "
    "планувальнику. Коротко зазнач, що кажуть різні джерела."
)

CHECKER_PROMPT = (
    "Ти агент-перевіряльник фактчекінгу. Аналізуєш зібрані планувальником "
    "результати перевірки в кількох джерелах, знаходиш підтвердження або "
    "спростування, вказуєш правдивість у відсотках (0-100%) і виносиш "
    "фінальний вердикт: фейк, правда або маніпуляція."
)


def get_domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc.removeprefix("www.")


def search_independent_sources(query: str, min_sources: int = 3) -> str:
    queries = [query]
    short_q = " ".join(query.split()[:4]).strip()
    if short_q and short_q != query:
        queries.append(short_q)

    picked = []
    seen_domains = set()
    last_err = None

    for q in queries:
        for _ in range(2):
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(q, max_results=8))
                    for r in raw:
                        if not isinstance(r, dict):
                            continue
                        href = str(r.get("href", "")).strip()
                        title = str(r.get("title", "")).strip()
                        body = str(r.get("body", "")).strip()
                        if not (title or body):
                            continue
                        domain = get_domain(href) if href else f"source_{len(picked) + 1}"
                        if domain in seen_domains:
                            continue
                        seen_domains.add(domain)
                        picked.append(f"- [{domain}] {title}: {body}")
                        if len(picked) >= min_sources:
                            return "\n".join(picked)
                    break
            except Exception as e:
                if "No results found" in str(e):
                    break
                time.sleep(1)
                last_err = e

    if picked:
        return "\n".join(picked)
    if last_err is not None:
        return f"Помилка пошуку: {last_err}"
    return "Незалежних джерел за запитом не знайдено."


def get_retry_sec(text: str) -> int:
    m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", text)
    return int(float(m.group(1))) + 2 if m else 20


def call_gemini(system_prompt: str, text: str) -> dict:
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0.2},
    }

    for _ in range(4):
        try:
            resp = requests.post(
                URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
        except requests.RequestException as e:
            print(f"Збій з'єднання: {e}. Повтор через 3 с...")
            time.sleep(3)
            continue

        if resp.status_code == 429:
            wait = get_retry_sec(resp.text)
            print(f"Ліміт запитів, чекаю {wait} с...")
            time.sleep(wait)
            continue
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
            raise RuntimeError("Некоректна відповідь API")
        return data

    raise RuntimeError("Запит не виконано після кількох спроб.")


def get_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    return "".join(
        p.get("text", "")
        for p in parts
        if isinstance(p, dict) and isinstance(p.get("text"), str)
    ).strip()


def ask_role(prompt: str, text: str, stats: dict) -> str:
    stats["steps"] += 1
    out = get_text(call_gemini(prompt, text))
    return out if out else "Відповідь не сформовано."


def plan_work(claim: str, stats: dict) -> list[dict]:
    raw = ask_role(PLANNER_PROMPT, claim, stats).strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        plan = []
        for idx, x in enumerate(items, 1):
            if isinstance(x, dict) and x.get("query"):
                try:
                    step = int(x.get("step", idx))
                except Exception:
                    step = idx
                try:
                    sources = max(2, min(4, int(x.get("sources_needed", 3))))
                except Exception:
                    sources = 3
                plan.append({
                    "step": step,
                    "query": str(x["query"]).strip(),
                    "sources_needed": sources,
                })
        plan.sort(key=lambda item: item["step"])
        if plan:
            return plan[:3]
    except Exception:
        pass
    return [{"step": 1, "query": " ".join(claim.split()[:5]), "sources_needed": 3}]


def execute_query(step_item: dict, stats: dict) -> str:
    stats["steps"] += 1
    sources_text = search_independent_sources(
        step_item["query"], min_sources=step_item["sources_needed"]
    )
    prompt = (
        f"Крок {step_item['step']}. Пошуковий запит: {step_item['query']}\n"
        f"Перевірити у незалежних джерелах (ціль: {step_item['sources_needed']}):\n"
        f"{sources_text}"
    )
    summary = ask_role(EXECUTOR_PROMPT, prompt, stats)
    return (
        f"Крок {step_item['step']} | Запит: {step_item['query']}\n"
        f"Висновок виконавця:\n{summary}\n"
        f"Незалежні джерела:\n{sources_text}"
    )


def planner_collect(claim: str, plan: list[dict], findings: list[str]) -> str:
    steps_order = ", ".join(str(p["step"]) for p in plan)
    return (
        f"Інформація для перевірки: {claim}\n"
        f"Порядок кроків планувальника: {steps_order}\n\n"
        + "\n\n".join(findings)
    )


def verify_findings(collected_report: str, stats: dict) -> str:
    return ask_role(CHECKER_PROMPT, collected_report, stats)


def main():
    claims = [
        "Офіційно першим програмістом в історії вважається Ада Лавлейс, яка у 1843 році написала перший комп'ютерний вірус на мові Python для аналітичної машини Чарльза Беббіджа, через що уряд Великої Британії заборонив цю машину до 1950 року.",
        "Людський мозок працює лише на 10% зі 100 можливих.",
        "Міжнародна космічна станція літає на висоті 400 км у зоні повної відсутності земної гравітації, тому космонавти перебувають у невагомості, а сама станція залишається на орбіті лише завдяки постійній роботі реактивних двигунів.",
    ]

    for i, claim in enumerate(claims, 1):
        t0 = time.perf_counter()
        stats = {"steps": 0}

        plan = plan_work(claim, stats)
        returned_to_planner = []
        for item in plan:
            print(f"Крок {item['step']}: запит '{item['query']}' (джерел: {item['sources_needed']})")
            res = execute_query(item, stats)
            returned_to_planner.append(res)

        report_for_checker = planner_collect(claim, plan, returned_to_planner)
        verdict = verify_findings(report_for_checker, stats)
        elapsed = time.perf_counter() - t0

        print(f"\nЗадача {i}: {claim}")
        print("План планувальника:")
        for p in plan:
            print(f"{p['step']}. Запит: {p['query']} (незалежних джерел: {p['sources_needed']})")
        print("Результати, повернуті планувальнику:")
        for r in returned_to_planner:
            print(r)
        print("Вердикт перевіряльника:")
        print(verdict)
        print(f"Кроків: {stats['steps']} | Час: {elapsed:.2f} с\n")


if __name__ == "__main__":
    main()