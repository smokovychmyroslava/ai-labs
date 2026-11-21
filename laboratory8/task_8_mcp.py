import asyncio
import json
import os
import re
import sys
import time
import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={API_KEY}"


def get_retry_sec(text: str) -> int:
    m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", text)
    return int(float(m.group(1))) + 2 if m else 20


def call_gemini(system_prompt: str, user_text: str) -> str:
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.1},
    }
    for attempt in range(5):
        try:
            resp = requests.post(URL, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        except requests.RequestException:
            time.sleep(3)
            continue
        if resp.status_code == 429:
            time.sleep(get_retry_sec(resp.text))
            continue
        if resp.status_code == 503:
            time.sleep(3 * (attempt + 1))
            continue
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)).strip()
    raise RuntimeError("Не вдалося отримати відповідь від API.")


def parse_tool_call(raw_text: str) -> dict | None:
    clean = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(clean)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


async def run_demo(user_query: str) -> None:
    print(f"Користувач: {user_query}")
    params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"Доступні MCP tools: {[t.name for t in tools.tools]}")

            plan_prompt = (
                "Обери інструмент, якщо він потрібен, і поверни строго JSON:\n"
                '{"tool": "calculate", "arguments": {"expression": "<вираз>"}}\n'
                "або\n"
                '{"tool": "text_stats", "arguments": {"text": "<текст>"}}\n'
                "або\n"
                '{"tool": "api_info", "arguments": {"topic": "<тема англійською для Wikipedia API>"}}\n'
                "Якщо інструмент не потрібен, поверни звичайний текст."
            )
            decision = call_gemini(plan_prompt, user_query)
            print(f"AI-модель -> MCP: {decision}")

            parsed = parse_tool_call(decision)
            tool_output = None

            if parsed and isinstance(parsed.get("tool"), str):
                tool_name = parsed["tool"]
                args = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
                print(f"MCP -> Зовнішній інструмент ({tool_name}): {args}")
                res = await session.call_tool(tool_name, arguments=args)
                tool_output = res.content[0].text if res.content else ""
                print(f"Зовнішній інструмент -> Результат: {tool_output}")

            context = f"Запит користувача: {user_query}"
            if tool_output is not None:
                context += f"\nРезультат інструмента: {tool_output}"

            answer = call_gemini("Дай стислу та точну відповідь українською мовою.", context)
            print(f"AI-модель -> Відповідь: {answer}")


async def main() -> None:
    await run_demo("Порахуй (240 * 15) + 85")
    print()
    await run_demo("Порахуй скільки слів і символів у фразі: Перевірка роботи MCP сервера у Python")
    print()
    await run_demo("Дай коротку довідку через API про Ada Lovelace")


if __name__ == "__main__":
    asyncio.run(main())