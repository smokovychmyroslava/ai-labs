import urllib.parse
import requests
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("DemoToolsServer")


@mcp.tool()
def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not expression or not set(expression) <= allowed:
        return "Неприпустимий вираз."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Помилка обчислення: {exc}"


@mcp.tool()
def text_stats(text: str) -> str:
    words = [w for w in text.split() if w.strip()]
    chars = len(text)
    return f"Слів: {len(words)}, символів: {chars}"


@mcp.tool()
def api_info(topic: str) -> str:
    clean_topic = topic.strip()
    if not clean_topic:
        return "Порожній запит для API."
    encoded = urllib.parse.quote(clean_topic.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    try:
        resp = requests.get(url, headers={"User-Agent": "MCPDemoLab/1.0"}, timeout=15)
        if not resp.ok:
            return f"API повернуло статус {resp.status_code} для запиту '{clean_topic}'."
        data = resp.json()
        title = data.get("title", clean_topic)
        extract = data.get("extract", "Опис відсутній.")
        return f"{title}: {extract}"
    except Exception as exc:
        return f"Помилка запиту до API: {exc}"


if __name__ == "__main__":
    mcp.run()