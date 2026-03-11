"""
День 17 — MCP Agent
===================
Агент подключается к day17_mcp_server.py через stdio transport,
вызывает инструменты и использует результаты для анализа репозитория.

Ключевое отличие от z16: здесь агент вызывает session.call_tool() —
не просто листает инструменты, а реально их использует.

Запуск:
  python zadanie17/day17_agent.py
  python zadanie17/day17_agent.py owner/repo
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _parse_result(result) -> Any:
    """
    Извлекает данные из CallToolResult.
    FastMCP сериализует:
      - dict  → 1 TextContent с JSON-строкой
      - list  → N TextContent, по одному на каждый элемент (пустой list → 0 TextContent)
    """
    if not result.content:
        return []  # пустой список — нет данных (напр. нет issues)

    def _decode(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    items = [_decode(c.text) for c in result.content if hasattr(c, "text")]

    # Один элемент и это dict — скорее всего dict-результат, не list
    if len(items) == 1 and isinstance(items[0], dict):
        return items[0]

    return items

SERVER_PATH = Path(__file__).parent / "day17_mcp_server.py"

# Репо по умолчанию — наш учебный проект
DEFAULT_OWNER = "termyter"
DEFAULT_REPO  = "llm-api-project"


# ── Ядро агента ───────────────────────────────────────────────────────────────

async def analyze_repo(owner: str, repo: str) -> dict:
    """
    Подключается к MCP-серверу и вызывает все три инструмента.
    Возвращает словарь с полными данными о репозитории.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── Вызов 1: информация о репозитории ─────────────────────────────
            print("  → get_repo_info ...", flush=True)
            result = await session.call_tool(
                "get_repo_info",
                {"owner": owner, "repo": repo},
            )
            repo_info = _parse_result(result)

            # ── Вызов 2: последние коммиты ────────────────────────────────────
            print("  → get_latest_commits ...", flush=True)
            result = await session.call_tool(
                "get_latest_commits",
                {"owner": owner, "repo": repo, "limit": 5},
            )
            commits = _parse_result(result)

            # ── Вызов 3: открытые задачи ──────────────────────────────────────
            print("  → list_open_issues ...", flush=True)
            result = await session.call_tool(
                "list_open_issues",
                {"owner": owner, "repo": repo, "limit": 5},
            )
            issues = _parse_result(result)

    return {"repo": repo_info, "commits": commits, "issues": issues}


# ── Форматирование ────────────────────────────────────────────────────────────

def format_terminal(data: dict) -> str:
    """Отчёт для терминала."""
    r = data["repo"]
    lines = [
        f"{'─' * 60}",
        f"📦  {r['name']}",
        f"    {r['description']}",
        f"    ⭐ {r['stars']} stars  🍴 {r['forks']} forks  "
        f"💻 {r['language']}  🐛 {r['open_issues']} issues",
        f"    🔗 {r['url']}",
        f"    Создан: {r['created_at']}  Обновлён: {r['updated_at']}",
        f"{'─' * 60}",
        f"🕐  Последние коммиты:",
    ]
    for c in data["commits"]:
        lines.append(f"    [{c['sha']}] {c['date']}  {c['author']}")
        lines.append(f"             {c['message']}")

    issues = data["issues"]
    lines.append(f"{'─' * 60}")
    if issues:
        lines.append(f"🐛  Открытые задачи ({len(issues)}):")
        for i in issues:
            labels = f"  [{', '.join(i['labels'])}]" if i["labels"] else ""
            lines.append(f"    #{i['number']}  {i['title']}{labels}")
            lines.append(f"         {i['url']}")
    else:
        lines.append("✅  Открытых задач нет")
    lines.append(f"{'─' * 60}")
    return "\n".join(lines)


def format_telegram(data: dict) -> str:
    """Отчёт для Telegram (Markdown)."""
    r = data["repo"]
    lines = [
        f"📦 *{r['name']}*",
        f"_{r['description']}_",
        f"",
        f"⭐ {r['stars']} stars · 🍴 {r['forks']} forks · 💻 {r['language']}",
        f"Обновлён: {r['updated_at']}",
        f"",
        f"*🕐 Последние коммиты:*",
    ]
    for c in data["commits"]:
        lines.append(f"`{c['sha']}` {c['date']} — {c['message']}")

    lines.append("")
    issues = data["issues"]
    if issues:
        lines.append(f"*🐛 Открытые задачи ({len(issues)}):*")
        for i in issues:
            lines.append(f"[#{i['number']} {i['title']}]({i['url']})")
    else:
        lines.append("✅ Открытых задач нет")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main():
    # Принимаем аргумент "owner/repo" или используем дефолт
    if len(sys.argv) > 1 and "/" in sys.argv[1]:
        owner, repo = sys.argv[1].split("/", 1)
    else:
        owner, repo = DEFAULT_OWNER, DEFAULT_REPO

    print(f"\n🤖 Агент анализирует репозиторий: {owner}/{repo}\n")
    data = await analyze_repo(owner, repo)
    print()
    print(format_terminal(data))


if __name__ == "__main__":
    asyncio.run(main())
