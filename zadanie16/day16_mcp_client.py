"""
День 16 — MCP Client
====================
Подключается к day16_mcp_server.py через stdio transport,
получает список инструментов и выводит их описания.

Запуск:  python zadanie16/day16_mcp_client.py
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PATH = Path(__file__).parent / "day16_mcp_server.py"


async def connect_and_list_tools() -> list[dict]:
    """
    Устанавливает соединение с MCP-сервером через stdio,
    вызывает list_tools() и возвращает список инструментов.
    """
    server_params = StdioServerParameters(
        command=sys.executable,          # тот же python/venv
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # ── 1. Handshake / инициализация ─────────────────────────────────
            info = await session.initialize()
            server_name    = info.serverInfo.name
            server_version = info.serverInfo.version

            # ── 2. Получаем список инструментов ──────────────────────────────
            result = await session.list_tools()
            tools  = result.tools

            return {
                "server_name":    server_name,
                "server_version": server_version,
                "tools":          tools,
            }


def format_for_terminal(data: dict) -> str:
    """Форматирует результат для вывода в терминале."""
    lines = [
        "═" * 60,
        f"  MCP Server: {data['server_name']}  v{data['server_version']}",
        "═" * 60,
        f"  Доступных инструментов: {len(data['tools'])}",
        "─" * 60,
    ]
    for i, tool in enumerate(data["tools"], 1):
        # Первая непустая строка описания
        desc_lines = [l.strip() for l in (tool.description or "—").split("\n") if l.strip()]
        desc_first_line = desc_lines[0] if desc_lines else "—"
        lines.append(f"  {i}. {tool.name}")
        lines.append(f"     {desc_first_line}")
        # Параметры из inputSchema
        props = (tool.inputSchema or {}).get("properties", {})
        if props:
            params = ", ".join(
                f"{k}: {v.get('type', '?')}" for k, v in props.items()
            )
            lines.append(f"     Параметры: ({params})")
        lines.append("")
    lines.append("═" * 60)
    return "\n".join(lines)


def format_for_telegram(data: dict) -> str:
    """Форматирует результат для Telegram (Markdown)."""
    lines = [
        f"🔌 *Подключено к MCP-серверу*",
        f"`{data['server_name']}` v`{data['server_version']}`",
        "",
        f"📋 *Доступных инструментов: {len(data['tools'])}*",
        "",
    ]
    for i, tool in enumerate(data["tools"], 1):
        _d = [l.strip() for l in (tool.description or "—").split("\n") if l.strip()]
        desc_first_line = _d[0] if _d else "—"
        props = (tool.inputSchema or {}).get("properties", {})
        params_str = ""
        if props:
            params_str = " • `" + "`, `".join(props.keys()) + "`"
        lines.append(f"{i}\\. `{tool.name}`")
        lines.append(f"   _{desc_first_line}_")
        if params_str:
            lines.append(f"   Параметры:{params_str}")
        lines.append("")
    return "\n".join(lines)


async def main():
    print("⏳ Подключаюсь к MCP-серверу...")
    try:
        data = await connect_and_list_tools()
        print("✅ Соединение установлено!\n")
        print(format_for_terminal(data))
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
