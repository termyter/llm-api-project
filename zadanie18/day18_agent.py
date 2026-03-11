"""
День 18 — MCP Agent: Weather via Open-Meteo
============================================
Подключается к day18_mcp_server.py через stdio,
вызывает все три погодных инструмента и выводит отчёт.

Запуск:
  python zadanie18/day18_agent.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PATH = Path(__file__).parent / "day18_mcp_server.py"


def _parse_result(result) -> Any:
    """
    FastMCP сериализует:
      - dict  → 1 TextContent с JSON-строкой
      - list  → N TextContent, по одному на элемент (пустой list → 0 TextContent)
    """
    if not result.content:
        return []

    def _decode(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    items = [_decode(c.text) for c in result.content if hasattr(c, "text")]

    if len(items) == 1 and isinstance(items[0], dict):
        return items[0]

    return items


async def get_weather_report(session: ClientSession) -> dict:
    """Вызывает все три инструмента и возвращает собранные данные."""

    # ── 1. Текущая погода по городу ──────────────────────────────────────
    result = await session.call_tool("get_current_weather", {"city": "Москва"})
    moscow = _parse_result(result)

    # ── 2. Прогноз на 3 дня ──────────────────────────────────────────────
    result = await session.call_tool(
        "get_forecast", {"city": "Санкт-Петербург", "days": 3}
    )
    spb_forecast = _parse_result(result)

    # ── 3. Погода по координатам (Париж) ─────────────────────────────────
    result = await session.call_tool(
        "get_weather_by_coords", {"latitude": 48.85, "longitude": 2.35}
    )
    paris = _parse_result(result)

    return {
        "moscow": moscow,
        "spb_forecast": spb_forecast,
        "paris": paris,
    }


def format_report(data: dict) -> str:
    lines = ["═" * 60, "  🌤  Weather Report — Open-Meteo MCP", "═" * 60]

    # Москва — текущая погода
    m = data["moscow"]
    if "error" in m:
        lines += ["", "  Москва: ❌ " + m["error"]]
    else:
        lines += [
            "",
            f"  📍 {m['city']}, {m['country']}",
            f"     Температура : {m['temperature']:+.1f}°C  "
            f"(ощущается {m['feels_like']:+.1f}°C)",
            f"     Влажность   : {m['humidity']}%",
            f"     Ветер       : {m['wind_speed']} км/ч",
            f"     Состояние   : {m['description']}",
        ]

    lines += ["", "─" * 60]

    # Санкт-Петербург — прогноз
    forecast = data["spb_forecast"]
    lines += ["", "  📅 Прогноз — Санкт-Петербург (3 дня)", ""]
    if not forecast or (isinstance(forecast, list) and "error" in forecast[0]):
        lines.append("  ❌ Ошибка получения прогноза")
    else:
        for day in forecast:
            precip = day.get("precipitation") or 0
            rain_str = f"  💧 {precip} мм" if precip else ""
            lines.append(
                f"     {day['date']}  "
                f"{day['temp_min']:+.0f}…{day['temp_max']:+.0f}°C  "
                f"{day['description']}{rain_str}"
            )

    lines += ["", "─" * 60]

    # Париж — по координатам
    p = data["paris"]
    if "error" in p:
        lines += ["", "  Paris (48.85°N, 2.35°E): ❌ " + p["error"]]
    else:
        lines += [
            "",
            f"  📍 Paris  ({p['latitude']}°N, {p['longitude']}°E)",
            f"     Температура : {p['temperature']:+.1f}°C  "
            f"(ощущается {p['feels_like']:+.1f}°C)",
            f"     Влажность   : {p['humidity']}%",
            f"     Ветер       : {p['wind_speed']} км/ч",
            f"     Состояние   : {p['description']}",
        ]

    lines += ["", "═" * 60]
    return "\n".join(lines)


async def main():
    print("⏳ Запускаю MCP-сервер и подключаюсь...")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Соединение установлено\n")

            data = await get_weather_report(session)
            print(format_report(data))


if __name__ == "__main__":
    asyncio.run(main())
