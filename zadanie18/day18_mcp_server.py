"""
День 18 — MCP Server: Open-Meteo Weather Tools
===============================================
MCP-сервер, оборачивающий Open-Meteo API.
Бесплатно, без API ключа, покрывает весь мир.

Инструменты:
  - get_current_weather(city)    → текущая погода
  - get_forecast(city, days)     → прогноз на N дней
  - get_weather_by_coords(lat, lon) → текущая погода по координатам

Запуск сервера:
  python zadanie18/day18_mcp_server.py
"""

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="Weather Tools",
    instructions=(
        "Погодный MCP-сервер на базе Open-Meteo API. "
        "Используй get_current_weather для текущей погоды по названию города, "
        "get_forecast для прогноза на несколько дней, "
        "get_weather_by_coords для точных координат."
    ),
)

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO Weather Interpretation Codes → человекочитаемые описания
_WMO_CODES = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "морозный туман",
    51: "лёгкая морось",
    53: "морось",
    55: "сильная морось",
    61: "небольшой дождь",
    63: "дождь",
    65: "сильный дождь",
    71: "небольшой снег",
    73: "снег",
    75: "сильный снег",
    77: "снежная крупа",
    80: "небольшой ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "снежный ливень",
    86: "сильный снежный ливень",
    95: "гроза",
    96: "гроза с небольшим градом",
    99: "гроза с градом",
}


def _describe_wmo(code: int) -> str:
    return _WMO_CODES.get(code, f"код {code}")


def _geocode(city: str) -> dict | None:
    """Преобразует название города в координаты."""
    r = httpx.get(
        _GEO_URL,
        params={"name": city, "count": 1, "language": "ru"},
        timeout=10,
    )
    if r.status_code != 200:
        return None
    results = r.json().get("results")
    if not results:
        return None
    g = results[0]
    return {
        "name": g.get("name", city),
        "country": g.get("country", ""),
        "latitude": g["latitude"],
        "longitude": g["longitude"],
        "timezone": g.get("timezone", "auto"),
    }


@mcp.tool()
def get_current_weather(city: str) -> dict:
    """
    Возвращает текущую погоду для заданного города.

    Параметры:
        city: Название города на любом языке (Москва, London, Paris)

    Возвращает dict с полями:
        city, country, temperature (°C), feels_like (°C),
        humidity (%), wind_speed (км/ч), description, weathercode
    """
    geo = _geocode(city)
    if geo is None:
        return {"error": f"Город не найден: {city}"}

    r = httpx.get(
        _FORECAST_URL,
        params={
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
            "current": (
                "temperature_2m,"
                "apparent_temperature,"
                "relative_humidity_2m,"
                "wind_speed_10m,"
                "weathercode"
            ),
            "timezone": geo["timezone"],
        },
        timeout=10,
    )
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "detail": r.text[:200]}

    w = r.json()["current"]
    return {
        "city": geo["name"],
        "country": geo["country"],
        "temperature": w["temperature_2m"],
        "feels_like": w["apparent_temperature"],
        "humidity": w["relative_humidity_2m"],
        "wind_speed": w["wind_speed_10m"],
        "weathercode": w["weathercode"],
        "description": _describe_wmo(w["weathercode"]),
    }


@mcp.tool()
def get_forecast(city: str, days: int = 5) -> list:
    """
    Возвращает прогноз погоды на несколько дней.

    Параметры:
        city: Название города
        days: Количество дней (1–7, по умолчанию 5)

    Возвращает список dict, каждый с полями:
        date, temp_max (°C), temp_min (°C),
        precipitation (мм), description, weathercode
    """
    days = max(1, min(days, 7))
    geo = _geocode(city)
    if geo is None:
        return [{"error": f"Город не найден: {city}"}]

    r = httpx.get(
        _FORECAST_URL,
        params={
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
            "daily": (
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_sum,"
                "weathercode"
            ),
            "forecast_days": days,
            "timezone": geo["timezone"],
        },
        timeout=10,
    )
    if r.status_code != 200:
        return [{"error": f"HTTP {r.status_code}"}]

    daily = r.json()["daily"]
    return [
        {
            "date": daily["time"][i],
            "temp_max": daily["temperature_2m_max"][i],
            "temp_min": daily["temperature_2m_min"][i],
            "precipitation": daily["precipitation_sum"][i],
            "weathercode": daily["weathercode"][i],
            "description": _describe_wmo(daily["weathercode"][i]),
        }
        for i in range(len(daily["time"]))
    ]


@mcp.tool()
def get_weather_by_coords(latitude: float, longitude: float) -> dict:
    """
    Возвращает текущую погоду по координатам (когда название города неизвестно).

    Параметры:
        latitude:  широта (например 55.75 для Москвы)
        longitude: долгота (например 37.62 для Москвы)

    Возвращает dict с полями:
        temperature (°C), feels_like (°C), humidity (%), wind_speed (км/ч),
        description, weathercode
    """
    r = httpx.get(
        _FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "apparent_temperature,"
                "relative_humidity_2m,"
                "wind_speed_10m,"
                "weathercode"
            ),
            "timezone": "auto",
        },
        timeout=10,
    )
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "detail": r.text[:200]}

    w = r.json()["current"]
    return {
        "latitude": latitude,
        "longitude": longitude,
        "temperature": w["temperature_2m"],
        "feels_like": w["apparent_temperature"],
        "humidity": w["relative_humidity_2m"],
        "wind_speed": w["wind_speed_10m"],
        "weathercode": w["weathercode"],
        "description": _describe_wmo(w["weathercode"]),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
