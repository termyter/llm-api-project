# myai/core/tools.py
# Реестр инструментов для агента: Python-функции и MCP инструменты.
# Поддерживает регистрацию через функцию register() и декоратор @tool.
# Умеет подключаться к MCP серверу (zadanie16) и загружать его инструменты.
# Предоставляет форматирование для LiteLLM tool calling и диспетчеризацию вызовов.

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


# ── Dataclass инструмента ─────────────────────────────────────────────────────

@dataclass
class Tool:
    """Описание одного инструмента агента.

    Attributes:
        name: Уникальное имя инструмента (используется при вызове).
        description: Текстовое описание — LLM читает его и решает, когда применять.
        parameters: JSON Schema параметров вызова (OpenAI function calling формат).
        func: Python-функция реализующая инструмент. None для MCP инструментов.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any] | None = field(default=None, repr=False)


# ── Глобальный реестр ─────────────────────────────────────────────────────────

_tools: dict[str, Tool] = {}


# ── Регистрация инструментов ──────────────────────────────────────────────────

def register(
    name: str,
    func: Callable[..., Any],
    description: str,
    parameters: dict[str, Any] | None = None,
) -> None:
    """Регистрирует Python-функцию как инструмент агента.

    Args:
        name: Имя инструмента. Должно быть уникальным в реестре.
        func: Callable — синхронная или асинхронная функция.
        description: Описание что делает инструмент (читает LLM).
        parameters: JSON Schema параметров. Если None — пустая схема без параметров.

    Example:
        def add(a: int, b: int) -> int:
            return a + b

        register("add", add, "Складывает два числа", {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "Первое число"},
                "b": {"type": "integer", "description": "Второе число"},
            },
            "required": ["a", "b"],
        })
    """
    resolved_params: dict[str, Any] = parameters if parameters is not None else {
        "type": "object",
        "properties": {},
    }
    _tools[name] = Tool(name=name, description=description, parameters=resolved_params, func=func)


def tool(
    description: str,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Декоратор для удобной регистрации функции как инструмента агента.

    Args:
        description: Описание инструмента (читает LLM при выборе инструмента).
        parameters: JSON Schema параметров. Если None — пустая схема.

    Returns:
        Декоратор, который регистрирует функцию и возвращает её неизменной.

    Example:
        @tool(
            description="Возвращает текущее время",
            parameters={"type": "object", "properties": {}},
        )
        def get_time() -> str:
            from datetime import datetime
            return datetime.now().isoformat()
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        register(
            name=func.__name__,
            func=func,
            description=description,
            parameters=parameters,
        )
        return func

    return decorator


# ── Загрузка MCP инструментов ─────────────────────────────────────────────────

async def load_mcp_tools() -> list[Tool]:
    """Загружает инструменты с MCP сервера zadanie16 и добавляет их в реестр.

    Импортирует connect_and_list_tools из zadanie16.day16_mcp_client,
    запускает соединение с MCP сервером через stdio transport,
    преобразует полученные инструменты в объекты Tool и регистрирует их.

    MCP инструменты имеют func=None — вызываются через отдельный MCP сеанс.

    Returns:
        Список загруженных Tool объектов (только те, что пришли из MCP).

    Raises:
        ImportError: если модуль zadanie16.day16_mcp_client недоступен.
        Exception: если подключение к MCP серверу не удалось.
    """
    from zadanie16.day16_mcp_client import connect_and_list_tools  # type: ignore[import]

    data: dict[str, Any] = await connect_and_list_tools()
    raw_tools = data.get("tools", [])

    loaded: list[Tool] = []

    for mcp_tool in raw_tools:
        # mcp_tool — объект mcp.types.Tool с полями: name, description, inputSchema
        name: str = mcp_tool.name
        description: str = mcp_tool.description or ""
        input_schema: dict[str, Any] = mcp_tool.inputSchema or {
            "type": "object",
            "properties": {},
        }

        t = Tool(
            name=name,
            description=description,
            parameters=input_schema,
            func=None,  # MCP инструменты вызываются через MCP сессию, не напрямую
        )
        _tools[name] = t
        loaded.append(t)

    return loaded


# ── Получение зарегистрированных инструментов ─────────────────────────────────

def get_all() -> list[Tool]:
    """Возвращает все зарегистрированные инструменты.

    Returns:
        Список всех объектов Tool из глобального реестра.
    """
    return list(_tools.values())


def get_for_litellm() -> list[dict[str, Any]]:
    """Конвертирует зарегистрированные инструменты в формат LiteLLM tool calling.

    Формат соответствует спецификации OpenAI function calling и принимается
    LiteLLM при передаче параметра `tools` в acompletion().

    Returns:
        Список словарей в формате:
        [
            {
                "type": "function",
                "function": {
                    "name": "имя_инструмента",
                    "description": "описание",
                    "parameters": { ...JSON Schema... },
                },
            },
            ...
        ]
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in _tools.values()
    ]


# ── Вызов инструментов ────────────────────────────────────────────────────────

async def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Находит инструмент по имени и вызывает его с переданными аргументами.

    Поддерживает как синхронные функции, так и корутины (async def).
    Результат всегда возвращается как строка — LLM ожидает строку в tool result.

    Args:
        name: Имя инструмента (совпадает с Tool.name в реестре).
        arguments: Словарь аргументов для вызова (из tool_calls LLM ответа).

    Returns:
        Результат выполнения инструмента, приведённый к строке.
        Если результат — dict или list, сериализуется как JSON.

    Raises:
        KeyError: если инструмент с таким именем не найден в реестре.
        TypeError: если инструмент не имеет callable func (например, MCP инструмент).
        Exception: проброс исключений из самого инструмента.
    """
    if name not in _tools:
        raise KeyError(
            f"Инструмент '{name}' не найден в реестре. "
            f"Доступные: {list(_tools.keys())}"
        )

    t = _tools[name]

    if t.func is None:
        raise TypeError(
            f"Инструмент '{name}' является MCP инструментом и не имеет "
            f"локальной функции. Используйте MCP сессию для его вызова."
        )

    import asyncio

    if asyncio.iscoroutinefunction(t.func):
        result = await t.func(**arguments)
    else:
        result = t.func(**arguments)

    # Приводим результат к строке
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, indent=2)
    return str(result)
