# myai/core/agent.py
# Агентный цикл: объединяет llm.py, memory.py и tools.py в единый run() интерфейс.
# Поддерживает стандартный OpenAI-совместимый tool calling loop (до 5 итераций).
# Предоставляет run() с инструментами и run_simple() без инструментов.

from __future__ import annotations

import json
import string
import random
from datetime import datetime
from typing import Any

import litellm

from myai.core import llm as llm_module
from myai.core import memory
from myai.core import tools


# ── Системный промпт по умолчанию ─────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT: str = (
    "Ты персональный AI ассистент. Отвечаешь на русском если пользователь пишет на русском.\n"
    "У тебя есть инструменты — используй их когда нужно.\n"
    "Ты помнишь историю разговора."
)

# Максимальное количество итераций в agent loop (защита от бесконечного цикла)
_MAX_ITERATIONS: int = 5


# ── Вспомогательная функция: вызов LiteLLM с полным ответом ──────────────────

async def _call_llm_raw(
    messages: list[dict[str, Any]],
    model: str | None,
    tools_list: list[dict[str, Any]] | None,
) -> Any:
    """Вызывает litellm.acompletion и возвращает полный объект ответа.

    В отличие от llm.chat() — возвращает не только строку, но весь объект
    ModelResponse, включая возможные tool_calls.

    Args:
        messages: Список сообщений в формате OpenAI (role/content).
        model: Название модели или None (берётся из config.yaml).
        tools_list: Список инструментов в формате LiteLLM или None.

    Returns:
        Объект ModelResponse от LiteLLM с полями choices, usage и т.д.

    Raises:
        RuntimeError: при ошибке запроса к LLM.
    """
    config = llm_module.load_config()
    llm_config = config.get("llm", {})

    resolved_model: str = model if model is not None else llm_config.get("model", "gpt-4o-mini")
    max_tokens: int = llm_config.get("max_tokens", 4096)
    temperature: float = llm_config.get("temperature", 0.7)

    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if tools_list:
        kwargs["tools"] = tools_list

    try:
        response = await litellm.acompletion(**kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Ошибка при вызове модели '{resolved_model}': {e}"
        ) from e

    return response


# ── Основная функция агента с инструментами ───────────────────────────────────

async def run(
    user_message: str,
    session_id: str = "default",
    model: str | None = None,
) -> str:
    """Запускает агентный цикл с поддержкой инструментов (tool calling).

    Загружает историю диалога, добавляет системный промпт, выполняет
    стандартный agent loop: LLM -> tool_calls -> результаты -> LLM -> ...
    до получения финального текстового ответа (или 5 итераций).

    Args:
        user_message: Текст сообщения от пользователя.
        session_id: Идентификатор сессии для хранения истории.
                    По умолчанию "default".
        model: Название модели LiteLLM. Если None — из config.yaml.

    Returns:
        Строка с финальным ответом ассистента.

    Raises:
        RuntimeError: если LLM не вернул ответ за 5 итераций.
    """
    # Инициализируем БД если нужно
    await memory.init_db()

    # Загружаем историю и строим список messages
    history: list[dict[str, Any]] = await memory.get_history(session_id)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    # Сохраняем сообщение пользователя в историю
    await memory.save_message("user", user_message, session_id)

    # Получаем инструменты в формате LiteLLM
    tools_list: list[dict[str, Any]] = tools.get_for_litellm()

    # ── Agent loop ────────────────────────────────────────────────────────────
    final_content: str = ""

    for iteration in range(_MAX_ITERATIONS):
        response = await _call_llm_raw(
            messages=messages,
            model=model,
            tools_list=tools_list if tools_list else None,
        )

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        # Если нет вызовов инструментов — получили финальный ответ
        if not tool_calls:
            final_content = message.content or ""
            break

        # Добавляем ответ ассистента (с tool_calls) в messages
        messages.append(message.model_dump() if hasattr(message, "model_dump") else dict(message))

        # Выполняем каждый tool call
        for tc in tool_calls:
            tool_name: str = tc.function.name
            tool_args_raw: str = tc.function.arguments or "{}"

            try:
                tool_args: dict[str, Any] = json.loads(tool_args_raw)
            except json.JSONDecodeError:
                tool_args = {}

            try:
                tool_result: str = await tools.call_tool(tool_name, tool_args)
            except Exception as e:
                tool_result = f"Ошибка при вызове инструмента '{tool_name}': {e}"

            # Добавляем результат инструмента в messages
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })
    else:
        # Исчерпали итерации — берём последний доступный контент
        last_message = response.choices[0].message  # type: ignore[possibly-undefined]
        final_content = getattr(last_message, "content", None) or (
            "Превышено максимальное количество итераций агентного цикла."
        )

    # Сохраняем финальный ответ ассистента в историю
    await memory.save_message("assistant", final_content, session_id)

    return final_content


# ── Упрощённая функция без инструментов ──────────────────────────────────────

async def run_simple(
    user_message: str,
    session_id: str = "default",
    model: str | None = None,
) -> str:
    """Запускает упрощённый агентный цикл без инструментов (tool calling).

    Аналогичен run(), но не передаёт инструменты в LLM. Подходит для
    простых вопросов, перефразирования, генерации текста — там где
    tool calling не нужен и только замедляет ответ.

    Args:
        user_message: Текст сообщения от пользователя.
        session_id: Идентификатор сессии для хранения истории.
                    По умолчанию "default".
        model: Название модели LiteLLM. Если None — из config.yaml.

    Returns:
        Строка с ответом ассистента.
    """
    # Инициализируем БД если нужно
    await memory.init_db()

    # Загружаем историю и строим список messages
    history: list[dict[str, Any]] = await memory.get_history(session_id)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    # Сохраняем сообщение пользователя в историю
    await memory.save_message("user", user_message, session_id)

    # Вызываем LLM без инструментов (один раз, без цикла)
    response = await _call_llm_raw(
        messages=messages,
        model=model,
        tools_list=None,
    )

    final_content: str = response.choices[0].message.content or ""

    # Сохраняем ответ ассистента в историю
    await memory.save_message("assistant", final_content, session_id)

    return final_content


# ── Генерация нового session_id ───────────────────────────────────────────────

async def new_session() -> str:
    """Генерирует уникальный идентификатор новой сессии.

    Формат: YYYYMMDD_HHMMSS_<6 случайных символов>
    Например: "20260309_142530_xk7pqr"

    Returns:
        Строка с уникальным идентификатором сессии.
    """
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix: str = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=6)
    )
    return f"{timestamp}_{suffix}"
