# myai/core/llm.py
# Обёртка над LiteLLM — единый интерфейс для вызова любой LLM модели.
# Читает конфигурацию из myai/config.yaml и предоставляет async функцию chat().

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import litellm
import yaml


# Путь к конфигурационному файлу относительно корня проекта
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Кэш конфига — загружается один раз при первом обращении
_config_cache: dict[str, Any] | None = None


def load_config() -> dict[str, Any]:
    """Читает config.yaml и возвращает словарь с настройками.

    Returns:
        Словарь с конфигурацией проекта (llm, memory, telegram, api, mcp).

    Raises:
        FileNotFoundError: если config.yaml не найден.
        yaml.YAMLError: если файл содержит некорректный YAML.
    """
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {_CONFIG_PATH}\n"
            "Убедитесь что запускаете из корня проекта llm-api-project."
        )

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _config_cache = yaml.safe_load(f)

    return _config_cache


def get_current_model() -> str:
    """Возвращает название модели, заданной в config.yaml.

    Returns:
        Строка с названием модели, например 'claude-3-5-sonnet-20241022'.
    """
    config = load_config()
    return config["llm"]["model"]


def list_models() -> list[str]:
    """Возвращает список популярных моделей, поддерживаемых через LiteLLM.

    Returns:
        Список строк с идентификаторами моделей.
    """
    return [
        "claude-3-5-sonnet-20241022",
        "gpt-4o",
        "gpt-4o-mini",
        "gemini/gemini-1.5-pro",
        "ollama/llama3",
    ]


async def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    stream: bool = False,
) -> str:
    """Отправляет сообщения в LLM и возвращает ответ в виде строки.

    Использует litellm.acompletion, который поддерживает Claude, GPT,
    Gemini, Ollama и более 100 других моделей через единый API.

    Args:
        messages: Список словарей с ключами 'role' и 'content'.
                  Например: [{"role": "user", "content": "Привет!"}]
        model: Название модели LiteLLM. Если None — берётся из config.yaml.
        tools: Список инструментов в формате OpenAI function calling.
               Если None — инструменты не передаются.
        stream: Если True — потоковый режим (не реализован в текущей версии).

    Returns:
        Строка с текстом ответа от модели.

    Raises:
        litellm.AuthenticationError: если API ключ не задан или неверный.
        litellm.RateLimitError: если превышен лимит запросов.
        litellm.BadRequestError: если запрос содержит некорректные параметры.
        RuntimeError: при других непредвиденных ошибках LiteLLM.
    """
    config = load_config()
    llm_config = config.get("llm", {})

    # Выбираем модель: аргумент > config.yaml
    resolved_model = model if model is not None else llm_config.get("model", "gpt-4o-mini")

    max_tokens: int = llm_config.get("max_tokens", 4096)
    temperature: float = llm_config.get("temperature", 0.7)

    # Формируем kwargs для litellm
    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if tools:
        kwargs["tools"] = tools

    if stream:
        kwargs["stream"] = True

    try:
        response = await litellm.acompletion(**kwargs)
    except litellm.AuthenticationError as e:
        raise litellm.AuthenticationError(
            message=(
                f"Ошибка аутентификации для модели '{resolved_model}'.\n"
                f"Проверьте что API ключ задан в переменных окружения.\n"
                f"Оригинальная ошибка: {e}"
            ),
            llm_provider=resolved_model.split("/")[0] if "/" in resolved_model else resolved_model,
            model=resolved_model,
        ) from e
    except litellm.RateLimitError as e:
        raise litellm.RateLimitError(
            message=(
                f"Превышен лимит запросов для модели '{resolved_model}'.\n"
                f"Подождите немного и попробуйте снова.\n"
                f"Оригинальная ошибка: {e}"
            ),
            llm_provider=resolved_model.split("/")[0] if "/" in resolved_model else resolved_model,
            model=resolved_model,
        ) from e
    except litellm.BadRequestError as e:
        raise litellm.BadRequestError(
            message=(
                f"Некорректный запрос к модели '{resolved_model}'.\n"
                f"Проверьте формат сообщений и параметры.\n"
                f"Оригинальная ошибка: {e}"
            ),
            llm_provider=resolved_model.split("/")[0] if "/" in resolved_model else resolved_model,
            model=resolved_model,
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Неожиданная ошибка при вызове модели '{resolved_model}': {e}"
        ) from e

    # Извлекаем текст ответа из структуры litellm
    content: str = response.choices[0].message.content or ""
    return content
