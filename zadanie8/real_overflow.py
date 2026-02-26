"""
День 8 — Реальное переполнение контекста DeepSeek.

Берём реальный большой файл (~127k токенов) и отправляем его в DeepSeek.
Смотрим, что происходит на самом деле, когда превышается настоящий лимит в 64k токенов.
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError, APIError

# ── Клиент ──────────────────────────────────────────────────────────────────

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"
CONTEXT_LIMIT = 64_000  # реальный лимит DeepSeek

# ── Загружаем большой файл ───────────────────────────────────────────────────

BIG_FILE = os.path.join(
    os.path.dirname(__file__),
    "../venv/lib/python3.12/site-packages/telegram/_bot.py",
)

print("=" * 60)
print("РЕАЛЬНОЕ переполнение контекста DeepSeek")
print("=" * 60)

with open(BIG_FILE, encoding="utf-8") as f:
    big_text = f.read()

# Дублируем текст чтобы точно превысить 128k токенов
big_text = big_text * 3

chars = len(big_text)
est_tokens = chars // 4
CONTEXT_LIMIT = 128_000  # реальный лимит DeepSeek (не 64k!)

print(f"\nФайл: telegram/_bot.py x3 (утроен)")
print(f"Размер: {chars:,} символов")
print(f"Оценка токенов: ~{est_tokens:,}")
print(f"Лимит DeepSeek: {CONTEXT_LIMIT:,}")
print(f"Превышение в: {est_tokens / CONTEXT_LIMIT:.1f}x раза\n")

# ── Отправляем напрямую в DeepSeek ──────────────────────────────────────────

user_message = f"""Вот исходный код Python-библиотеки python-telegram-bot.
Проанализируй этот код и скажи, что он делает в двух словах.

```python
{big_text}
```
"""

msg_chars = len(user_message)
msg_est_tokens = msg_chars // 4

print(f"Сообщение: {msg_chars:,} символов (~{msg_est_tokens:,} токенов)")
print(f"Отправляем в DeepSeek API...\n")

try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Ты Python-эксперт."},
            {"role": "user", "content": user_message},
        ],
        max_tokens=100,  # нам нужен только ответ (или ошибка)
    )

    # Если дошли сюда — DeepSeek принял запрос (неожиданно!)
    answer = resp.choices[0].message.content
    usage = resp.usage
    print("✅ DeepSeek ответил (неожиданно!):")
    print(f"   prompt_tokens:     {usage.prompt_tokens:,}")
    print(f"   completion_tokens: {usage.completion_tokens:,}")
    print(f"   total_tokens:      {usage.total_tokens:,}")
    print(f"\nОтвет: {answer[:200]}")

except BadRequestError as e:
    print("💥 РЕАЛЬНАЯ ОШИБКА DeepSeek API:")
    print(f"   Тип:    BadRequestError")
    print(f"   Статус: {e.status_code}")
    print(f"   Тело:   {e.body}")
    print(f"\n→ Это и есть настоящее переполнение контекста!")
    print(f"→ DeepSeek отказал принять {msg_est_tokens:,} токенов")

except APIError as e:
    print("💥 API ошибка:")
    print(f"   {type(e).__name__}: {e}")

except Exception as e:
    print(f"💥 Неожиданная ошибка: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Вывод:")
print(f"  Наш ContextOverflowError = pre-flight check (chars/4 > limit)")
print(f"  Реальная ошибка DeepSeek = HTTP 400 от API сервера")
print(f"  Pre-flight дешевле: токены не тратятся до отправки")
print("=" * 60)
