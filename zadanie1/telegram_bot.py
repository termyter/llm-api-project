#!/usr/bin/env python3
"""
Telegram бот для работы с LLM через API
Поддерживает режимы из всех заданий курса.
"""

import os
import time
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# DeepSeek — для заданий 2, 3, 4
deepseek = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com"
)

# RouterAI — для задания 5 (сравнение моделей)
routerai = OpenAI(
    api_key=os.getenv("ROUTERAI_API_KEY"),
    base_url="https://routerai.ru/api/v1"
)

# Хранилище: pending_text[user_id] = текст ожидающий обработки
pending_text = {}


# ─────────────────────────── КЛАВИАТУРА ───────────────────────────

def zadanie_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📝 Задание 1 — Обычный ответ", callback_data="z1")],
        [InlineKeyboardButton("📐 Задание 2 — Форматы (с/без ограничений)", callback_data="z2")],
        [InlineKeyboardButton("🧠 Задание 3 — Методы рассуждения", callback_data="z3")],
        [InlineKeyboardButton("🌡️ Задание 4 — Температура (0 / 0.7 / 1.2)", callback_data="z4")],
        [InlineKeyboardButton("🤖 Задание 5 — Сравнение моделей", callback_data="z5")],
    ])


# ─────────────────────────── КОМАНДЫ ───────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот-лаборатория LLM.\n\n"
        "🍚 Отвечу на твой вопрос за миска риса!\n\n"
        "Напиши любой вопрос — и я покажу меню заданий.\n\n"
        "Команды:\n"
        "/start — это сообщение\n"
        "/help — помощь\n"
        "/clean — очистить диалог"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❓ Как использовать:\n\n"
        "1. Напиши любой вопрос или текст\n"
        "2. Выбери задание из меню\n"
        "3. Смотри как разные настройки влияют на ответ\n\n"
        "📋 Задания:\n"
        "1 — Обычный ответ (DeepSeek)\n"
        "2 — Сравнение форматов (с ограничениями и без)\n"
        "3 — 4 метода рассуждения (прямой, CoT, мета, эксперты)\n"
        "4 — Разные температуры (0, 0.7, 1.2)\n"
        "5 — Разные модели (слабая, средняя, сильная)"
    )


async def clean_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pending_text.pop(update.effective_user.id, None)
    await update.message.reply_text("🧹 Диалог очищен! Можем начинать заново. 🍚")


# ─────────────────────────── ВХОДЯЩИЕ СООБЩЕНИЯ ────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    pending_text[user_id] = user_text

    await update.message.reply_text(
        f"📨 Твой запрос:\n«{user_text}»\n\nВыбери задание:",
        reply_markup=zadanie_keyboard()
    )


# ─────────────────────────── ОБРАБОТКА КНОПОК ──────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    zadanie = query.data
    text = pending_text.get(user_id)

    if not text:
        await query.edit_message_text("❌ Сначала напиши вопрос, потом выбирай задание.")
        return

    await query.edit_message_text(f"⏳ Обрабатываю по заданию {zadanie[-1]}...")

    try:
        if zadanie == "z1":
            result = await run_zadanie1(text)
        elif zadanie == "z2":
            result = await run_zadanie2(text)
        elif zadanie == "z3":
            result = await run_zadanie3(text)
        elif zadanie == "z4":
            result = await run_zadanie4(text)
        elif zadanie == "z5":
            result = await run_zadanie5(text)
        else:
            result = "Неизвестное задание"

        # Telegram ограничивает сообщения 4096 символами — режем если надо
        for chunk in split_text(result):
            await query.message.reply_text(chunk)

    except Exception as e:
        logger.error(f"Ошибка в задании {zadanie}: {e}")
        await query.message.reply_text(f"❌ Ошибка: {str(e)}")


def split_text(text, max_len=4000):
    """Разбивает длинный текст на части."""
    parts = []
    while len(text) > max_len:
        parts.append(text[:max_len])
        text = text[max_len:]
    parts.append(text)
    return parts


# ─────────────────────────── ЗАДАНИЕ 1 ─────────────────────────────

async def run_zadanie1(text):
    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Ты полезный ассистент. Отвечай кратко и по делу."},
            {"role": "user", "content": text}
        ],
        max_tokens=500
    )
    answer = resp.choices[0].message.content
    tokens = resp.usage.total_tokens
    return (
        "📝 ЗАДАНИЕ 1 — Обычный ответ (DeepSeek)\n"
        "─" * 30 + "\n"
        f"{answer}\n\n"
        f"📊 Токенов: {tokens}"
    )


# ─────────────────────────── ЗАДАНИЕ 2 ─────────────────────────────

async def run_zadanie2(text):
    out = ["📐 ЗАДАНИЕ 2 — Сравнение форматов\n" + "─" * 30]

    # Запрос 1: без ограничений
    r1 = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": text}]
    )
    out.append(
        "1️⃣ БЕЗ ограничений:\n"
        f"{r1.choices[0].message.content}\n"
        f"📊 Токенов: {r1.usage.total_tokens}"
    )

    # Запрос 2: с ограничениями
    r2 = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Отвечай строго в 3 пункта. Каждый пункт — одно предложение."},
            {"role": "user", "content": text}
        ],
        max_tokens=150,
        stop=["4."]
    )
    out.append(
        "2️⃣ С ограничениями (3 пункта, max 150 токенов):\n"
        f"{r2.choices[0].message.content}\n"
        f"📊 Токенов: {r2.usage.total_tokens}"
    )

    return "\n\n".join(out)


# ─────────────────────────── ЗАДАНИЕ 3 ─────────────────────────────

async def run_zadanie3(text):
    out = ["🧠 ЗАДАНИЕ 3 — Методы рассуждения\n" + "─" * 30]

    # Метод 1: прямой ответ
    r1 = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": text}]
    )
    out.append(f"1️⃣ Прямой ответ:\n{r1.choices[0].message.content}")

    # Метод 2: Chain of Thought
    r2 = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": f"Думай пошагово, объясни каждый шаг:\n{text}"}]
    )
    out.append(f"2️⃣ Chain of Thought:\n{r2.choices[0].message.content}")

    # Метод 3: мета-промпт
    meta = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": f"Напиши идеальный промпт для ответа на: «{text}»"}]
    )
    best_prompt = meta.choices[0].message.content
    r3 = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": best_prompt}]
    )
    out.append(f"3️⃣ Мета-промпт:\n[промпт]: {best_prompt[:200]}...\n[ответ]: {r3.choices[0].message.content}")

    # Метод 4: группа экспертов
    r4 = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": (
                f"Ответь на вопрос от лица трёх экспертов:\n"
                f"1. Эксперт-теоретик\n2. Практик\n3. Критик\n\n{text}"
            )
        }]
    )
    out.append(f"4️⃣ Группа экспертов:\n{r4.choices[0].message.content}")

    return "\n\n".join(out)


# ─────────────────────────── ЗАДАНИЕ 4 ─────────────────────────────

async def run_zadanie4(text):
    out = ["🌡️ ЗАДАНИЕ 4 — Сравнение температур\n" + "─" * 30]

    for temp in [0, 0.7, 1.2]:
        r = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": text}],
            temperature=temp
        )
        label = {0: "Детерминированный", 0.7: "Баланс", 1.2: "Креативный"}[temp]
        out.append(
            f"🌡️ temp={temp} ({label}):\n"
            f"{r.choices[0].message.content}\n"
            f"📊 Токенов: {r.usage.total_tokens}"
        )

    return "\n\n".join(out)


# ─────────────────────────── ЗАДАНИЕ 5 ─────────────────────────────

async def run_zadanie5(text):
    out = ["🤖 ЗАДАНИЕ 5 — Сравнение моделей\n" + "─" * 30]

    models = [
        ("amazon/nova-micro-v1",      "Слабая  (Nova Micro)"),
        ("deepseek/deepseek-chat",    "Средняя (DeepSeek V3)"),
        ("anthropic/claude-opus-4.6", "Сильная (Claude Opus 4.6)"),
    ]

    for model_id, label in models:
        start = time.time()
        r = routerai.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": text}]
        )
        elapsed = time.time() - start
        out.append(
            f"🔹 {label}:\n"
            f"{r.choices[0].message.content}\n"
            f"⏱ {elapsed:.1f}с | 📊 {r.usage.total_tokens} токенов"
        )

    return "\n\n".join(out)


# ─────────────────────────── ЗАПУСК ────────────────────────────────

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")


def main():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN не найден в .env!")
        return

    app = Application.builder().token(bot_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clean", clean_command))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print("🤖 Бот запущен!")
    print("💬 Отвечу на твой вопрос за миска риса!")
    print("Нажмите Ctrl+C для остановки.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
