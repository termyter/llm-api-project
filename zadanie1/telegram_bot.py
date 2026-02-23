#!/usr/bin/env python3
"""
Telegram бот для работы с LLM через API
Поддерживает режимы из всех заданий курса + настройки параметров.
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

# DeepSeek — для заданий 1, 2, 3, 4
deepseek = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com"
)

# RouterAI — для задания 5 и моделей Anthropic/Amazon
routerai = OpenAI(
    api_key=os.getenv("ROUTERAI_API_KEY"),
    base_url="https://routerai.ru/api/v1"
)

# Хранилище: pending_text[user_id] = текст ожидающий обработки
pending_text = {}

# Настройки пользователей: user_settings[user_id] = {...}
user_settings = {}

DEFAULT_SETTINGS = {
    "temperature": 0.7,
    "model": "deepseek-chat",
    "format": "free",  # free | points | short
}

MODEL_LABELS = {
    "deepseek-chat":           "DeepSeek V3",
    "amazon/nova-micro-v1":    "Nova Micro (слабая)",
    "anthropic/claude-opus-4.6": "Claude Opus (сильная)",
}

FORMAT_LABELS = {
    "free":   "Свободный",
    "points": "3 пункта",
    "short":  "Кратко (1 абзац)",
}

FORMAT_PROMPTS = {
    "free":   "Ты полезный ассистент. Отвечай по делу.",
    "points": "Отвечай строго в 3 пункта. Каждый пункт — одно предложение.",
    "short":  "Дай ответ в одном коротком абзаце, не более 3 предложений.",
}


def get_settings(user_id):
    return user_settings.get(user_id, dict(DEFAULT_SETTINGS))


def set_setting(user_id, key, value):
    if user_id not in user_settings:
        user_settings[user_id] = dict(DEFAULT_SETTINGS)
    user_settings[user_id][key] = value


# ─────────────────────────── КЛАВИАТУРЫ ───────────────────────────

def zadanie_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📝 Задание 1 — Обычный ответ", callback_data="z1")],
        [InlineKeyboardButton("📐 Задание 2 — Форматы (с/без ограничений)", callback_data="z2")],
        [InlineKeyboardButton("🧠 Задание 3 — Методы рассуждения", callback_data="z3")],
        [InlineKeyboardButton("🌡️ Задание 4 — Температура (0 / 0.7 / 1.2)", callback_data="z4")],
        [InlineKeyboardButton("🤖 Задание 5 — Сравнение моделей", callback_data="z5")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
    ])


def settings_keyboard(user_id):
    s = get_settings(user_id)
    temp = s["temperature"]
    model = s["model"]
    fmt = s["format"]

    return InlineKeyboardMarkup([
        [InlineKeyboardButton("─── Температура ───", callback_data="noop")],
        [
            InlineKeyboardButton(f"{'✅' if temp == 0 else '0'} Точный", callback_data="set_temp_0"),
            InlineKeyboardButton(f"{'✅' if temp == 0.7 else '0.7'} Баланс", callback_data="set_temp_07"),
            InlineKeyboardButton(f"{'✅' if temp == 1.2 else '1.2'} Креатив", callback_data="set_temp_12"),
        ],
        [InlineKeyboardButton("─── Модель ───", callback_data="noop")],
        [InlineKeyboardButton(
            f"{'✅ ' if model == 'deepseek-chat' else ''}DeepSeek V3",
            callback_data="set_model_deepseek"
        )],
        [InlineKeyboardButton(
            f"{'✅ ' if model == 'amazon/nova-micro-v1' else ''}Nova Micro (слабая)",
            callback_data="set_model_nova"
        )],
        [InlineKeyboardButton(
            f"{'✅ ' if model == 'anthropic/claude-opus-4.6' else ''}Claude Opus (сильная)",
            callback_data="set_model_claude"
        )],
        [InlineKeyboardButton("─── Формат ответа ───", callback_data="noop")],
        [
            InlineKeyboardButton(f"{'✅' if fmt == 'free' else '📄'} Свободный", callback_data="set_fmt_free"),
            InlineKeyboardButton(f"{'✅' if fmt == 'points' else '📋'} 3 пункта", callback_data="set_fmt_points"),
            InlineKeyboardButton(f"{'✅' if fmt == 'short' else '✂️'} Кратко", callback_data="set_fmt_short"),
        ],
        [InlineKeyboardButton("◀️ Назад", callback_data="back")],
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
        "/settings — настройки параметров\n"
        "/clean — очистить диалог"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❓ Как использовать:\n\n"
        "1. Напиши любой вопрос или текст\n"
        "2. Выбери задание из меню\n"
        "3. Смотри как разные настройки влияют на ответ\n\n"
        "📋 Задания:\n"
        "1 — Обычный ответ (использует твои настройки)\n"
        "2 — Сравнение форматов (с ограничениями и без)\n"
        "3 — 4 метода рассуждения (прямой, CoT, мета, эксперты)\n"
        "4 — Разные температуры (0, 0.7, 1.2)\n"
        "5 — Разные модели (слабая, средняя, сильная)\n\n"
        "⚙️ Настройки влияют только на Задание 1."
    )


async def clean_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pending_text.pop(update.effective_user.id, None)
    await update.message.reply_text("🧹 Диалог очищен! Можем начинать заново. 🍚")


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    s = get_settings(user_id)
    await update.message.reply_text(
        f"⚙️ Текущие настройки:\n"
        f"🌡️ Температура: {s['temperature']}\n"
        f"🤖 Модель: {MODEL_LABELS.get(s['model'], s['model'])}\n"
        f"📄 Формат: {FORMAT_LABELS.get(s['format'], s['format'])}\n\n"
        f"Выбери что изменить:",
        reply_markup=settings_keyboard(user_id)
    )


# ─────────────────────────── ВХОДЯЩИЕ СООБЩЕНИЯ ────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    pending_text[user_id] = user_text

    s = get_settings(user_id)
    await update.message.reply_text(
        f"📨 Твой запрос:\n«{user_text}»\n\n"
        f"⚙️ Настройки: temp={s['temperature']} | {MODEL_LABELS.get(s['model'])} | {FORMAT_LABELS.get(s['format'])}\n\n"
        f"Выбери задание:",
        reply_markup=zadanie_keyboard()
    )


# ─────────────────────────── ОБРАБОТКА КНОПОК ──────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    data = query.data

    # --- Настройки ---
    if data == "noop":
        return

    if data == "settings":
        s = get_settings(user_id)
        await query.edit_message_text(
            f"⚙️ Текущие настройки:\n"
            f"🌡️ Температура: {s['temperature']}\n"
            f"🤖 Модель: {MODEL_LABELS.get(s['model'], s['model'])}\n"
            f"📄 Формат: {FORMAT_LABELS.get(s['format'], s['format'])}\n\n"
            f"Выбери что изменить:",
            reply_markup=settings_keyboard(user_id)
        )
        return

    if data == "back":
        text = pending_text.get(user_id, "")
        s = get_settings(user_id)
        await query.edit_message_text(
            f"📨 Твой запрос:\n«{text}»\n\n"
            f"⚙️ Настройки: temp={s['temperature']} | {MODEL_LABELS.get(s['model'])} | {FORMAT_LABELS.get(s['format'])}\n\n"
            f"Выбери задание:" if text else "Напиши вопрос и выбери задание:",
            reply_markup=zadanie_keyboard()
        )
        return

    if data.startswith("set_"):
        _, param, val = data.split("_", 2)

        if param == "temp":
            temp_map = {"0": 0, "07": 0.7, "12": 1.2}
            set_setting(user_id, "temperature", temp_map[val])
        elif param == "model":
            model_map = {
                "deepseek": "deepseek-chat",
                "nova": "amazon/nova-micro-v1",
                "claude": "anthropic/claude-opus-4.6",
            }
            set_setting(user_id, "model", model_map[val])
        elif param == "fmt":
            set_setting(user_id, "format", val)

        s = get_settings(user_id)
        await query.edit_message_text(
            f"⚙️ Текущие настройки:\n"
            f"🌡️ Температура: {s['temperature']}\n"
            f"🤖 Модель: {MODEL_LABELS.get(s['model'], s['model'])}\n"
            f"📄 Формат: {FORMAT_LABELS.get(s['format'], s['format'])}\n\n"
            f"Выбери что изменить:",
            reply_markup=settings_keyboard(user_id)
        )
        return

    # --- Задания ---
    text = pending_text.get(user_id)
    if not text:
        await query.edit_message_text("❌ Сначала напиши вопрос, потом выбирай задание.")
        return

    await query.edit_message_text(f"⏳ Обрабатываю задание {data[-1]}...")

    try:
        s = get_settings(user_id)
        if data == "z1":
            result = await run_zadanie1(text, s)
        elif data == "z2":
            result = await run_zadanie2(text)
        elif data == "z3":
            result = await run_zadanie3(text)
        elif data == "z4":
            result = await run_zadanie4(text)
        elif data == "z5":
            result = await run_zadanie5(text)
        else:
            result = "Неизвестное задание"

        for chunk in split_text(result):
            await query.message.reply_text(chunk)

    except Exception as e:
        logger.error(f"Ошибка в задании {data}: {e}")
        await query.message.reply_text(f"❌ Ошибка: {str(e)}")


def split_text(text, max_len=4000):
    parts = []
    while len(text) > max_len:
        parts.append(text[:max_len])
        text = text[max_len:]
    parts.append(text)
    return parts


# ─────────────────────────── ЗАДАНИЕ 1 ─────────────────────────────

async def run_zadanie1(text, settings):
    model_id = settings["model"]
    temp = settings["temperature"]
    fmt = settings["format"]
    system_prompt = FORMAT_PROMPTS[fmt]

    # Выбираем клиент в зависимости от модели
    client = deepseek if model_id == "deepseek-chat" else routerai

    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=temp,
        max_tokens=500
    )
    answer = resp.choices[0].message.content
    tokens = resp.usage.total_tokens
    return (
        f"📝 ЗАДАНИЕ 1 — Ответ с настройками\n"
        f"🤖 Модель: {MODEL_LABELS.get(model_id, model_id)}\n"
        f"🌡️ Температура: {temp}\n"
        f"📄 Формат: {FORMAT_LABELS[fmt]}\n"
        "─" * 30 + "\n"
        f"{answer}\n\n"
        f"📊 Токенов: {tokens}"
    )


# ─────────────────────────── ЗАДАНИЕ 2 ─────────────────────────────

async def run_zadanie2(text):
    out = ["📐 ЗАДАНИЕ 2 — Сравнение форматов\n" + "─" * 30]

    r1 = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": text}]
    )
    out.append(
        "1️⃣ БЕЗ ограничений:\n"
        f"{r1.choices[0].message.content}\n"
        f"📊 Токенов: {r1.usage.total_tokens}"
    )

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

    r1 = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": text}]
    )
    out.append(f"1️⃣ Прямой ответ:\n{r1.choices[0].message.content}")

    r2 = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": f"Думай пошагово, объясни каждый шаг:\n{text}"}]
    )
    out.append(f"2️⃣ Chain of Thought:\n{r2.choices[0].message.content}")

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

    answers = []
    for model_id, label in models:
        t0 = time.time()
        r = routerai.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": text}]
        )
        elapsed = time.time() - t0
        answer = r.choices[0].message.content
        answers.append((model_id, label, answer))
        out.append(
            f"🔹 {label}:\n"
            f"{answer}\n"
            f"⏱ {elapsed:.1f}с | 📊 {r.usage.total_tokens} токенов"
        )

    # Финальный вывод от Claude Opus
    analysis = routerai.chat.completions.create(
        model="anthropic/claude-opus-4.6",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты — эксперт по оценке LLM. Тебе дали три ответа на один вопрос "
                    "от слабой, средней и сильной модели. "
                    "Напиши короткий вывод (3-5 предложений): "
                    "чем отличаются ответы, какая модель справилась лучше и почему, "
                    "когда стоит использовать каждую."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {text}\n\n"
                    f"--- {answers[0][1]} ---\n{answers[0][2]}\n\n"
                    f"--- {answers[1][1]} ---\n{answers[1][2]}\n\n"
                    f"--- {answers[2][1]} ---\n{answers[2][2]}"
                )
            }
        ]
    )
    out.append(
        "🏆 ВЫВОД от Claude Opus:\n"
        "─" * 30 + "\n"
        f"{analysis.choices[0].message.content}"
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
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print("🤖 Бот запущен!")
    print("💬 Отвечу на твой вопрос за миска риса!")
    print("Нажмите Ctrl+C для остановки.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
