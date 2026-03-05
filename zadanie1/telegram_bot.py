#!/usr/bin/env python3
"""
Telegram бот для работы с LLM через API
Поддерживает режимы из всех заданий курса + настройки параметров.
"""

import asyncio
import os
import sys
import time
import logging
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from openai import OpenAI, BadRequestError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zadanie6.agent import Agent
from zadanie8.day8_tokens import TokenAwareAgent, ContextOverflowError
from zadanie9.day9_compression import SummaryAgent
from zadanie10.day10_strategies import (
    SlidingWindowAgent, StickyFactsAgent, BranchingAgent,
    DEMO_SHORT, BRANCH_DEMO_BASE, BRANCH_DEMO_FORK_A, BRANCH_DEMO_FORK_B,
    SYSTEM_PROMPT as Z10_SYSTEM_PROMPT,
    BRANCH_A, BRANCH_B, BRANCH_MAIN,
)
from zadanie11.day11_memory import (
    MemoryAgent,
    DEMO_INTRO, DEMO_RECALL,
)
from zadanie12.day12_profile import (
    PersonalizedAgent, UserProfile, ProfileManager,
    PRESET_PROFILES, DEMO_QUESTION, DEMO_INTRO_MESSAGES,
)
from zadanie13.day13_statemachine import (
    TaskStateMachine, TaskContext, DEMO_TASK, TaskState,
)

load_dotenv()

# ID владельца бота — только он может перезапускать
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))


def is_admin(user_id: int) -> bool:
    return ADMIN_ID != 0 and user_id == ADMIN_ID


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

# Агенты: agent_sessions[user_id] = Agent(...)
agent_sessions = {}

# Режим агента: user_id в этом множестве → все сообщения идут в агент
agent_mode = set()

# Задание 8: агенты с подсчётом токенов
token_sessions: dict[int, TokenAwareAgent] = {}
token_mode: set[int] = set()

# Задание 9: агенты со сжатием
summary_sessions: dict[int, SummaryAgent] = {}
summary_mode: set[int] = set()

# Задание 10: агенты со стратегиями управления контекстом
# z10_sessions[user_id] = {"strategy": str, "agent": agent_obj}
z10_sessions: dict[int, dict] = {}
z10_mode: set[int] = set()

# Задание 11: агенты с 3-слойной памятью
# z11_sessions[user_id] = MemoryAgent
z11_sessions: dict[int, MemoryAgent] = {}
z11_mode: set[int] = set()

# Задание 12: персонализированный агент
# z12_sessions[user_id] = PersonalizedAgent
z12_sessions: dict[int, PersonalizedAgent] = {}
z12_mode: set[int] = set()
z12_profile_manager = ProfileManager()

# Задание 13: Task State Machine
# z13_sessions[user_id] = TaskContext  (текущая задача)
z13_sessions: dict[int, TaskContext] = {}
z13_mode: set[int] = set()   # ожидаем ввод задачи от пользователя
_z13_fsm: TaskStateMachine | None = None  # ленивая инициализация


def get_z13_fsm() -> TaskStateMachine:
    global _z13_fsm
    if _z13_fsm is None:
        _z13_fsm = TaskStateMachine(client=deepseek)
    return _z13_fsm

DEFAULT_SETTINGS = {
    "temperature": 0.7,
    "model": "deepseek-chat",
    "format": "free",  # free | points | short
    "api": "deepseek",  # deepseek | routerai
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

API_LABELS = {
    "deepseek": "DeepSeek API (прямой)",
    "routerai": "RouterAI",
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
        [InlineKeyboardButton("💬 Задание 6 — Агент (чат с историей)", callback_data="z6")],
        [InlineKeyboardButton("💾 Задание 7 — Агент с памятью", callback_data="z7")],
        [InlineKeyboardButton("🔢 Задание 8 — Токены (счётчик в реальном времени)", callback_data="z8")],
        [InlineKeyboardButton("🗜 Задание 9 — Сжатие истории (summary)", callback_data="z9")],
        [InlineKeyboardButton("🌿 Задание 10 — Стратегии контекста", callback_data="z10")],
        [InlineKeyboardButton("🧠 Задание 11 — Память ассистента (3 слоя)", callback_data="z11")],
        [InlineKeyboardButton("🎭 Задание 12 — Персонализация профилей", callback_data="z12")],
        [InlineKeyboardButton("🔄 Задание 13 — Task State Machine (FSM)", callback_data="z13")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
    ])


def summary_agent_keyboard():
    """Кнопки для режима агента со сжатием (задание 9)."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🗑 Сбросить чат", callback_data="t9_reset")],
        [InlineKeyboardButton("📊 Статистика сессии", callback_data="t9_stats")],
        [InlineKeyboardButton("📝 Показать summary", callback_data="t9_summary")],
        [InlineKeyboardButton("🚪 Выйти из режима", callback_data="t9_exit")],
    ])


def z11_chat_keyboard() -> InlineKeyboardMarkup:
    """Кнопки управления в режиме агента с памятью (задание 11)."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Показать всю память", callback_data="t11_show_memory"),
         InlineKeyboardButton("📊 Статистика", callback_data="t11_stats")],
        [InlineKeyboardButton("🗑 Сбросить сессию (STM+WM)", callback_data="t11_reset_session"),
         InlineKeyboardButton("🧹 Забыть всё (LTM)", callback_data="t11_forget_all")],
        [InlineKeyboardButton("🎬 Авто-демо", callback_data="t11_demo")],
        [InlineKeyboardButton("🚪 Выйти", callback_data="t11_exit")],
    ])


def z10_menu_keyboard():
    """Меню выбора стратегии для задания 10."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🪟 Скользящее окно", callback_data="z10_sw")],
        [InlineKeyboardButton("📌 Закреплённые факты", callback_data="z10_facts")],
        [InlineKeyboardButton("🌿 Ветвление", callback_data="z10_branch")],
        [InlineKeyboardButton("🎬 Авто-демо всех стратегий", callback_data="z10_demo")],
    ])


def z10_chat_keyboard(strategy: str, has_branch: bool = False) -> InlineKeyboardMarkup:
    """Кнопки управления во время чата с заданием 10."""
    rows = [
        [InlineKeyboardButton("🗑 Сбросить", callback_data="t10_reset"),
         InlineKeyboardButton("📊 Статистика", callback_data="t10_stats")],
        [InlineKeyboardButton("🔀 Сменить стратегию", callback_data="t10_change")],
    ]
    if strategy == "sticky_facts":
        rows.insert(1, [InlineKeyboardButton("📌 Показать факты", callback_data="t10_show_facts")])
    if strategy == "branching":
        rows.insert(1, [
            InlineKeyboardButton("📍 Создать checkpoint", callback_data="t10_checkpoint"),
        ])
        if has_branch:
            rows.insert(2, [
                InlineKeyboardButton(f"🌿 {BRANCH_A}", callback_data="t10_fork_a"),
                InlineKeyboardButton(f"🌱 {BRANCH_B}", callback_data="t10_fork_b"),
            ])
    rows.append([InlineKeyboardButton("🚪 Выйти", callback_data="t10_exit")])
    return InlineKeyboardMarkup(rows)


def token_agent_keyboard(show_overflow: bool = False):
    """Кнопки для режима агента с подсчётом токенов (задание 8)."""
    rows = [
        [InlineKeyboardButton("🗑 Сбросить чат", callback_data="t8_reset")],
        [InlineKeyboardButton("📊 Статистика сессии", callback_data="t8_stats")],
        [InlineKeyboardButton("🔥 Реальное переполнение API (131k токенов)", callback_data="t8_overflow")],
        [InlineKeyboardButton("🚪 Выйти из режима", callback_data="t8_exit")],
    ]
    return InlineKeyboardMarkup(rows)


def agent_keyboard():
    """Кнопка для выхода из режима агента."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🗑 Новый чат", callback_data="agent_reset")],
        [InlineKeyboardButton("🔄 Сменить модель", callback_data="agent_restart_model")],
        [InlineKeyboardButton("♻️ Перезапустить агент", callback_data="agent_simulate_restart")],
        [InlineKeyboardButton("🚪 Выйти из агента", callback_data="agent_exit")],
    ])


def settings_keyboard(user_id):
    s = get_settings(user_id)
    temp = s["temperature"]
    model = s["model"]
    fmt = s["format"]
    api = s.get("api", "deepseek")

    rows = [
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
        [InlineKeyboardButton("─── API источник ───", callback_data="noop")],
        [
            InlineKeyboardButton(
                f"{'✅ ' if api == 'deepseek' else ''}🔑 DeepSeek API",
                callback_data="set_api_deepseek"
            ),
            InlineKeyboardButton(
                f"{'✅ ' if api == 'routerai' else ''}🌐 RouterAI",
                callback_data="set_api_routerai"
            ),
        ],
        [InlineKeyboardButton("◀️ Назад", callback_data="back")],
    ]
    if is_admin(user_id):
        rows.insert(-1, [InlineKeyboardButton("🔄 Перезапустить бота", callback_data="admin_restart")])
    return InlineKeyboardMarkup(rows)


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
        "5 — Разные модели (слабая, средняя, сильная)\n"
        "6 — Агент с памятью (полноценный чат с историей)\n\n"
        "⚙️ Настройки влияют на Задания 1 и 6.\n"
        "/newchat — сбросить историю агента"
    )


async def clean_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pending_text.pop(update.effective_user.id, None)
    await update.message.reply_text("🧹 Диалог очищен! Можем начинать заново. 🍚")


async def newchat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in agent_sessions:
        agent_sessions[user_id].reset()
        await update.message.reply_text(
            "🗑 История агента сброшена. Начинаем новый диалог!\n\n"
            "Напиши сообщение — и я отвечу.", reply_markup=agent_keyboard()
        )
    else:
        await update.message.reply_text("Ты ещё не в режиме агента. Выбери 💬 Задание 6.")


async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Нет доступа.")
        return
    await update.message.reply_text("🔄 Перезапускаюсь...")
    await asyncio.sleep(1)
    sys.exit(0)


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    s = get_settings(user_id)
    await update.message.reply_text(
        f"⚙️ Текущие настройки:\n"
        f"🌡️ Температура: {s['temperature']}\n"
        f"🤖 Модель: {MODEL_LABELS.get(s['model'], s['model'])}\n"
        f"📄 Формат: {FORMAT_LABELS.get(s['format'], s['format'])}\n"
        f"🔌 API: {API_LABELS.get(s.get('api', 'deepseek'))}\n\n"
        f"Выбери что изменить:",
        reply_markup=settings_keyboard(user_id)
    )


# ─────────────────────────── ВХОДЯЩИЕ СООБЩЕНИЯ ────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text

    # Задание 13: Task State Machine — ожидаем задачу для FSM
    if user_id in z13_mode:
        z13_mode.discard(user_id)
        await update.message.reply_text("⏳ Принято! Запускаю автомат задачи...")
        await start_z13_task(user_id, user_text, update.message.reply_text)
        return

    # Задание 12: персонализированный агент
    if user_id in z12_mode:
        agent = z12_sessions.get(user_id)
        if not agent:
            await update.message.reply_text("❌ Сессия не найдена. Выбери 🎭 Задание 12 заново.")
            return
        try:
            typing = await update.message.reply_text("⏳ Думаю...")
            reply, stat = await asyncio.to_thread(agent.chat, user_text)
            await typing.delete()
            for chunk in split_text(reply):
                await update.message.reply_text(chunk)
            stat_text = (
                f"📊 ХОД {stat.turn} │ Профиль: {agent.profile.style}/{agent.profile.expertise}\n"
                f"{'─' * 28}\n"
                f"Токены вход/выход : {stat.prompt_tokens:,} / {stat.completion_tokens:,}\n"
                f"Стоимость         : {stat.cost_rub:.4f} ₽\n"
                f"LTM фактов        : {agent.ltm.count()}"
            )
            await update.message.reply_text(stat_text, reply_markup=z12_chat_keyboard())
        except Exception as e:
            logger.error(f"Ошибка z12-агента: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")
        return

    # Задание 11: режим агента с 3-слойной памятью
    if user_id in z11_mode:
        agent = z11_sessions.get(user_id)
        if not agent:
            await update.message.reply_text("❌ Сессия не найдена. Выбери 🧠 Задание 11 заново.")
            return
        try:
            typing = await update.message.reply_text("⏳ Думаю...")
            reply, stat = await asyncio.to_thread(agent.chat, user_text)
            await typing.delete()
            for chunk in split_text(reply):
                await update.message.reply_text(chunk)
            stat_text = (
                f"📊 ХОД {stat.turn}\n"
                f"{'─' * 28}\n"
                f"Токены вход/выход : {stat.prompt_tokens:,} / {stat.completion_tokens:,}\n"
                f"Стоимость         : {stat.cost_rub:.4f} ₽\n"
                f"LTM загружено     : {stat.ltm_facts_loaded} фактов\n"
                f"LTM сохранено     : {stat.ltm_facts_saved} новых фактов\n"
                f"Всего в LTM       : {agent.ltm.count()} фактов"
            )
            await update.message.reply_text(stat_text, reply_markup=z11_chat_keyboard())
        except Exception as e:
            logger.error(f"Ошибка z11-агента: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")
        return

    # Задание 10: режим агента со стратегиями контекста
    if user_id in z10_mode:
        session = z10_sessions.get(user_id)
        if not session:
            await update.message.reply_text("❌ Сессия не найдена. Выбери 🌿 Задание 10 заново.")
            return
        agent = session["agent"]
        strategy = session["strategy"]
        try:
            typing = await update.message.reply_text("⏳ Думаю...")
            reply, stat = await asyncio.to_thread(agent.chat, user_text)
            await typing.delete()
            await update.message.reply_text(f"👤 Пользователь:\n{user_text}")
            for chunk in split_text(reply):
                await update.message.reply_text(f"🤖 Агент:\n{chunk}")
            has_branch = strategy == "branching" and agent.checkpoint is not None
            stat_text = _z10_stat_text(stat, agent)
            await update.message.reply_text(stat_text, reply_markup=z10_chat_keyboard(strategy, has_branch))
        except Exception as e:
            logger.error(f"Ошибка z10-агента: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")
        return

    # Задание 9: режим агента со сжатием истории
    if user_id in summary_mode:
        agent = summary_sessions.get(user_id)
        if not agent:
            await update.message.reply_text("❌ Сессия не найдена. Выбери 🗜 Задание 9 заново.")
            return
        try:
            typing = await update.message.reply_text("⏳ Думаю...")
            answer, stat = agent.chat(user_text)
            await typing.delete()
            for chunk in split_text(answer):
                await update.message.reply_text(chunk)

            compress_note = ""
            if stat.compressed:
                compress_note = (
                    f"\n🗜 Сжатие! Потрачено {stat.summary_tokens:,} токенов на summary.\n"
                    f"📝 История сжата до {agent.keep_recent} последних сообщений."
                )

            stat_text = (
                f"📊 Ход {stat.turn}{compress_note}\n"
                f"{'─' * 28}\n"
                f"Вход (контекст)  : {stat.prompt_tokens:>7,} т\n"
                f"Выход (ответ)    : {stat.completion_tokens:>7,} т\n"
                f"На сжатие        : {stat.summary_tokens:>7,} т\n"
                f"Итого за ход     : {stat.total_tokens:>7,} т\n"
                f"{'─' * 28}\n"
                f"В рабочей памяти : {agent.context_size:>3} сообщ.\n"
                f"Summary есть     : {'да ✅' if agent.summary else 'нет —'}\n"
                f"Стоимость хода   : {stat.cost_rub:.5f} ₽"
            )
            await update.message.reply_text(stat_text, reply_markup=summary_agent_keyboard())
        except Exception as e:
            logger.error(f"Ошибка summary-агента: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")
        return

    # Задание 8: режим агента с подсчётом токенов
    if user_id in token_mode:
        agent = token_sessions.get(user_id)
        if not agent:
            await update.message.reply_text("❌ Сессия не найдена. Выбери 🔢 Задание 8 заново.")
            return
        try:
            typing = await update.message.reply_text("⏳ Думаю...")
            answer, stat = agent.chat(user_text)
            await typing.delete()
            for chunk in split_text(answer):
                await update.message.reply_text(chunk)
            await update.message.reply_text(
                format_token_stat(stat, agent),
                reply_markup=token_agent_keyboard()
            )
        except ContextOverflowError as e:
            await update.message.reply_text(
                f"💥 ПЕРЕПОЛНЕНИЕ КОНТЕКСТА!\n\n{e}\n\n"
                f"Нажми 🗑 Сбросить чат чтобы начать заново.",
                reply_markup=token_agent_keyboard()
            )
        except Exception as e:
            logger.error(f"Ошибка token-агента: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")
        return

    # Режим агента: все сообщения идут напрямую в агент
    if user_id in agent_mode:
        agent = agent_sessions.get(user_id)
        if not agent:
            await update.message.reply_text("❌ Агент не найден. Попробуй /newchat")
            return
        try:
            typing = await update.message.reply_text("⏳ Думаю...")
            answer, tokens = agent.chat(user_text)
            await typing.delete()
            for chunk in split_text(answer):
                await update.message.reply_text(chunk)
            await update.message.reply_text(
                f"📊 Токенов: {tokens} | 💬 Сообщений в памяти: {agent.turn_count}",
                reply_markup=agent_keyboard()
            )
        except Exception as e:
            logger.error(f"Ошибка агента: {e}")
            await update.message.reply_text(f"❌ Ошибка агента: {e}")
        return

    # Обычный режим: сохраняем текст и показываем меню
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
            f"📄 Формат: {FORMAT_LABELS.get(s['format'], s['format'])}\n"
            f"🔌 API: {API_LABELS.get(s.get('api', 'deepseek'))}\n\n"
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
        elif param == "api":
            set_setting(user_id, "api", val)

        s = get_settings(user_id)
        await query.edit_message_text(
            f"⚙️ Текущие настройки:\n"
            f"🌡️ Температура: {s['temperature']}\n"
            f"🤖 Модель: {MODEL_LABELS.get(s['model'], s['model'])}\n"
            f"📄 Формат: {FORMAT_LABELS.get(s['format'], s['format'])}\n"
            f"🔌 API: {API_LABELS.get(s.get('api', 'deepseek'))}\n\n"
            f"Выбери что изменить:",
            reply_markup=settings_keyboard(user_id)
        )
        return

    # --- Агент: сброс / выход ---
    if data == "agent_reset":
        if user_id in agent_sessions:
            agent_sessions[user_id].reset()
        await query.edit_message_text(
            "🗑 История сброшена. Начинаем новый диалог!\n\nНапиши сообщение:",
            reply_markup=agent_keyboard()
        )
        return

    if data == "agent_exit":
        agent_mode.discard(user_id)
        await query.edit_message_text(
            "🚪 Вышел из режима агента.\n\nНапиши вопрос — и я покажу меню заданий."
        )
        return

    if data == "agent_simulate_restart":
        # Имитация перезапуска: удаляем агент из памяти, но JSON-история остаётся на диске
        agent = agent_sessions.pop(user_id, None)
        turns = agent.turn_count if agent else 0
        agent_mode.discard(user_id)
        await query.edit_message_text(
            f"♻️ Агент перезапущен!\n\n"
            f"💾 История ({turns} сообщений) сохранена на диске.\n\n"
            f"Напиши вопрос → выбери 💾 Задание 7\n"
            f"→ увидишь что история загрузилась автоматически!"
        )
        return

    if data == "agent_restart_model":
        s = get_settings(user_id)
        model_id = s["model"]
        api = s.get("api", "deepseek")
        if model_id == "deepseek-chat" and api == "routerai":
            client = routerai
            actual_model = "deepseek/deepseek-chat"
        elif model_id == "deepseek-chat":
            client = deepseek
            actual_model = "deepseek-chat"
        else:
            client = routerai
            actual_model = model_id
        agent = Agent(
            client=client,
            model_id=actual_model,
            user_id=user_id,
            data_dir=os.path.join(os.path.dirname(__file__), "..", "data"),
        )
        agent_sessions[user_id] = agent
        agent_mode.add(user_id)
        await query.edit_message_text(
            f"🔄 Модель обновлена!\n"
            f"🤖 {MODEL_LABELS.get(model_id, model_id)}\n"
            f"🔌 {API_LABELS.get(api)}\n"
            f"💬 История: {agent.turn_count} сообщений сохранена\n\n"
            f"Продолжай писать:",
            reply_markup=agent_keyboard()
        )
        return

    # --- Задание 8: управление token-агентом ---
    if data == "t8_reset":
        agent = token_sessions.get(user_id)
        if agent:
            agent.reset()
        await query.edit_message_text(
            "🗑 Чат сброшен. Токены обнулены.\n\nНапиши сообщение:",
            reply_markup=token_agent_keyboard()
        )
        return

    if data == "t8_exit":
        token_mode.discard(user_id)
        await query.edit_message_text(
            "🚪 Вышел из режима токенов.\n\nНапиши вопрос — и я покажу меню заданий."
        )
        return

    if data == "t8_overflow":
        await query.answer()  # сразу отвечаем Telegram чтобы не было timeout
        await query.message.reply_text(
            "🔥 РЕАЛЬНОЕ переполнение контекста DeepSeek\n\n"
            "Генерирую ~140,000 токенов текста и отправляю в API...\n"
            "Лимит модели: 131,072 токена\n\n"
            "⏳ Бот будет лагать пока грузится запрос — это нормально!"
        )
        # Генерируем текст ~140k токенов (560k символов / 4 ≈ 140k токенов)
        base = "Нейронные сети используют многослойную архитектуру для обработки данных. "
        huge_text = base * 8000
        est_tokens = len(huge_text) // 4

        user_msg = f"Проанализируй этот текст и кратко ответь о чём он:\n\n{huge_text}"

        import asyncio

        def _call_api():
            return deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Ты помощник."},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=50,
            )

        try:
            resp = await asyncio.to_thread(_call_api)
            # Неожиданно принял — показываем сколько токенов
            usage = resp.usage
            await query.message.reply_text(
                f"✅ Неожиданно! DeepSeek принял запрос:\n"
                f"prompt_tokens = {usage.prompt_tokens:,}\n"
                f"Попробуй увеличить объём текста.",
                reply_markup=token_agent_keyboard()
            )
        except BadRequestError as e:
            body = e.body or {}
            msg = body.get("message", str(e)) if isinstance(body, dict) else str(e)
            await query.message.reply_text(
                f"💥 РЕАЛЬНАЯ ОШИБКА DEEPSEEK API!\n\n"
                f"HTTP 400 BadRequestError:\n"
                f"{msg}\n\n"
                f"📊 Мы отправили: ~{est_tokens:,} токенов\n"
                f"🚫 Лимит модели: 131,072 токена\n\n"
                f"✅ Наш pre-flight check (ContextOverflowError)\n"
                f"останавливает нас ДО вызова API и экономит деньги!",
                reply_markup=token_agent_keyboard()
            )
        return

    # ── Задание 9: кнопки управления ─────────────────────────────────────────
    if data == "t9_reset":
        agent = summary_sessions.get(user_id)
        if agent:
            agent.reset()
        await query.edit_message_text(
            "🗑 Чат сброшен. Summary очищен.\n\nНапиши сообщение:",
            reply_markup=summary_agent_keyboard()
        )
        return

    if data == "t9_exit":
        summary_mode.discard(user_id)
        await query.edit_message_text(
            "🚪 Вышел из режима сжатия.\n\nНапиши вопрос — покажу меню заданий."
        )
        return

    if data == "t9_summary":
        agent = summary_sessions.get(user_id)
        if not agent:
            await query.answer("Сессия не найдена!", show_alert=True)
            return
        if not agent.summary:
            await query.answer("Summary пока пустое — история ещё не сжималась.", show_alert=True)
            return
        await query.message.reply_text(
            f"📝 ТЕКУЩЕЕ SUMMARY\n{'─' * 32}\n{agent.summary}",
            reply_markup=summary_agent_keyboard()
        )
        return

    if data == "t9_stats":
        agent = summary_sessions.get(user_id)
        if not agent or not agent.stats:
            await query.answer("Нет данных — напиши хотя бы одно сообщение!", show_alert=True)
            return
        lines = [
            "📊 СТАТИСТИКА СЕССИИ — ЗАДАНИЕ 9\n" + "─" * 36,
            f"{'Ход':>3}  {'Вход':>7}  {'Сжатие':>7}  {'Итого':>7}  {'₽':>7}",
            "─" * 36,
        ]
        for s in agent.stats:
            compress = "🗜" if s.compressed else "  "
            lines.append(
                f"{s.turn:>3}{compress} {s.prompt_tokens:>7,}  "
                f"{s.summary_tokens:>7,}  {s.total_tokens:>7,}  {s.cost_rub:>7.5f}"
            )
        total_in = sum(s.prompt_tokens + s.summary_tokens for s in agent.stats)
        total_cost = sum(s.cost_rub for s in agent.stats)
        lines += [
            "─" * 36,
            f"Итого токенов : {total_in:,}",
            f"Итого стоимость: {total_cost:.5f} ₽",
            f"Сжатий: {sum(1 for s in agent.stats if s.compressed)}",
        ]
        await query.message.reply_text("\n".join(lines), reply_markup=summary_agent_keyboard())
        return

    if data == "t8_stats":
        agent = token_sessions.get(user_id)
        if not agent or not agent.stats:
            await query.answer("Нет данных — напиши хотя бы одно сообщение!", show_alert=True)
            return
        lines = [
            "📊 СТАТИСТИКА СЕССИИ — ЗАДАНИЕ 8\n" + "─" * 32,
            f"{'Ход':>3}  {'Вход':>8}  {'Выход':>7}  {'₽/ход':>8}",
            "─" * 32,
        ]
        for s in agent.stats:
            lines.append(
                f"{s.turn:>3}  {s.prompt_tokens:>8,}  "
                f"{s.completion_tokens:>7,}  {s.turn_cost_rub:>8.5f}"
            )
        last = agent.stats[-1]
        lines += [
            "─" * 32,
            f"Итого токенов : {last.session_total:,}",
            f"Итого стоимость: {last.session_cost_rub:.5f} ₽",
            f"Ходов: {len(agent.stats)}",
        ]
        await query.message.reply_text(
            "\n".join(lines),
            reply_markup=token_agent_keyboard()
        )
        return

    if data == "admin_restart":
        if not is_admin(user_id):
            await query.answer("⛔ Нет доступа", show_alert=True)
            return
        await query.edit_message_text("🔄 Перезапускаюсь...")
        await asyncio.sleep(1)
        sys.exit(0)

    # ── Задание 13: Task State Machine ───────────────────────────────────────
    if data == "z13":
        await query.edit_message_text(
            "🔄 *ЗАДАНИЕ 13 — Task State Machine*\n\n"
            "Детерминированный конечный автомат задачи.\n\n"
            "Три этапа с разными LLM-промптами:\n"
            "📋 *Планирование* — LLM составляет план, НЕ выполняет\n"
            "⚙️ *Выполнение* — LLM выполняет по утверждённому плану\n"
            "🔍 *Валидация* — LLM проверяет и обосновывает результат\n\n"
            "Переходы детерминированы — не LLM решает, а код.\n"
            "Состояние сохраняется в JSON (пауза/продолжение).\n\n"
            "Введи задачу — и автомат запустится:",
            parse_mode="Markdown",
            reply_markup=z13_entry_keyboard(),
        )
        z13_mode.add(user_id)
        return

    # ── Задание 10: управление ────────────────────────────────────────────────
    if data == "z10":
        await query.edit_message_text(
            "🌿 ЗАДАНИЕ 10 — Стратегии управления контекстом\n\n"
            "Три подхода к хранению истории диалога:\n\n"
            "🪟 Скользящее окно — помним только последние N сообщений\n"
            "📌 Закреплённые факты — LLM извлекает факты + скользящее окно\n"
            "🌿 Ветвление — checkpoint + независимые ветки диалога\n\n"
            "🎬 Авто-демо — одно и то же ТЗ через все 3 стратегии",
            reply_markup=z10_menu_keyboard()
        )
        return

    if data == "z10_demo":
        await query.edit_message_text("🎬 Запускаю авто-демо всех стратегий...\n\nЭто займёт ~30 секунд.")
        await run_z10_auto_demo(user_id, query.message.reply_text)
        return

    if data in ("z10_sw", "z10_facts", "z10_branch"):
        strategy_map = {"z10_sw": "sliding_window", "z10_facts": "sticky_facts", "z10_branch": "branching"}
        strategy = strategy_map[data]
        await query.edit_message_text(f"⏳ Запускаю стратегию...")
        await start_z10_strategy(user_id, strategy, query.message.reply_text)
        return

    if data == "t10_exit":
        z10_mode.discard(user_id)
        await query.edit_message_text("🚪 Вышел из режима стратегий.\n\nНапиши вопрос — покажу меню заданий.")
        return

    if data == "t10_reset":
        session = z10_sessions.get(user_id)
        if session:
            session["agent"].reset()
        strategy = session["strategy"] if session else "sliding_window"
        await query.edit_message_text(
            "🗑 Чат сброшен.\n\nНапиши сообщение:",
            reply_markup=z10_chat_keyboard(strategy)
        )
        return

    if data == "t10_change":
        z10_mode.discard(user_id)
        await query.edit_message_text(
            "🔀 Выбери новую стратегию:",
            reply_markup=z10_menu_keyboard()
        )
        return

    if data == "t10_stats":
        session = z10_sessions.get(user_id)
        if not session or not session["agent"].stats:
            await query.answer("Нет данных — напиши хотя бы одно сообщение!", show_alert=True)
            return
        agent = session["agent"]
        strategy = session["strategy"]
        lines = [
            f"📊 СТАТИСТИКА — {'СКОЛ. ОКНО' if strategy == 'sliding_window' else 'ФАКТЫ' if strategy == 'sticky_facts' else 'ВЕТВЛЕНИЕ'}\n" + "─" * 36,
            f"{'Ход':>3}  {'Вход':>7}  {'Доп':>5}  {'Выход':>7}  {'₽':>8}",
            "─" * 36,
        ]
        for s in agent.stats:
            lines.append(
                f"{s.turn:>3}  {s.prompt_tokens:>7,}  {s.extra_tokens:>5,}  "
                f"{s.completion_tokens:>7,}  {s.cost_rub:>8.5f}"
            )
        total_cost = sum(s.cost_rub for s in agent.stats)
        total_tok = sum(s.total_tokens for s in agent.stats)
        lines += ["─" * 36, f"Итого токенов : {total_tok:,}", f"Итого стоимость: {total_cost:.5f} ₽"]
        has_branch = strategy == "branching" and agent.checkpoint is not None
        await query.message.reply_text("\n".join(lines), reply_markup=z10_chat_keyboard(strategy, has_branch))
        return

    if data == "t10_show_facts":
        session = z10_sessions.get(user_id)
        if not session or session["strategy"] != "sticky_facts":
            await query.answer("Не в режиме закреплённых фактов!", show_alert=True)
            return
        agent = session["agent"]
        await query.message.reply_text(
            f"📌 ЗАКРЕПЛЁННЫЕ ФАКТЫ\n{'─' * 30}\n{agent.get_facts_text()}",
            reply_markup=z10_chat_keyboard("sticky_facts")
        )
        return

    if data == "t10_checkpoint":
        session = z10_sessions.get(user_id)
        if not session or session["strategy"] != "branching":
            await query.answer("Не в режиме ветвления!", show_alert=True)
            return
        agent = session["agent"]
        agent.create_checkpoint()
        hist_len = len(agent.checkpoint)
        await query.message.reply_text(
            f"📍 Checkpoint создан!\n"
            f"Сохранено {hist_len} сообщений.\n\n"
            f"Теперь можешь форкнуть в ветку А или Б:",
            reply_markup=z10_chat_keyboard("branching", has_branch=True)
        )
        return

    if data in ("t10_fork_a", "t10_fork_b"):
        session = z10_sessions.get(user_id)
        if not session or session["strategy"] != "branching":
            await query.answer("Не в режиме ветвления!", show_alert=True)
            return
        agent = session["agent"]
        branch_name = BRANCH_A if data == "t10_fork_a" else BRANCH_B
        emoji = "🌿" if data == "t10_fork_a" else "🌱"
        agent.fork(branch_name, from_checkpoint=True)
        await query.message.reply_text(
            f"{emoji} Форкнул в «{branch_name}»\n"
            f"Ветка создана из checkpoint. Пиши сообщение в этой ветке:",
            reply_markup=z10_chat_keyboard("branching", has_branch=True)
        )
        return

    # ── Задание 11 callbacks ─────────────────────────────────────────
    if data == "t11_show_memory":
        agent = z11_sessions.get(user_id)
        if not agent:
            await query.answer("Сессия не найдена!", show_alert=True)
            return
        all_facts = agent.ltm.all_facts()
        stm_msgs = agent.stm.get()
        wm_data = agent.wm.all()
        lines = [f"🧠 ПАМЯТЬ АССИСТЕНТА\n{'─' * 32}"]
        lines.append(f"📥 STM (краткосрочная) — {len(stm_msgs)} сообщений:")
        for m in stm_msgs[-4:]:
            role_emoji = "👤" if m["role"] == "user" else "🤖"
            lines.append(f"  {role_emoji} {m['content'][:80]}{'…' if len(m['content']) > 80 else ''}")
        lines.append(f"\n🗂 WM (рабочая) — {len(wm_data)} записей:")
        if wm_data:
            for k, v in wm_data.items():
                lines.append(f"  • {k}: {v}")
        else:
            lines.append("  (пусто)")
        lines.append(f"\n🗄 LTM (долгосрочная) — {len(all_facts)} фактов:")
        if all_facts:
            for f in all_facts:
                lines.append(f"  • {f}")
        else:
            lines.append("  (пусто — напиши что-то о себе!)")
        await query.message.reply_text("\n".join(lines), reply_markup=z11_chat_keyboard())
        return

    if data == "t11_stats":
        agent = z11_sessions.get(user_id)
        if not agent or not agent.stats:
            await query.answer("Нет данных — напиши хотя бы одно сообщение!", show_alert=True)
            return
        lines = [f"📊 СТАТИСТИКА — ПАМЯТЬ АССИСТЕНТА\n{'─' * 36}"]
        lines.append(f"{'Ход':>3}  {'Вход':>7}  {'Выход':>7}  {'LTM↓':>5}  {'LTM↑':>5}  {'₽':>8}")
        lines.append("─" * 36)
        for s in agent.stats:
            lines.append(
                f"{s.turn:>3}  {s.prompt_tokens:>7,}  {s.completion_tokens:>7,}  "
                f"{s.ltm_facts_loaded:>5}  {s.ltm_facts_saved:>5}  {s.cost_rub:>8.5f}"
            )
        total_cost = sum(s.cost_rub for s in agent.stats)
        total_tok = sum(s.total_tokens for s in agent.stats)
        lines += [
            "─" * 36,
            f"Итого токенов  : {total_tok:,}",
            f"Итого стоимость: {total_cost:.5f} ₽",
            f"Фактов в LTM   : {agent.ltm.count()}",
        ]
        await query.message.reply_text("\n".join(lines), reply_markup=z11_chat_keyboard())
        return

    if data == "t11_reset_session":
        agent = z11_sessions.get(user_id)
        if agent:
            agent.reset_session()
        await query.edit_message_text(
            "🗑 Сессия сброшена (STM + WM очищены).\n"
            "LTM (долгосрочная память) сохранена.\n\n"
            "Напиши что-нибудь — агент помнит тебя из прошлых сессий:",
            reply_markup=z11_chat_keyboard()
        )
        return

    if data == "t11_forget_all":
        agent = z11_sessions.get(user_id)
        if agent:
            agent.forget_all()
        await query.edit_message_text(
            "🧹 Полная очистка — забыто всё (STM + WM + LTM).\n\n"
            "Агент не помнит о тебе ничего. Начнём с чистого листа.\n"
            "Напиши что-нибудь:",
            reply_markup=z11_chat_keyboard()
        )
        return

    if data == "t11_demo":
        agent = z11_sessions.get(user_id)
        if not agent:
            await query.answer("Сессия не найдена!", show_alert=True)
            return
        await query.edit_message_text("🎬 Запускаю авто-демо памяти...")
        await run_z11_auto_demo(user_id, query.message.reply_text)
        return

    if data == "t11_exit":
        z11_mode.discard(user_id)
        await query.edit_message_text(
            "👋 Вышел из режима памяти ассистента.\n\n"
            "LTM (долгосрочная память) сохранена — при следующем входе агент тебя вспомнит!"
        )
        return

    # ── Задание 12 callbacks ─────────────────────────────────────────
    if data == "t12_profile":
        agent = z12_sessions.get(user_id)
        if not agent:
            await query.answer("Сессия не найдена!", show_alert=True)
            return
        text = f"👤 *Текущий профиль*\n\n{agent.profile.as_summary()}\n\nLTM: {agent.ltm.count()} фактов"
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=z12_chat_keyboard())
        return

    if data == "t12_switch":
        await query.edit_message_text(
            "🎭 Выбери профиль:",
            reply_markup=z12_preset_keyboard(),
        )
        return

    if data.startswith("t12_preset_"):
        preset_key = data.replace("t12_preset_", "")
        preset_map = {"техник": "техник", "менеджер": "менеджер", "новичок": "новичок"}
        if preset_key not in preset_map:
            await query.answer("Неизвестный профиль", show_alert=True)
            return
        profile = UserProfile.from_preset(str(user_id), preset_map[preset_key])
        agent = z12_sessions.get(user_id)
        if not agent:
            await query.answer("Сессия не найдена!", show_alert=True)
            return
        agent.set_profile(profile)
        agent.reset_session()
        label = PRESET_PROFILES[preset_map[preset_key]]["display_name"]
        await query.edit_message_text(
            f"✅ Профиль переключён: *{label}*\n\n{profile.as_summary()}\n\nСессия сброшена. Пиши — отвечу в новом стиле!",
            parse_mode="Markdown",
            reply_markup=z12_chat_keyboard(),
        )
        return

    if data == "t12_reset":
        agent = z12_sessions.get(user_id)
        if agent:
            agent.reset_session()
        await query.edit_message_text(
            "🗑 Сессия сброшена (STM очищен). Профиль и LTM сохранены.",
            reply_markup=z12_chat_keyboard(),
        )
        return

    if data == "t12_demo":
        agent = z12_sessions.get(user_id)
        if not agent:
            await query.answer("Сессия не найдена!", show_alert=True)
            return
        await query.edit_message_text("🎭 Запускаю демо трёх профилей...")
        await run_z12_auto_demo(user_id, query.message.reply_text)
        return

    if data == "t12_exit":
        z12_mode.discard(user_id)
        await query.edit_message_text(
            "👋 Вышел из режима персонализации.\n\n"
            "Профиль сохранён — при следующем входе агент будет в том же стиле."
        )
        return

    # ── Задание 13 callbacks ─────────────────────────────────────────
    if data == "t13_advance":
        ctx = z13_sessions.get(user_id)
        if not ctx:
            await query.answer("Задача не найдена!", show_alert=True)
            return
        fsm = get_z13_fsm()
        new_state = fsm.advance(ctx)
        if new_state == TaskState.DONE:
            await query.edit_message_text(
                f"✅ *Задача завершена!*\n\n{ctx.summary()}",
                parse_mode="Markdown",
                reply_markup=z13_task_keyboard(new_state),
            )
            return
        intro = fsm.get_stage_intro(ctx)
        await query.edit_message_text(
            intro or f"▶️ Переходим к: {ctx.label()}",
            parse_mode="Markdown",
        )
        await query.message.reply_text("⏳ Запускаю этап...")
        result = await asyncio.to_thread(fsm.run_current_stage, ctx)
        for chunk in split_text(result):
            await query.message.reply_text(chunk)
        await query.message.reply_text(
            ctx.summary(),
            parse_mode="Markdown",
            reply_markup=z13_task_keyboard(ctx.state()),
        )
        return

    if data == "t13_retry":
        ctx = z13_sessions.get(user_id)
        if not ctx:
            await query.answer("Задача не найдена!", show_alert=True)
            return
        fsm = get_z13_fsm()
        fsm.retry_execution(ctx)
        await query.edit_message_text("🔁 Повторяю выполнение...")
        result = await asyncio.to_thread(fsm.run_current_stage, ctx)
        for chunk in split_text(result):
            await query.message.reply_text(chunk)
        await query.message.reply_text(
            ctx.summary(),
            parse_mode="Markdown",
            reply_markup=z13_task_keyboard(ctx.state()),
        )
        return

    if data == "t13_pause":
        ctx = z13_sessions.get(user_id)
        if not ctx:
            await query.answer("Задача не найдена!", show_alert=True)
            return
        get_z13_fsm().pause(ctx)
        await query.edit_message_text(
            f"⏸️ *Задача поставлена на паузу*\n\n{ctx.summary()}\n\n"
            f"Состояние сохранено. Нажми «▶️ Продолжить» чтобы вернуться.",
            parse_mode="Markdown",
            reply_markup=z13_task_keyboard(ctx.state()),
        )
        return

    if data == "t13_resume":
        ctx = z13_sessions.get(user_id)
        if not ctx:
            await query.answer("Задача не найдена!", show_alert=True)
            return
        fsm = get_z13_fsm()
        resumed_state = fsm.resume(ctx)
        await query.edit_message_text(
            f"▶️ *Продолжаю с этапа: {ctx.label()}*\n\n{ctx.summary()}",
            parse_mode="Markdown",
            reply_markup=z13_task_keyboard(resumed_state),
        )
        return

    if data == "t13_new":
        z13_mode.add(user_id)
        z13_sessions.pop(user_id, None)
        await query.edit_message_text(
            "📝 Введи новую задачу для автомата:",
            reply_markup=z13_entry_keyboard(),
        )
        return

    if data == "t13_demo":
        await query.edit_message_text("🎬 Запускаю авто-демо Task State Machine...")
        await run_z13_auto_demo(user_id, query.message.reply_text)
        return

    if data == "t13_exit":
        z13_mode.discard(user_id)
        z13_sessions.pop(user_id, None)
        await query.edit_message_text(
            "👋 Вышел из Task State Machine.\n\n"
            "Состояние задачи удалено из памяти."
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
        send = query.message.reply_text

        if data == "z1":
            parts = await run_zadanie1(text, s)
        elif data == "z2":
            parts = await run_zadanie2(text)
        elif data == "z3":
            parts = await run_zadanie3(text)
        elif data == "z4":
            parts = await run_zadanie4(text)
        elif data == "z5":
            parts = await run_zadanie5(text, send)
            parts = []  # z5 отправляет сам по мере поступления
        elif data == "z6":
            await start_agent_mode(user_id, text, s, send)
            parts = []  # z6 сам управляет сообщениями
        elif data == "z7":
            await start_agent_mode(user_id, text, s, send, zadanie=7)
            parts = []  # z7 сам управляет сообщениями
        elif data == "z8":
            await start_token_agent_mode(user_id, text, send)
            parts = []  # z8 сам управляет сообщениями
        elif data == "z9":
            await start_summary_agent_mode(user_id, text, send)
            parts = []  # z9 сам управляет сообщениями
        elif data == "z10":
            # z10 через pending_text — запускаем меню
            await query.edit_message_text(
                "🌿 ЗАДАНИЕ 10 — Стратегии управления контекстом\n\n"
                "Три подхода к хранению истории диалога:\n\n"
                "🪟 Скользящее окно — помним только последние N сообщений\n"
                "📌 Закреплённые факты — LLM извлекает факты + скользящее окно\n"
                "🌿 Ветвление — checkpoint + независимые ветки диалога\n\n"
                "🎬 Авто-демо — одно и то же ТЗ через все 3 стратегии",
                reply_markup=z10_menu_keyboard()
            )
            parts = []
        elif data == "z11":
            await start_z11_agent(user_id, text, send)
            parts = []
        elif data == "z12":
            await start_z12_agent(user_id, text, send)
            parts = []
        elif data == "z13":
            await start_z13_task(user_id, text, send)
            parts = []
        else:
            parts = ["Неизвестное задание"]

        for part in parts:
            for chunk in split_text(part):
                await send(chunk)

    except Exception as e:
        logger.error(f"Ошибка в задании {data}: {e}")
        await query.message.reply_text(f"❌ Ошибка: {str(e)}")


def split_text(text, max_len=4000):
    """Разбивает текст на части по границам абзацев."""
    if len(text) <= max_len:
        return [text]
    parts = []
    while len(text) > max_len:
        cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = max_len
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        parts.append(text)
    return parts


# ─────────────────────────── ЗАДАНИЕ 6 — АГЕНТ ─────────────────────

async def start_agent_mode(user_id, first_message, settings, send, zadanie=6):
    """Активировать режим агента и обработать первое сообщение."""
    model_id = settings["model"]
    api = settings.get("api", "deepseek")

    # Выбираем клиент и модель (те же правила что в zadanie1)
    if model_id == "deepseek-chat" and api == "routerai":
        client = routerai
        actual_model = "deepseek/deepseek-chat"
    elif model_id == "deepseek-chat":
        client = deepseek
        actual_model = "deepseek-chat"
    else:
        client = routerai
        actual_model = model_id

    # Создаём агента с user_id — он загрузит историю из файла автоматически
    agent = Agent(
        client=client,
        model_id=actual_model,
        user_id=user_id,
        data_dir=os.path.join(os.path.dirname(__file__), "..", "data"),
    )
    agent_sessions[user_id] = agent
    agent_mode.add(user_id)

    z_label = {6: "💬 ЗАДАНИЕ 6 — Агент (чат с историей)", 7: "💾 ЗАДАНИЕ 7 — Агент с памятью"}[zadanie]
    if agent.is_restored:
        status = (
            f"{z_label}\n"
            f"🤖 Модель: {MODEL_LABELS.get(model_id, model_id)}\n"
            f"🔌 API: {API_LABELS.get(api)}\n\n"
            f"📂 Загружена история: {agent.turn_count} сообщений\n"
            f"Продолжаем с того места где остановились!\n\n"
            f"/newchat — начать новый диалог"
        )
    else:
        status = (
            f"{z_label}\n"
            f"🤖 Модель: {MODEL_LABELS.get(model_id, model_id)}\n"
            f"🔌 API: {API_LABELS.get(api)}\n\n"
            f"Новый диалог. История сохраняется автоматически.\n"
            f"/newchat — сбросить историю"
        )
    await send(status)

    # Первое сообщение сразу отправляем в агент
    answer, tokens = agent.chat(first_message)
    for chunk in split_text(answer):
        await send(chunk)
    await send(
        f"📊 Токенов: {tokens} | 💬 Сообщений в памяти: {agent.turn_count}",
        reply_markup=agent_keyboard()
    )


# ─────────────────────────── ЗАДАНИЕ 8 — TOKEN AGENT ───────────────

def format_token_stat(stat, agent: TokenAwareAgent) -> str:
    """Форматирует статистику токенов для отправки в Telegram."""
    ctx_pct = stat.prompt_tokens / agent.context_limit * 100
    bar_filled = min(16, int(16 * stat.prompt_tokens / agent.context_limit))
    bar = "█" * bar_filled + "░" * (16 - bar_filled)

    return (
        f"📊 ТОКЕНЫ — ХОД {stat.turn}\n"
        f"{'─' * 30}\n"
        f"Вход  (prompt)  : {stat.prompt_tokens:>8,} токенов\n"
        f"Выход (ответ)   : {stat.completion_tokens:>8,} токенов\n"
        f"Итого за ход    : {stat.total_tokens:>8,}\n"
        f"{'─' * 30}\n"
        f"Нарастающий     : {stat.session_total:>8,} токенов\n"
        f"Стоимость хода  : {stat.turn_cost_rub:>10.5f} ₽\n"
        f"Стоимость сессии: {stat.session_cost_rub:>10.5f} ₽\n"
        f"{'─' * 30}\n"
        f"Контекст: [{bar}] {ctx_pct:.1f}%"
    )


async def start_token_agent_mode(user_id: int, first_message: str, send) -> None:
    """Запустить токен-агент (задание 8) и обработать первое сообщение."""
    agent = TokenAwareAgent(
        deepseek,
        model_id="deepseek-chat",
        system_prompt="Ты полезный ассистент. Отвечай по делу.",
    )
    token_sessions[user_id] = agent
    token_mode.add(user_id)

    await send(
        "🔢 ЗАДАНИЕ 8 — Агент с подсчётом токенов\n\n"
        "После каждого ответа показываю:\n"
        "• prompt_tokens — вход (история + запрос)\n"
        "• completion_tokens — ответ модели\n"
        "• нарастающий итог и стоимость\n"
        "• заполнение контекста (64 000 токенов)\n\n"
        "Пиши — и токены будут считаться в реальном времени!\n"
        "💥 Кнопка «Симулировать переполнение» — показывает что ломается."
    )

    try:
        answer, stat = agent.chat(first_message)
        for chunk in split_text(answer):
            await send(chunk)
        await send(format_token_stat(stat, agent), reply_markup=token_agent_keyboard())
    except Exception as e:
        logger.error(f"Ошибка token-агента (первое сообщение): {e}")
        await send(f"❌ Ошибка: {e}")


# ─────────────────────────── ЗАДАНИЕ 10 — СТРАТЕГИИ КОНТЕКСТА ───────

def _z10_stat_text(stat, agent) -> str:
    """Форматировать статистику одного хода для задания 10."""
    strategy_labels = {
        "sliding_window": "🪟 Скользящее окно",
        "sticky_facts":   "📌 Закреплённые факты",
    }
    strategy_raw = stat.strategy
    # branching[ветка-а] → extract just "branching"
    if strategy_raw.startswith("branching"):
        label = f"🌿 Ветвление [{agent.active_branch}]"
    else:
        label = strategy_labels.get(strategy_raw, strategy_raw)

    lines = [
        f"📊 Ход {stat.turn} · {label}",
        "─" * 30,
        f"Вход         : {stat.prompt_tokens:>7,} т",
        f"Выход        : {stat.completion_tokens:>7,} т",
    ]
    if stat.extra_tokens:
        lines.append(f"Факты (доп.) : {stat.extra_tokens:>7,} т")
    lines.append(f"Итого        : {stat.total_tokens:>7,} т")
    lines.append(f"Стоимость    : {stat.cost_rub:.4f} ₽")
    lines.append(f"Контекст     : {agent.context_size()} сообщ.")
    return "\n".join(lines)


async def start_z10_strategy(user_id: int, strategy: str, send) -> None:
    """Создать агента нужной стратегии и переключить пользователя в режим z10."""
    if strategy == "sliding_window":
        agent = SlidingWindowAgent(deepseek, "deepseek-chat", Z10_SYSTEM_PROMPT, window_size=8)
        intro = (
            "🪟 СКОЛЬЗЯЩЕЕ ОКНО\n\n"
            "Как работает:\n"
            "• В контексте всегда только последние 8 сообщений\n"
            "• Старые сообщения автоматически отбрасываются\n"
            "• Просто и дёшево — но факты из начала могут «забыться»\n\n"
            "Напиши первое сообщение:"
        )
    elif strategy == "sticky_facts":
        agent = StickyFactsAgent(deepseek, "deepseek-chat", Z10_SYSTEM_PROMPT, window_size=6)
        intro = (
            "📌 ЗАКРЕПЛЁННЫЕ ФАКТЫ\n\n"
            "Как работает:\n"
            "• После каждого хода LLM извлекает ключевые факты (JSON)\n"
            "• Факты всегда присутствуют в system-промпте\n"
            "• + последние 6 сообщений в контексте\n"
            "• Важное не теряется даже в длинных диалогах\n\n"
            "Напиши первое сообщение:"
        )
    else:  # branching
        agent = BranchingAgent(deepseek, "deepseek-chat", Z10_SYSTEM_PROMPT)
        intro = (
            "🌿 ВЕТВЛЕНИЕ\n\n"
            "Как работает:\n"
            "• Создаёшь checkpoint — снимок текущей истории\n"
            "• Форкаешь: ветка А и ветка Б развиваются независимо\n"
            "• Переключаешься между ветками, сравниваешь ответы\n\n"
            "Напиши первое сообщение (или создай checkpoint сразу):"
        )

    z10_sessions[user_id] = {"strategy": strategy, "agent": agent}
    z10_mode.add(user_id)
    await send(intro)


async def run_z10_auto_demo(user_id: int, send) -> None:
    """Прогнать три стратегии на одном сценарии и показать результаты."""
    await send(
        "🎬 АВТО-ДЕМО — 3 стратегии управления контекстом\n\n"
        "Один и тот же диалог пройдём через:\n"
        "1️⃣ Скользящее окно\n"
        "2️⃣ Закреплённые факты\n"
        "3️⃣ Ветвление (с форком после 2 сообщений)\n\n"
        "⏳ Запускаю..."
    )

    # ── 1. Sliding Window ────────────────────────────────────────────
    await send("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n🪟 СТРАТЕГИЯ 1 — Скользящее окно\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    sw = SlidingWindowAgent(deepseek, "deepseek-chat", Z10_SYSTEM_PROMPT, window_size=8)
    sw_total = 0
    for msg in DEMO_SHORT:
        await send(f"👤 Пользователь:\n{msg}")
        try:
            reply, stat = await asyncio.to_thread(sw.chat, msg)
            sw_total += stat.total_tokens
            for chunk in split_text(reply):
                await send(f"🤖 Агент:\n{chunk}")
            await send(_z10_stat_text(stat, sw))
        except Exception as e:
            await send(f"❌ Ошибка: {e}")
            break

    # ── 2. Sticky Facts ──────────────────────────────────────────────
    await send("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n📌 СТРАТЕГИЯ 2 — Закреплённые факты\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    sf = StickyFactsAgent(deepseek, "deepseek-chat", Z10_SYSTEM_PROMPT, window_size=6)
    sf_total = 0
    for msg in DEMO_SHORT:
        await send(f"👤 Пользователь:\n{msg}")
        try:
            reply, stat = await asyncio.to_thread(sf.chat, msg)
            sf_total += stat.total_tokens
            for chunk in split_text(reply):
                await send(f"🤖 Агент:\n{chunk}")
            await send(_z10_stat_text(stat, sf))
        except Exception as e:
            await send(f"❌ Ошибка: {e}")
            break
    await send(f"📌 Извлечённые факты:\n{sf.get_facts_text()}")

    # ── 3. Branching ─────────────────────────────────────────────────
    await send("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n🌿 СТРАТЕГИЯ 3 — Ветвление\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    br = BranchingAgent(deepseek, "deepseek-chat", Z10_SYSTEM_PROMPT)
    br_total = 0
    # Общая база (2 сообщения)
    await send("📋 Общая база (main):")
    for msg in BRANCH_DEMO_BASE:
        await send(f"👤 Пользователь:\n{msg}")
        try:
            reply, stat = await asyncio.to_thread(br.chat, msg)
            br_total += stat.total_tokens
            for chunk in split_text(reply):
                await send(f"🤖 Агент:\n{chunk}")
        except Exception as e:
            await send(f"❌ Ошибка: {e}")
            break

    # Checkpoint + форки
    br.create_checkpoint()
    await send("📍 Checkpoint создан. Форкаю в две ветки...")

    # Ветка А — бюджет срезан
    br.fork(BRANCH_A, from_checkpoint=True)
    await send(f"🌿 {BRANCH_A} — бюджет срезан:")
    await send(f"👤 Пользователь:\n{BRANCH_DEMO_FORK_A}")
    try:
        reply_a, stat_a = await asyncio.to_thread(br.chat, BRANCH_DEMO_FORK_A)
        br_total += stat_a.total_tokens
        for chunk in split_text(reply_a):
            await send(f"🤖 Агент [{BRANCH_A}]:\n{chunk}")
    except Exception as e:
        await send(f"❌ Ошибка: {e}")

    # Ветка Б — бюджет увеличен
    br.fork(BRANCH_B, from_checkpoint=True)
    await send(f"🌱 {BRANCH_B} — бюджет увеличен:")
    await send(f"👤 Пользователь:\n{BRANCH_DEMO_FORK_B}")
    try:
        reply_b, stat_b = await asyncio.to_thread(br.chat, BRANCH_DEMO_FORK_B)
        br_total += stat_b.total_tokens
        for chunk in split_text(reply_b):
            await send(f"🤖 Агент [{BRANCH_B}]:\n{chunk}")
    except Exception as e:
        await send(f"❌ Ошибка: {e}")

    # ── Итоговая сводка ──────────────────────────────────────────────
    await send(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📊 ИТОГ АВТО-ДЕМО — расход токенов\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🪟 Скользящее окно   : {sw_total:,} т\n"
        f"📌 Закреплённые факты: {sf_total:,} т  (+доп. вызов на факты)\n"
        f"🌿 Ветвление          : {br_total:,} т  (main + две ветки)"
    )

    # ── Сравнительные выводы ─────────────────────────────────────────
    await send(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🔬 СРАВНЕНИЕ СТРАТЕГИЙ\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "📝 Качество ответа\n"
        "🪟 Окно — хорошее, пока диалог короткий\n"
        "📌 Факты — стабильно точное: факты всегда в промпте\n"
        "🌿 Ветвление — отличное внутри ветки, форки не мешают друг другу\n"
        "\n"
        "🧠 Стабильность (не теряет детали)\n"
        "🪟 Окно — ⚠️ теряет всё что вышло за N сообщений\n"
        "📌 Факты — ✅ ключевые параметры сохраняются всегда\n"
        "🌿 Ветвление — ✅ каждая ветка помнит свою историю полностью\n"
        "\n"
        "💰 Расход токенов\n"
        "🪟 Окно — ✅ самый дешёвый: только окно, без extras\n"
        "📌 Факты — ⚠️ дороже: +1 вызов на извлечение фактов за ход\n"
        "🌿 Ветвление — ⚠️ растёт с числом веток и глубиной истории\n"
        "\n"
        "👤 Удобство для пользователя\n"
        "🪟 Окно — ✅ просто: пишешь и получаешь ответ\n"
        "📌 Факты — ✅ прозрачно: можно посмотреть что запомнилось\n"
        "🌿 Ветвление — 🛠 требует осознанных действий (checkpoint → fork)\n"
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📌 Когда что выбирать?\n"
        "• Короткий диалог → 🪟 Скользящее окно\n"
        "• Длинный диалог с важными фактами → 📌 Закреплённые факты\n"
        "• «А что если...» сценарии / сравнение вариантов → 🌿 Ветвление",
        reply_markup=z10_menu_keyboard()
    )


# ─────────────────────────── ЗАДАНИЕ 9 — SUMMARY AGENT ─────────────

async def start_summary_agent_mode(user_id: int, first_message: str, send) -> None:
    """Запустить агент со сжатием истории (задание 9) и обработать первое сообщение."""
    agent = SummaryAgent(
        deepseek,
        model_id="deepseek-chat",
        system_prompt="Ты полезный ассистент. Отвечай по делу.",
        keep_recent=6,
        compress_after=6,
    )
    summary_sessions[user_id] = agent
    summary_mode.add(user_id)

    await send(
        "🗜 ЗАДАНИЕ 9 — Агент со сжатием истории\n\n"
        "Как работает:\n"
        "• Последние 6 сообщений хранятся как есть\n"
        "• Когда накапливается ещё 6+ — старые сжимаются в summary\n"
        "• Summary подставляется в запрос вместо старой истории\n"
        "• Токены экономятся при длинных диалогах!\n\n"
        "🗜 — значок появляется когда произошло сжатие\n"
        "📊 Статистика — сравни токены до/после сжатия\n"
        "📝 Показать summary — текущий сжатый контекст"
    )

    try:
        answer, stat = agent.chat(first_message)
        for chunk in split_text(answer):
            await send(chunk)

        stat_text = (
            f"📊 Ход {stat.turn}\n"
            f"{'─' * 28}\n"
            f"Вход (контекст)  : {stat.prompt_tokens:>7,} т\n"
            f"Итого за ход     : {stat.total_tokens:>7,} т\n"
            f"Summary есть     : {'да ✅' if agent.summary else 'нет —'}"
        )
        await send(stat_text, reply_markup=summary_agent_keyboard())
    except Exception as e:
        logger.error(f"Ошибка summary-агента (первое сообщение): {e}")
        await send(f"❌ Ошибка: {e}")


# ─────────────────────────── ЗАДАНИЕ 11 ────────────────────────────

async def start_z11_agent(user_id: int, first_message: str, send) -> None:
    """Создать MemoryAgent для пользователя и обработать первое сообщение."""
    persist_dir = os.path.join(os.path.dirname(__file__), "..", ".chroma_db")
    agent = MemoryAgent(
        client=deepseek,
        user_id=user_id,
        model="deepseek-chat",
        stm_window=10,
        persist_dir=persist_dir,
    )
    z11_sessions[user_id] = agent
    z11_mode.add(user_id)

    existing_facts = agent.ltm.count()
    intro_suffix = (
        f"\n\n📚 Я помню тебя из прошлой сессии ({existing_facts} фактов в LTM)."
        if existing_facts > 0
        else "\n\nЭто первая встреча — расскажи о себе, я запомню!"
    )

    await send(
        "🧠 ЗАДАНИЕ 11 — Модель памяти ассистента\n\n"
        "Три слоя памяти:\n"
        "• STM (краткосрочная) — последние 10 сообщений сессии\n"
        "• WM (рабочая) — контекст текущей задачи\n"
        "• LTM (долгосрочная) — факты о тебе в ChromaDB (не исчезают!)\n\n"
        "Как работает:\n"
        "1️⃣ LLM ищет похожие факты в LTM по запросу\n"
        "2️⃣ Факты + STM → формирует контекст\n"
        "3️⃣ После ответа — извлекает новые факты и сохраняет в LTM"
        + intro_suffix
    )

    try:
        typing = await send("⏳ Думаю...")
        reply, stat = await asyncio.to_thread(agent.chat, first_message)
        await typing.delete()
        for chunk in split_text(reply):
            await send(chunk)
        stat_text = (
            f"📊 ХОД {stat.turn}\n"
            f"{'─' * 28}\n"
            f"Токены вход/выход : {stat.prompt_tokens:,} / {stat.completion_tokens:,}\n"
            f"Стоимость         : {stat.cost_rub:.4f} ₽\n"
            f"LTM загружено     : {stat.ltm_facts_loaded} фактов\n"
            f"LTM сохранено     : {stat.ltm_facts_saved} новых фактов\n"
            f"Всего в LTM       : {agent.ltm.count()} фактов"
        )
        await send(stat_text, reply_markup=z11_chat_keyboard())
    except Exception as e:
        logger.error(f"Ошибка z11-агента (первое сообщение): {e}")
        await send(f"❌ Ошибка: {e}")


async def run_z11_auto_demo(user_id: int, send) -> None:
    """Авто-демо: сначала вводим факты (DEMO_INTRO), потом проверяем память (DEMO_RECALL)."""
    agent = z11_sessions.get(user_id)
    if not agent:
        await send("❌ Сессия не найдена. Войди в задание 11 заново.")
        return

    # Сбросить сессию, но сохранить LTM
    agent.reset_session()

    await send(
        "🎬 АВТО-ДЕМО — Долгосрочная память\n\n"
        "Фаза 1: Вводим информацию о пользователе (3 сообщения)\n"
        "→ Агент извлечёт и сохранит факты в LTM\n\n"
        "Фаза 2: Новая сессия (STM сброшен)\n"
        "→ Проверяем, помнит ли агент факты только из LTM\n\n"
        "⏳ Запускаю..."
    )

    # ── Фаза 1: ввод фактов ─────────────────────────────────────────
    await send("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n📥 ФАЗА 1 — Запоминаем факты\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for msg in DEMO_INTRO:
        await send(f"👤 Пользователь:\n{msg}")
        try:
            reply, stat = await asyncio.to_thread(agent.chat, msg)
            for chunk in split_text(reply):
                await send(f"🤖 Агент:\n{chunk}")
            await send(
                f"📊 LTM: сохранено {stat.ltm_facts_saved} фактов | "
                f"Итого в LTM: {agent.ltm.count()}"
            )
        except Exception as e:
            await send(f"❌ Ошибка: {e}")

    # Показать все сохранённые факты
    all_facts = agent.ltm.all_facts()
    facts_display = "\n".join(f"  • {f}" for f in all_facts) if all_facts else "  (нет)"
    await send(f"🗄 Факты в LTM:\n{facts_display}")

    # ── Сброс STM — имитация новой сессии ──────────────────────────
    agent.stm.clear()
    agent.wm.clear()
    await send(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🔄 НОВАЯ СЕССИЯ — STM сброшен\n"
        "Агент не видит предыдущего диалога.\n"
        "Только LTM сохранена.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    # ── Фаза 2: проверка памяти ──────────────────────────────────────
    await send("📤 ФАЗА 2 — Проверяем долгосрочную память")
    for msg in DEMO_RECALL:
        await send(f"👤 Пользователь:\n{msg}")
        try:
            reply, stat = await asyncio.to_thread(agent.chat, msg)
            for chunk in split_text(reply):
                await send(f"🤖 Агент:\n{chunk}")
            await send(f"📊 LTM загружено: {stat.ltm_facts_loaded} фактов | токены: {stat.total_tokens:,}")
        except Exception as e:
            await send(f"❌ Ошибка: {e}")

    await send(
        "✅ Демо завершено!\n\n"
        "Агент отвечал правильно, используя только LTM —\n"
        "без истории диалога в STM.\n\n"
        f"Итого фактов в LTM: {agent.ltm.count()}",
        reply_markup=z11_chat_keyboard()
    )


# ─────────────────────────── ЗАДАНИЕ 1 ─────────────────────────────

async def run_zadanie1(text, settings):
    model_id = settings["model"]
    temp = settings["temperature"]
    fmt = settings["format"]
    api = settings.get("api", "deepseek")
    system_prompt = FORMAT_PROMPTS[fmt]

    # Выбираем клиент и реальный model_id
    if model_id == "deepseek-chat" and api == "routerai":
        client = routerai
        actual_model = "deepseek/deepseek-chat"
    elif model_id == "deepseek-chat":
        client = deepseek
        actual_model = "deepseek-chat"
    else:
        client = routerai
        actual_model = model_id

    resp = client.chat.completions.create(
        model=actual_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=temp,
        max_tokens=500
    )
    answer = resp.choices[0].message.content
    tokens = resp.usage.total_tokens
    api_label = API_LABELS.get(api)
    return [
        f"📝 ЗАДАНИЕ 1 — Ответ с настройками\n"
        f"🤖 Модель: {MODEL_LABELS.get(model_id, model_id)}\n"
        f"🔌 API: {api_label}\n"
        f"🌡️ Температура: {temp}\n"
        f"📄 Формат: {FORMAT_LABELS[fmt]}\n"
        "─" * 30 + "\n"
        f"{answer}\n\n"
        f"📊 Токенов: {tokens}"
    ]


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

    return out


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

    return out


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

    return out


# ─────────────────────────── ЗАДАНИЕ 5 ─────────────────────────────

async def run_zadanie5(text, send):
    await send("🤖 ЗАДАНИЕ 5 — Сравнение моделей")

    models = [
        ("amazon/nova-micro-v1",      "Слабая (Nova Micro)"),
        ("deepseek/deepseek-chat",    "Средняя (DeepSeek V3)"),
        ("anthropic/claude-opus-4.6", "Сильная (Claude Opus 4.6)"),
    ]

    answers = []
    for model_id, label in models:
        await send(f"⏳ Запрашиваю {label}...")
        t0 = time.time()
        r = routerai.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": text}]
        )
        elapsed = time.time() - t0
        answer = r.choices[0].message.content
        answers.append((label, answer))

        msg = f"🔹 {label}:\n{answer}\n\n⏱ {elapsed:.1f}с | 📊 {r.usage.total_tokens} токенов"
        for chunk in split_text(msg):
            await send(chunk)

    # Финальный вывод от Claude Opus
    await send("⏳ Claude Opus анализирует все ответы...")
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
                    + "\n\n".join(f"--- {lbl} ---\n{ans}" for lbl, ans in answers)
                )
            }
        ]
    )
    conclusion = analysis.choices[0].message.content
    msg = f"🏆 ВЫВОД от Claude Opus:\n\n{conclusion}"
    for chunk in split_text(msg):
        await send(chunk)


# ─────────────────────────── ЗАДАНИЕ 12: ПЕРСОНАЛИЗАЦИЯ ─────────────

def z12_chat_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("👤 Мой профиль", callback_data="t12_profile"),
         InlineKeyboardButton("✏️ Сменить профиль", callback_data="t12_switch")],
        [InlineKeyboardButton("🎭 Авто-демо (3 профиля)", callback_data="t12_demo")],
        [InlineKeyboardButton("🗑 Сбросить сессию", callback_data="t12_reset"),
         InlineKeyboardButton("🚪 Выйти", callback_data="t12_exit")],
    ])


def z12_preset_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("👨‍💻 Технарь", callback_data="t12_preset_техник")],
        [InlineKeyboardButton("📊 Менеджер", callback_data="t12_preset_менеджер")],
        [InlineKeyboardButton("🌱 Новичок", callback_data="t12_preset_новичок")],
        [InlineKeyboardButton("↩️ Назад", callback_data="t12_profile")],
    ])


async def start_z12_agent(user_id: int, first_message: str, send) -> None:
    z12_mode.add(user_id)
    if user_id not in z12_sessions:
        profile = z12_profile_manager.load(str(user_id))
        agent = PersonalizedAgent(
            client=deepseek,
            user_id=str(user_id),
            profile=profile,
            model="deepseek-chat",
            stm_window=10,
        )
        z12_sessions[user_id] = agent

    agent = z12_sessions[user_id]
    ltm_count = agent.ltm.count()

    intro = (
        f"🎭 *Задание 12 — Персонализация*\n\n"
        f"Активный профиль:\n{agent.profile.as_summary()}\n\n"
        f"LTM: {ltm_count} фактов о тебе\n\n"
        f"Напиши что-нибудь — или нажми «🎭 Авто-демо» чтобы увидеть\n"
        f"один вопрос через 3 разных профиля."
    )
    await send(intro, parse_mode="Markdown")

    if first_message:
        typing = await send("⏳ Думаю...")
        try:
            reply, stat = await asyncio.to_thread(agent.chat, first_message)
            if hasattr(typing, "delete"):
                await typing.delete()
            for chunk in split_text(reply):
                await send(chunk)
            await send(
                f"📊 Профиль: {agent.profile.style}/{agent.profile.expertise} | "
                f"Токены: {stat.prompt_tokens}+{stat.completion_tokens}",
                reply_markup=z12_chat_keyboard(),
            )
        except Exception as e:
            await send(f"❌ Ошибка: {e}")


async def run_z12_auto_demo(user_id: int, send) -> None:
    """Отправляет DEMO_QUESTION через 3 разных профиля, показывает разницу."""
    presets = ["техник", "менеджер", "новичок"]
    labels = {"техник": "👨‍💻 Технарь", "менеджер": "📊 Менеджер", "новичок": "🌱 Новичок"}

    await send(
        f"🎭 *Авто-демо: один вопрос — три профиля*\n\n"
        f"Вопрос: _{DEMO_QUESTION}_\n\n"
        f"Смотрим как отвечают разные профили...",
        parse_mode="Markdown",
    )

    results = []
    for preset_key in presets:
        profile = UserProfile.from_preset(f"demo_{preset_key}", preset_key)
        agent = PersonalizedAgent(
            client=deepseek,
            user_id=f"demo_{preset_key}",
            profile=profile,
            model="deepseek-chat",
            stm_window=4,
            demo_mode=True,  # без LTM и факт-экстракции — в 2x быстрее
        )

        label = labels[preset_key]
        await send(f"⏳ Генерирую ответ для {label}...")
        reply, stat = await asyncio.to_thread(agent.chat, DEMO_QUESTION)
        results.append((label, profile, reply, stat))

    # Показываем результаты
    for label, profile, reply, stat in results:
        header = (
            f"─────────────────────────\n"
            f"{label}\n"
            f"Стиль: {profile.style} | Формат: {profile.format} | Уровень: {profile.expertise}\n"
            f"─────────────────────────"
        )
        await send(header)
        for chunk in split_text(reply):
            await send(chunk)
        await send(f"📊 {stat.prompt_tokens}+{stat.completion_tokens} токенов")

    # ── Сравнительные выводы ──────────────────────────────────────────────────
    lengths = [(label, len(reply.split())) for label, _, reply, _ in results]
    tokens_list = [(label, stat.prompt_tokens + stat.completion_tokens) for label, _, _, stat in results]

    longest = max(lengths, key=lambda x: x[1])
    shortest = min(lengths, key=lambda x: x[1])

    comparison = (
        "📊 *Сравнение ответов*\n\n"
        "*Длина (слов):*\n"
    )
    for label, wc in lengths:
        bar = "█" * (wc // 50) + f" {wc}"
        comparison += f"  {label}: {bar}\n"

    comparison += "\n*Токены (запрос+ответ):*\n"
    for label, tok in tokens_list:
        comparison += f"  {label}: {tok} токенов\n"

    comparison += (
        f"\n*Выводы:*\n"
        f"• Самый подробный: {longest[0]} ({longest[1]} слов)\n"
        f"• Самый лаконичный: {shortest[0]} ({shortest[1]} слов)\n"
        f"• Разница в длине: ×{round(longest[1]/max(shortest[1],1), 1)}\n\n"
        f"*Что изменил профиль:*\n"
        f"• 👨‍💻 Технарь — глубокие технические детали, термины, примеры кода\n"
        f"• 📊 Менеджер — маркированный список, бизнес-фокус, без технических деталей\n"
        f"• 🌱 Новичок — простые аналогии, дружелюбный тон, без жаргона\n\n"
        f"Один промпт → три разных адаптации под аудиторию."
    )

    await send(comparison, parse_mode="Markdown")
    await send(
        "✅ *Демо завершено*\n\nПопробуй сам — выбери профиль и задай свой вопрос.",
        parse_mode="Markdown",
        reply_markup=z12_chat_keyboard(),
    )


# ─────────────────────────── ЗАДАНИЕ 13: TASK STATE MACHINE ─────────

def z13_entry_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура при входе — до того как задача введена."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎬 Авто-демо (готовая задача)", callback_data="t13_demo")],
        [InlineKeyboardButton("🚪 Выйти", callback_data="t13_exit")],
    ])


def z13_task_keyboard(state: TaskState) -> InlineKeyboardMarkup:
    """Клавиатура зависит от текущего состояния автомата."""
    if state == TaskState.PLANNING:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Выполнить по плану", callback_data="t13_advance")],
            [InlineKeyboardButton("⏸️ Пауза", callback_data="t13_pause"),
             InlineKeyboardButton("🎬 Демо", callback_data="t13_demo")],
            [InlineKeyboardButton("📝 Новая задача", callback_data="t13_new"),
             InlineKeyboardButton("🚪 Выйти", callback_data="t13_exit")],
        ])
    elif state == TaskState.EXECUTION:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🔍 Проверить результат", callback_data="t13_advance")],
            [InlineKeyboardButton("🔁 Повторить выполнение", callback_data="t13_retry")],
            [InlineKeyboardButton("⏸️ Пауза", callback_data="t13_pause")],
            [InlineKeyboardButton("📝 Новая задача", callback_data="t13_new"),
             InlineKeyboardButton("🚪 Выйти", callback_data="t13_exit")],
        ])
    elif state == TaskState.VALIDATION:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Завершить задачу", callback_data="t13_advance")],
            [InlineKeyboardButton("🔁 Повторить выполнение", callback_data="t13_retry")],
            [InlineKeyboardButton("⏸️ Пауза", callback_data="t13_pause")],
            [InlineKeyboardButton("📝 Новая задача", callback_data="t13_new"),
             InlineKeyboardButton("🚪 Выйти", callback_data="t13_exit")],
        ])
    elif state == TaskState.PAUSED:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("▶️ Продолжить", callback_data="t13_resume")],
            [InlineKeyboardButton("📝 Новая задача", callback_data="t13_new"),
             InlineKeyboardButton("🚪 Выйти", callback_data="t13_exit")],
        ])
    else:  # DONE или FAILED
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📝 Новая задача", callback_data="t13_new")],
            [InlineKeyboardButton("🚪 Выйти", callback_data="t13_exit")],
        ])


async def start_z13_task(user_id: int, task_text: str, send) -> None:
    """Создать новую задачу и запустить этап планирования."""
    fsm = get_z13_fsm()
    ctx = fsm.create_task(str(user_id), task_text)
    z13_sessions[user_id] = ctx

    # Этап 1: Планирование
    intro = fsm.get_stage_intro(ctx)
    await send(intro, parse_mode="Markdown")

    try:
        plan = await asyncio.to_thread(fsm.run_current_stage, ctx)
        for chunk in split_text(plan):
            await send(chunk)
        await send(
            ctx.summary(),
            parse_mode="Markdown",
            reply_markup=z13_task_keyboard(ctx.state()),
        )
    except Exception as e:
        logger.error(f"Ошибка z13 планирование: {e}")
        await send(f"❌ Ошибка на этапе планирования: {e}")


async def run_z13_auto_demo(user_id: int, send) -> None:
    """Прогнать DEMO_TASK через все три этапа автоматически."""
    fsm = get_z13_fsm()
    ctx = fsm.create_task(f"demo_{user_id}", DEMO_TASK)
    z13_sessions[user_id] = ctx

    await send(
        f"🔄 *Авто-демо Task State Machine*\n\n"
        f"Задача: _{DEMO_TASK}_\n\n"
        f"Прогоняем три этапа автоматически...",
        parse_mode="Markdown",
    )

    stages = [TaskState.PLANNING, TaskState.EXECUTION, TaskState.VALIDATION]
    stage_names = {
        TaskState.PLANNING: "📋 Этап 1/3 — Планирование",
        TaskState.EXECUTION: "⚙️ Этап 2/3 — Выполнение",
        TaskState.VALIDATION: "🔍 Этап 3/3 — Валидация",
    }

    for stage in stages:
        await send(f"*{stage_names[stage]}*", parse_mode="Markdown")
        result = await asyncio.to_thread(fsm.run_current_stage, ctx)
        for chunk in split_text(result):
            await send(chunk)
        if stage != TaskState.VALIDATION:
            fsm.advance(ctx)

    # Итог
    passed = "✅ ПРИНЯТО" if ctx.validation_passed else "⚠️ ТРЕБУЕТ ДОРАБОТКИ"
    await send(
        f"🏁 *Демо завершено*\n\n"
        f"Валидация: {passed}\n\n"
        f"{ctx.summary()}\n\n"
        f"Попробуй сам — введи свою задачу!",
        parse_mode="Markdown",
        reply_markup=z13_task_keyboard(TaskState.DONE),
    )


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
    app.add_handler(CommandHandler("newchat", newchat_command))
    app.add_handler(CommandHandler("restart", restart_command))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print("🤖 Бот запущен!")
    print("💬 Отвечу на твой вопрос за миска риса!")
    print("Нажмите Ctrl+C для остановки.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
