"""
Day 11: Сравнение 3 слоёв памяти ассистента

Демонстрирует ключевые различия STM / WM / LTM на конкретных сценариях.
Запуск: python -m zadanie11.compare_memory_layers
"""
from __future__ import annotations
import os, time, shutil
from dotenv import load_dotenv
from openai import OpenAI
from zadanie11.day11_memory import ShortTermMemory, WorkingMemory, LongTermMemory, MemoryAgent

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com",
)

PERSIST_DIR = "/tmp/compare_chroma"
USER_ID = "compare_demo_user"

# ─── утилиты вывода ────────────────────────────────────────────────────────────
W = shutil.get_terminal_size((80, 20)).columns

def hr(char="─"): print(char * W)
def header(title): hr("═"); print(f"  {title}"); hr("═")
def section(title): print(); hr(); print(f"  {title}"); hr()

def table_row(label, stm, wm, ltm, w=22):
    print(f"  {label:<{w}} │ {stm:<{w}} │ {wm:<{w}} │ {ltm:<{w}}")

def comparison_table():
    section("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    W22 = 22
    table_row("Характеристика", "STM", "WM", "LTM")
    hr()
    rows = [
        ("Хранение",          "in-memory list",      "in-memory dict",      "ChromaDB SQLite"),
        ("Персистентность",   "❌ теряется",          "❌ теряется",          "✅ навсегда"),
        ("Область действия",  "текущая сессия",      "текущая задача",      "между сессиями"),
        ("Поиск",             "порядковый (окно)",   "по ключу O(1)",       "семантический"),
        ("Модель эмбеддинга", "не нужна",            "не нужна",            "all-MiniLM-L6-v2"),
        ("Стоимость поиска",  "0 руб.",              "0 руб.",              "0 руб. (локально)"),
        ("Стоимость записи",  "0 руб.",              "0 руб.",              "LLM-вызов (факты)"),
        ("Лимит объёма",      "N сообщений",         "неограниченно",       "неограниченно"),
        ("Ответ на вопрос",   "только из диалога",   "только если занесли", "по смыслу запроса"),
        ("Сброс при рестарте","да",                  "да",                  "нет"),
    ]
    for r in rows:
        table_row(*r)
    hr()


# ─── Сценарий 1: STM — оконная память ─────────────────────────────────────────
def demo_stm():
    section("СЦЕНАРИЙ 1 — STM: что происходит при переполнении окна")
    stm = ShortTermMemory(window=4)

    msgs = [
        ("user",      "Меня зовут Алексей"),
        ("assistant", "Понял, Алексей!"),
        ("user",      "Я работаю в Яндексе"),
        ("assistant", "Интересно!"),
        ("user",      "Мне нравится кофе"),        # <- окно сдвигается
        ("assistant", "Хороший выбор!"),
        ("user",      "Какая у меня профессия?"),   # <- имя уже за окном
    ]

    for role, text in msgs:
        stm.add(role, text)

    print(f"\n  Всего сообщений добавлено : {len(stm._messages)}")
    print(f"  Размер окна               : 4")
    print(f"  Видно в контексте         : {len(stm.get())} сообщений\n")

    print("  Что видит модель:")
    for m in stm.get():
        emoji = "👤" if m["role"] == "user" else "🤖"
        print(f"    {emoji} [{m['role']}] {m['content']}")

    print("\n  ⚠️  Факт 'Меня зовут Алексей' выпал из окна!")
    print("  ⚠️  Модель НЕ сможет ответить на 'Какая у меня профессия?'")
    print("\n  Вывод: STM ограничен окном — старая информация теряется")


# ─── Сценарий 2: WM — контекст текущей задачи ─────────────────────────────────
def demo_wm():
    section("СЦЕНАРИЙ 2 — WM: отслеживание текущей задачи")
    wm = WorkingMemory()

    print("\n  Пользователь начал работу над задачей...")
    wm.set("task",    "рефакторинг модуля авторизации")
    wm.set("file",    "auth/views.py")
    wm.set("status",  "в процессе")
    wm.set("blocker", "нужен review от Дмитрия")

    print("  Состояние WM:")
    print(wm.as_context_str())

    print("\n  Модель видит контекст задачи в каждом системном промпте.")
    print("  При следующем сообщении — она знает, над чем работает пользователь.\n")

    wm.set("status", "завершено")
    print("  После обновления (task done):")
    print(wm.as_context_str())

    print("\n  ⚠️  После reset_session() — WM очищается!")
    wm.clear()
    print(f"  WM после clear: {wm.all()}")
    print("\n  Вывод: WM идеален для multi-turn задач, но не переживает рестарт")


# ─── Сценарий 3: LTM — кросс-сессионная память ────────────────────────────────
def demo_ltm():
    section("СЦЕНАРИЙ 3 — LTM: память сохраняется между сессиями")

    # Очистить старые данные
    ltm1 = LongTermMemory(user_id=USER_ID, persist_dir=PERSIST_DIR)
    ltm1.clear()

    print("\n  [СЕССИЯ 1] Пользователь рассказывает о себе...")
    facts = [
        "Пользователя зовут Мария",
        "Мария — дата-аналитик в Сбере",
        "Мария живёт в Санкт-Петербурге",
        "У Марии есть собака — лабрадор Фокс",
    ]
    for f in facts:
        ltm1.save_fact(f)
        print(f"  💾 сохранено: {f}")

    print(f"\n  Итого в LTM: {ltm1.count()} фактов")
    print("  [СЕССИЯ 1 завершена. Бот перезапущен.]\n")

    time.sleep(0.5)

    # Новый объект — новая сессия
    print("  [СЕССИЯ 2] Новый LTM-объект с тем же user_id...")
    ltm2 = LongTermMemory(user_id=USER_ID, persist_dir=PERSIST_DIR)
    print(f"  Факты без запроса: {ltm2.count()} штук (загружено из ChromaDB)\n")

    queries = [
        ("Как меня зовут?",     "имя пользователя"),
        ("Где я работаю?",      "место работы"),
        ("Есть ли у меня пет?", "питомец"),
    ]
    for query, hint in queries:
        results = ltm2.search(query, n=2)
        print(f"  🔍 запрос: «{query}»")
        for r in results:
            print(f"     → {r}")
        print()

    print("  ✅ LTM вернул правильные факты из прошлой сессии без единого STM-сообщения!")
    print("  Вывод: LTM — единственный слой, переживающий рестарт бота")

    # Очистить за собой
    ltm2.clear()


# ─── Сценарий 4: совместная работа всех 3 слоёв ───────────────────────────────
def demo_synergy():
    section("СЦЕНАРИЙ 4 — СИНЕРГИЯ: как все 3 слоя работают вместе (реальный LLM-вызов)")

    agent = MemoryAgent(
        client=client,
        user_id=USER_ID + "_synergy",
        model="deepseek-chat",
        stm_window=6,
        persist_dir=PERSIST_DIR,
    )
    agent.forget_all()  # чистый старт

    script = [
        "Меня зовут Дмитрий, я бэкенд-разработчик.",
        "Я работаю над API для платёжного сервиса.",
        "Как меня зовут и над чем я работаю?",
    ]

    print()
    for i, msg in enumerate(script, 1):
        print(f"  👤 [{i}] {msg}")
        reply, stat = agent.chat(msg)
        print(f"  🤖 {reply}")
        print(f"     STM:{len(agent.stm.get())} сообщ │ LTM:{agent.ltm.count()} фактов │ "
              f"загружено:{stat.ltm_facts_loaded} │ сохранено:{stat.ltm_facts_saved} │ "
              f"{stat.prompt_tokens}+{stat.completion_tokens} токенов")
        print()

    agent.forget_all()


# ─── main ──────────────────────────────────────────────────────────────────────
def main():
    header("ДЕНЬ 11 — СРАВНЕНИЕ 3 СЛОЁВ ПАМЯТИ АССИСТЕНТА")
    print("  STM (ShortTermMemory)  — окно диалога, in-memory")
    print("  WM  (WorkingMemory)    — контекст задачи, in-memory")
    print("  LTM (LongTermMemory)   — факты о пользователе, ChromaDB SQLite")

    comparison_table()
    demo_stm()
    demo_wm()
    demo_ltm()
    demo_synergy()

    header("ИТОГ")
    print("  │ Слой │ Когда использовать                                    │")
    hr()
    print("  │ STM  │ Текущий диалог — что сказали последние N сообщений   │")
    print("  │ WM   │ Состояние задачи — файл, статус, зависимости         │")
    print("  │ LTM  │ Кто пользователь — имя, профессия, предпочтения      │")
    hr()
    print("  Все три слоя дополняют друг друга — каждый отвечает за свою зону памяти.")
    print()


if __name__ == "__main__":
    main()
