#!/usr/bin/env python3
"""
myai/test_core.py — Быстрая проверка всех компонентов Phase 1.

Запуск:
    cd /Users/ivanemelanov/Xcode/llm-api-project
    source venv/bin/activate
    python -m myai.test_core
"""

import asyncio
import sys
import os


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты вывода
# ──────────────────────────────────────────────────────────────────────────────

def ok(msg: str) -> None:
    print(f"  ✅ {msg}")

def fail(msg: str) -> None:
    print(f"  ❌ {msg}")

def section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ──────────────────────────────────────────────────────────────────────────────
# Тест 1: Импорты
# ──────────────────────────────────────────────────────────────────────────────

def test_imports() -> bool:
    section("1️⃣  Импорты")
    all_ok = True

    modules = [
        ("myai.core.llm",    ["load_config", "chat", "get_current_model", "list_models"]),
        ("myai.core.memory", ["init_db", "save_message", "get_history", "clear_session"]),
        ("myai.core.tools",  ["register", "get_all", "get_for_litellm", "call_tool"]),
        ("myai.core.agent",  ["run", "run_simple", "new_session"]),
    ]

    for module_name, attrs in modules:
        try:
            module = __import__(module_name, fromlist=attrs)
            missing = [a for a in attrs if not hasattr(module, a)]
            if missing:
                fail(f"{module_name} — нет атрибутов: {missing}")
                all_ok = False
            else:
                ok(f"{module_name} — {', '.join(attrs)}")
        except ImportError as e:
            fail(f"{module_name} — ImportError: {e}")
            all_ok = False

    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Тест 2: Config
# ──────────────────────────────────────────────────────────────────────────────

def test_config() -> bool:
    section("2️⃣  Config (config.yaml)")
    from myai.core.llm import load_config, get_current_model, list_models

    try:
        cfg = load_config()
        ok(f"load_config() вернул {len(cfg)} секций: {list(cfg.keys())}")
    except Exception as e:
        fail(f"load_config() — {e}")
        return False

    try:
        model = get_current_model()
        ok(f"get_current_model() = '{model}'")
    except Exception as e:
        fail(f"get_current_model() — {e}")
        return False

    try:
        models = list_models()
        ok(f"list_models() = {models}")
    except Exception as e:
        fail(f"list_models() — {e}")
        return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Тест 3: Memory (SQLite — без API ключа)
# ──────────────────────────────────────────────────────────────────────────────

async def test_memory() -> bool:
    section("3️⃣  Memory (SQLite)")
    from myai.core import memory

    test_session = "test_session_delete_me"

    try:
        await memory.init_db()
        ok("init_db() — БД инициализирована")
    except Exception as e:
        fail(f"init_db() — {e}")
        return False

    try:
        await memory.save_message("user", "Привет, это тест", test_session)
        await memory.save_message("assistant", "Тест прошёл успешно", test_session)
        ok("save_message() — 2 сообщения сохранены")
    except Exception as e:
        fail(f"save_message() — {e}")
        return False

    try:
        history = await memory.get_history(test_session)
        assert len(history) == 2, f"ожидал 2 сообщения, получил {len(history)}"
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        ok(f"get_history() — вернул {len(history)} сообщения, роли: {[m['role'] for m in history]}")
    except Exception as e:
        fail(f"get_history() — {e}")
        return False

    try:
        sessions = await memory.list_sessions()
        assert test_session in sessions
        ok(f"list_sessions() — нашёл тестовую сессию в {len(sessions)} сессиях")
    except Exception as e:
        fail(f"list_sessions() — {e}")
        return False

    try:
        await memory.clear_session(test_session)
        history_after = await memory.get_history(test_session)
        assert len(history_after) == 0
        ok("clear_session() — сессия очищена")
    except Exception as e:
        fail(f"clear_session() — {e}")
        return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Тест 4: Tools (без API ключа)
# ──────────────────────────────────────────────────────────────────────────────

async def test_tools() -> bool:
    section("4️⃣  Tools Registry")
    from myai.core import tools

    # Регистрируем тестовый инструмент
    async def _test_tool(text: str) -> str:
        return f"ТЕСТ: {text}"

    try:
        tools.register(
            name="test_echo",
            func=_test_tool,
            description="Тестовый инструмент — эхо",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Текст для эхо"}},
                "required": ["text"],
            }
        )
        ok("register() — инструмент 'test_echo' зарегистрирован")
    except Exception as e:
        fail(f"register() — {e}")
        return False

    try:
        all_tools = tools.get_all()
        names = [t.name for t in all_tools]
        assert "test_echo" in names
        ok(f"get_all() — {len(all_tools)} инструментов: {names}")
    except Exception as e:
        fail(f"get_all() — {e}")
        return False

    try:
        litellm_tools = tools.get_for_litellm()
        assert len(litellm_tools) > 0
        assert litellm_tools[0]["type"] == "function"
        ok(f"get_for_litellm() — {len(litellm_tools)} инструментов в формате LiteLLM")
    except Exception as e:
        fail(f"get_for_litellm() — {e}")
        return False

    try:
        result = await tools.call_tool("test_echo", {"text": "работает!"})
        assert "ТЕСТ: работает!" in result
        ok(f"call_tool() — результат: '{result}'")
    except Exception as e:
        fail(f"call_tool() — {e}")
        return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Тест 5: Agent new_session
# ──────────────────────────────────────────────────────────────────────────────

async def test_agent_session() -> bool:
    section("5️⃣  Agent — new_session()")
    from myai.core.agent import new_session

    try:
        sid = await new_session()
        assert len(sid) > 10
        assert "_" in sid
        ok(f"new_session() = '{sid}'")
    except Exception as e:
        fail(f"new_session() — {e}")
        return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Тест 6: LLM chat (нужен API ключ)
# ──────────────────────────────────────────────────────────────────────────────

async def test_llm_chat() -> bool:
    section("6️⃣  LLM chat() — требует ANTHROPIC_API_KEY")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  ⚠️  ANTHROPIC_API_KEY не задан — пропускаем LLM тест")
        print("     Чтобы протестировать: export ANTHROPIC_API_KEY=your_key")
        return True  # не считаем это ошибкой

    from myai.core.llm import chat

    try:
        print("  ⏳ Отправляем запрос к API...")
        response = await chat(
            messages=[{"role": "user", "content": "Скажи только слово: РАБОТАЕТ"}]
        )
        ok(f"chat() ответил: '{response[:100]}'")
    except Exception as e:
        fail(f"chat() — {e}")
        return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Тест 7: Full agent run_simple (нужен API ключ)
# ──────────────────────────────────────────────────────────────────────────────

async def test_agent_run() -> bool:
    section("7️⃣  Agent run_simple() — требует ANTHROPIC_API_KEY")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  ⚠️  ANTHROPIC_API_KEY не задан — пропускаем тест агента")
        return True

    from myai.core.agent import run_simple

    try:
        print("  ⏳ Запускаем агентный цикл...")
        response = await run_simple(
            user_message="Привет! Ответь одним словом: всё работает?",
            session_id="test_agent_run",
        )
        ok(f"run_simple() ответил: '{response[:150]}'")

        # Чистим тестовую сессию
        from myai.core import memory
        await memory.clear_session("test_agent_run")
    except Exception as e:
        fail(f"run_simple() — {e}")
        return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Главная функция
# ──────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n" + "═" * 50)
    print("  🧪 myai Core Tests")
    print("═" * 50)

    results = []

    # Синхронные тесты
    results.append(("Imports",  test_imports()))
    results.append(("Config",   test_config()))

    # Асинхронные тесты
    results.append(("Memory",   await test_memory()))
    results.append(("Tools",    await test_tools()))
    results.append(("Session",  await test_agent_session()))
    results.append(("LLM chat", await test_llm_chat()))
    results.append(("Agent",    await test_agent_run()))

    # Итог
    print("\n" + "═" * 50)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n  Итог: {passed}/{total} тестов прошли\n")

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status}  {name}")

    print()
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
