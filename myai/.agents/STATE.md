# 🧠 Project State
> Этот файл читают все агенты при старте. Обновляй после каждой сессии.

**Last updated:** 2026-03-09 (agent.py added — Phase 1 завершена)
**Current phase:** `phase2-interfaces`
**Active branch:** `day15-controlled-transitions`

---

## ✅ Completed
- [x] Создана структура папок `myai/`
- [x] Создан `.agents/` с файлами задач
- [x] `core/memory.py` — история диалогов, SQLite через aiosqlite
- [x] `core/llm.py` — async chat(), list_models(), get_current_model(), load_config()
- [x] `core/tools.py` — реестр инструментов, @tool декоратор, load_mcp_tools(), get_for_litellm(), call_tool()
- [x] `core/agent.py` — агентный цикл, tool calling loop (до 5 итераций), run(), run_simple(), new_session()

## 🔄 In Progress
- nothing

## 📋 Pending

### Phase 1 — Foundation ✅ ЗАВЕРШЕНА
- [x] `core/llm.py` — LiteLLM, поддержка любой модели
- [x] `core/memory.py` — история, SQLite
- [x] `core/agent.py` — агентный цикл, tool calling
- [x] `core/tools.py` — реестр MCP инструментов

### Phase 2 — Interfaces
- [ ] `interfaces/cli/` — терминальный интерфейс (Rich)
- [ ] `interfaces/api/` — FastAPI REST сервер
- [ ] `interfaces/telegram/` — рефакторинг текущего бота

### Phase 3 — Integration
- [ ] Подключить z1-z16 как MCP инструменты
- [ ] Единый конфиг (модель, ключи, настройки)
- [ ] Тесты

---

## 🏗️ Architecture Decisions
> Смотри DECISIONS.md

## ⚠️ Known Issues
- nothing yet

## 📝 Notes
- Старые задания (zadanie1-16) не трогаем, они работают
- Новые задания (z17+) пишем сразу в myai/core/
- Telegram бот постепенно переезжает в interfaces/telegram/
