# 🧠 Project State
> Этот файл читают все агенты при старте. Обновляй после каждой сессии.

**Last updated:** 2026-03-09
**Current phase:** `foundation`
**Active branch:** `day15-controlled-transitions`

---

## ✅ Completed
- [x] Создана структура папок `myai/`
- [x] Создан `.agents/` с файлами задач

## 🔄 In Progress
- nothing

## 📋 Pending

### Phase 1 — Foundation
- [ ] `core/llm.py` — LiteLLM, поддержка любой модели
- [ ] `core/memory.py` — история, SQLite
- [ ] `core/agent.py` — агентный цикл, tool calling
- [ ] `core/tools.py` — реестр MCP инструментов

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
