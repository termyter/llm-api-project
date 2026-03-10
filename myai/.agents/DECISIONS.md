# 🏗️ Architecture Decisions

## Multi-LLM
- **Библиотека:** LiteLLM
- **Причина:** единый API для Claude, GPT, Gemini, Ollama и 100+ моделей
- **Конфиг:** `myai/config.yaml` — пользователь выбирает модель

## Memory
- **Краткосрочная:** в памяти процесса (история диалога)
- **Долгосрочная:** SQLite (`myai/data/memory.db`)
- **Векторная:** ChromaDB (уже используется в проекте)

## Tools
- **Протокол:** MCP (уже реализован в z16)
- **Регистрация:** декораторы `@tool` в `core/tools.py`
- **Существующие задания:** z1-z16 подключаются как MCP tools

## Interfaces
- **Primary:** CLI (Rich library) — для разработки и тестирования
- **Mobile/Remote:** Telegram Bot — рефакторинг текущего
- **API:** FastAPI — для будущего Flutter приложения
- **Future:** Flutter (iOS, Android, Web, Desktop)

## Project Structure
```
myai/
  core/          ← бизнес-логика, не зависит от интерфейса
  interfaces/    ← UI слои (telegram, cli, api)
  data/          ← SQLite, ChromaDB
  config.yaml    ← настройки пользователя
  .agents/       ← файлы состояния агентов
```

## Naming
- Проект: `myai` (временное название, можно переименовать)
- Модуль: `from myai.core.llm import chat`
