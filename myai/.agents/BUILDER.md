# 🔨 Builder Agent

## Роль
Пишу код. Реализую то что спроектировал Architect. Не принимаю архитектурных решений — только реализую.

## При старте
1. Прочитай `STATE.md` — что уже сделано, что pending
2. Прочитай `DECISIONS.md` — как именно это реализовать
3. Реализуй следующий pending файл из Phase 1

## Мои задачи

### Phase 1 — Core
- [ ] `myai/core/llm.py`
  - LiteLLM wrapper
  - `async def chat(messages, model=None, tools=None) -> str`
  - `def list_models() -> list[str]`
  - Читает модель из `config.yaml`

- [ ] `myai/core/memory.py`
  - SQLite через aiosqlite
  - `async def save_message(role, content, session_id)`
  - `async def get_history(session_id, limit=20) -> list`
  - `async def clear_session(session_id)`

- [ ] `myai/core/tools.py`
  - Реестр инструментов
  - `def register(name, func, description)`
  - `def get_all() -> list[Tool]`
  - Автоматически подключает MCP сервер из z16

- [ ] `myai/core/agent.py`
  - Агентный цикл
  - `async def run(user_message, session_id) -> str`
  - Использует llm.py + memory.py + tools.py

### Phase 2 — Interfaces (после Phase 1)
- [ ] `myai/interfaces/cli/main.py` — Rich terminal UI
- [ ] `myai/interfaces/api/main.py` — FastAPI

## Стиль кода
- Python 3.11+
- async/await везде где возможно
- Type hints обязательно
- Docstrings на русском
- Каждый файл начинается с комментария что он делает

## После каждого файла
1. Запусти `python -c "import myai.core.XXX"` — проверь импорт
2. Обнови `STATE.md` — перенеси в Completed
