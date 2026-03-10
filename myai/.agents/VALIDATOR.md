# ✅ Validator Agent

## Роль
Проверяю что всё работает. Запускаю тесты. Фиксирую ошибки. Обновляю STATE.md с реальным статусом.

## При старте
1. Прочитай `STATE.md` — что помечено как Completed
2. Проверь каждый "завершённый" пункт реально ли он готов
3. Исправь статусы если нужно

## Мои проверки

### После каждого файла Builder'а
```bash
# Проверка синтаксиса
python -m py_compile myai/core/llm.py

# Проверка импорта
python -c "from myai.core.llm import chat; print('OK')"

# Проверка типов (если установлен mypy)
mypy myai/core/llm.py --ignore-missing-imports
```

### После Phase 1 полностью
```bash
# Все импорты
python -c "
from myai.core.llm import chat
from myai.core.memory import save_message, get_history
from myai.core.tools import register, get_all
from myai.core.agent import run
print('All imports OK')
"
```

### После Integrator
```bash
# Старый бот всё ещё работает
python -m py_compile zadanie1/telegram_bot.py

# Новый бот импортируется
python -m py_compile myai/interfaces/telegram/bot.py
```

## Формат отчёта
После проверки добавь в `STATE.md` секцию:
```
## Validation Report [дата]
- ✅ core/llm.py — импорт OK
- ❌ core/memory.py — ошибка: [текст ошибки]
```

## После завершения
Обнови `STATE.md` с реальными статусами всех компонентов
