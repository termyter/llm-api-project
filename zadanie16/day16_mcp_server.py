"""
День 16 — MCP Server
====================
Минимальный MCP-сервер на FastMCP.
Запускается как subprocess клиентом через stdio transport.

Инструменты демонстрируют концепцию: агент читает описание
и понимает, как использовать инструмент — без отдельного CLI.
"""

from mcp.server.fastmcp import FastMCP

# ── Создаём сервер ────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="LLM Course Tools",
    instructions=(
        "Сервер инструментов учебного курса по LLM-агентам. "
        "Предоставляет утилиты для работы с состояниями задач, "
        "HTTP API и мониторинга прогресса обучения."
    ),
)


# ── Tool 1: Проверка перехода состояний (тема z13–z15) ───────────────────────
@mcp.tool()
def check_transition(from_state: str, to_state: str) -> dict:
    """
    Проверяет, допустим ли переход между состояниями задачи (FSM).

    Допустимые переходы:
      INIT → PLANNING → EXECUTING → REVIEW → DONE
      REVIEW → EXECUTING  (возврат на доработку)
      любое → CANCELLED

    Возвращает {"allowed": bool, "reason": str}.

    Используй перед любым изменением статуса задачи,
    чтобы не нарушить бизнес-логику автомата состояний.
    """
    ALLOWED = {
        ("INIT", "PLANNING"),
        ("PLANNING", "EXECUTING"),
        ("EXECUTING", "REVIEW"),
        ("REVIEW", "EXECUTING"),
        ("REVIEW", "DONE"),
    }
    to_upper = to_state.upper()
    pair = (from_state.upper(), to_upper)
    if to_upper == "CANCELLED":
        return {"allowed": True, "reason": "CANCELLED доступен из любого состояния"}
    if pair in ALLOWED:
        return {"allowed": True, "reason": f"Переход {from_state} → {to_state} разрешён"}
    return {
        "allowed": False,
        "reason": (
            f"Переход {from_state} → {to_state} запрещён. "
            f"Допустимые переходы из {from_state.upper()}: "
            + ", ".join(t for f, t in ALLOWED if f == from_state.upper())
            or "нет"
        ),
    }


# ── Tool 2: HTTP GET через описание (концепция из чата Максима) ──────────────
@mcp.tool()
def fetch_url(url: str, timeout_sec: int = 10) -> dict:
    """
    Выполняет HTTP GET запрос к указанному URL и возвращает статус и тело ответа.

    Эквивалент: curl -s -o body.txt -w "%{http_code}" <url> --max-time <timeout_sec>

    Параметры:
      url         — полный URL с протоколом (https://...)
      timeout_sec — таймаут в секундах (по умолчанию 10)

    Возвращает {"status": int, "body": str, "error": str | null}.

    Используй для проверки доступности API, получения JSON-данных
    или мониторинга endpoint'ов без написания отдельного CLI.
    """
    import urllib.request
    import urllib.error

    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")[:2000]
            return {"status": resp.status, "body": body, "error": None}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": "", "error": str(e)}
    except Exception as e:
        return {"status": 0, "body": "", "error": str(e)}


# ── Tool 3: Прогресс курса ────────────────────────────────────────────────────
@mcp.tool()
def get_course_progress(current_day: int) -> dict:
    """
    Возвращает прогресс прохождения учебного курса по LLM-агентам.

    Параметры:
      current_day — номер текущего дня (1–30)

    Возвращает:
      percent       — процент завершения
      completed     — список пройденных тем
      next_topic    — следующая тема
      days_left     — осталось дней

    Используй для отображения дашборда прогресса студента.
    """
    topics = {
        1: "Базовый LLM-вызов",
        2: "Стриминг",
        3: "История диалога",
        4: "Системный промпт",
        5: "Параметры генерации",
        6: "Структурированный вывод",
        7: "Инструменты (tool use)",
        8: "Параллельные вызовы",
        9: "RAG (базовый)",
        10: "Семантический поиск",
        11: "Память ассистента",
        12: "Персонализация",
        13: "Task State Machine",
        14: "FSM + инварианты",
        15: "Контролируемые переходы",
        16: "MCP подключение",
        17: "MCP инструменты",
        18: "Агент с памятью",
        19: "Мультиагент",
        20: "Финальный проект",
    }
    total = 20
    percent = round(min(current_day, total) / total * 100)
    completed = [topics[d] for d in range(1, min(current_day + 1, total + 1)) if d in topics]
    next_day = current_day + 1
    next_topic = topics.get(next_day, "Курс завершён!")
    return {
        "percent": percent,
        "completed": completed,
        "next_topic": next_topic,
        "days_left": max(0, total - current_day),
    }


# ── Tool 4: Список модулей курса ──────────────────────────────────────────────
@mcp.tool()
def list_course_modules() -> list[dict]:
    """
    Возвращает полный список модулей учебного курса с их статусами.

    Каждый модуль содержит:
      day       — номер дня
      title     — название темы
      category  — категория (basics / fsm / agents / mcp)
      done      — реализован ли в боте

    Используй для навигации по курсу или построения оглавления.
    """
    return [
        {"day": 1,  "title": "Базовый LLM-вызов",          "category": "basics", "done": True},
        {"day": 2,  "title": "Стриминг ответа",             "category": "basics", "done": True},
        {"day": 3,  "title": "История диалога",             "category": "basics", "done": True},
        {"day": 13, "title": "Task State Machine",          "category": "fsm",    "done": True},
        {"day": 14, "title": "FSM + Инварианты",            "category": "fsm",    "done": True},
        {"day": 15, "title": "Контролируемые переходы",     "category": "fsm",    "done": True},
        {"day": 16, "title": "MCP подключение",             "category": "mcp",    "done": True},
        {"day": 17, "title": "MCP инструменты",             "category": "mcp",    "done": False},
        {"day": 18, "title": "Агент с памятью",             "category": "agents", "done": False},
        {"day": 19, "title": "Мультиагент",                 "category": "agents", "done": False},
    ]


# ── Tool 5: Ping (проверка соединения) ────────────────────────────────────────
@mcp.tool()
def ping(message: str = "hello") -> str:
    """
    Эхо-инструмент для проверки работоспособности MCP-сервера.

    Параметры:
      message — любая строка (по умолчанию "hello")

    Возвращает строку вида "pong: <message>".

    Используй первым при диагностике: если ping работает,
    сервер запущен и соединение установлено корректно.
    """
    return f"pong: {message}"


# ── Запуск ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # stdio transport — стандарт для локального MCP
    mcp.run(transport="stdio")
