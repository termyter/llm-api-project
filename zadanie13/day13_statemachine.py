"""
День 13. Task State Machine — детерминированный конечный автомат задачи.

Архитектура (из обсуждения):
┌──────────┬──────────────────────────────────────┬──────────────┐
│  Этап    │  Обогащение (Pre)                    │  LLM output  │
├──────────┼──────────────────────────────────────┼──────────────┤
│ Planning │ Запрос + "Составь план, не выполняй" │ → План       │
│ Execution│ Запрос + План + "Выполни по плану"   │ → Результат  │
│Validation│ Запрос + Результат + "Обоснуй/проверь│ → Обоснование│
│ Done     │ —                                    │ —            │
└──────────┴──────────────────────────────────────┴──────────────┘

Переходы детерминированы — не LLM решает, а код.
Пауза/продолжение — сохранение состояния в JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from openai import OpenAI

# ─── Константы ───────────────────────────────────────────────────────────────

PERSIST_DIR = Path(__file__).parent / ".tasks"
MODEL = "deepseek-chat"

DEMO_TASK = "Напиши функцию на Python для парсинга CSV файла с обработкой ошибок."

# ─── Этапы автомата ──────────────────────────────────────────────────────────

class TaskState(str, Enum):
    PLANNING   = "planning"
    EXECUTION  = "execution"
    VALIDATION = "validation"
    DONE       = "done"
    PAUSED     = "paused"
    FAILED     = "failed"

STATE_LABELS = {
    TaskState.PLANNING:   "📋 Планирование",
    TaskState.EXECUTION:  "⚙️ Выполнение",
    TaskState.VALIDATION: "🔍 Валидация",
    TaskState.DONE:       "✅ Готово",
    TaskState.PAUSED:     "⏸️ Пауза",
    TaskState.FAILED:     "❌ Ошибка",
}

# Следующий этап в цепочке
NEXT_STATE = {
    TaskState.PLANNING:   TaskState.EXECUTION,
    TaskState.EXECUTION:  TaskState.VALIDATION,
    TaskState.VALIDATION: TaskState.DONE,
}

# ─── Контекст задачи ─────────────────────────────────────────────────────────

@dataclass
class TaskContext:
    task_id: str
    user_task: str                  # Исходный запрос пользователя
    current_state: str = TaskState.PLANNING
    paused_from: Optional[str] = None   # Из какого состояния поставлена пауза

    # Выходы каждого этапа
    plan: str = ""
    execution_result: str = ""
    validation_result: str = ""
    validation_passed: bool = False

    retry_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def state(self) -> TaskState:
        return TaskState(self.current_state)

    def label(self) -> str:
        return STATE_LABELS.get(self.state(), self.current_state)

    def progress_bar(self) -> str:
        steps = [TaskState.PLANNING, TaskState.EXECUTION, TaskState.VALIDATION, TaskState.DONE]
        icons = {
            TaskState.PLANNING:   "📋",
            TaskState.EXECUTION:  "⚙️",
            TaskState.VALIDATION: "🔍",
            TaskState.DONE:       "✅",
        }
        state = self.state()
        parts = []
        for s in steps:
            if s == state:
                parts.append(f"[{icons[s]}]")
            elif steps.index(s) < steps.index(state) if state != TaskState.PAUSED else False:
                parts.append(f" {icons[s]} ")
            else:
                parts.append(f" {icons[s]} ")
        return " → ".join(p.strip() for p in parts)

    def summary(self) -> str:
        state = self.state()
        lines = [
            f"📌 Задача: {self.user_task[:80]}{'...' if len(self.user_task) > 80 else ''}",
            f"🔄 Состояние: {self.label()}",
            f"📊 Прогресс: {_progress_str(state)}",
        ]
        if self.retry_count:
            lines.append(f"🔁 Попыток выполнения: {self.retry_count}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskContext":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _progress_str(state: TaskState) -> str:
    order = [TaskState.PLANNING, TaskState.EXECUTION, TaskState.VALIDATION, TaskState.DONE]
    labels = ["📋 план", "⚙️ выполн.", "🔍 валид.", "✅ готово"]
    try:
        idx = order.index(state)
    except ValueError:
        idx = 0
    return " → ".join(
        f"**{l}**" if i == idx else l
        for i, l in enumerate(labels)
    )


# ─── Персистентность ─────────────────────────────────────────────────────────

class TaskStorage:
    def __init__(self, persist_dir: Path = PERSIST_DIR):
        self.dir = persist_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.json"

    def save(self, ctx: TaskContext) -> None:
        ctx.updated_at = datetime.now().isoformat()
        with open(self._path(ctx.task_id), "w", encoding="utf-8") as f:
            json.dump(ctx.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, task_id: str) -> Optional[TaskContext]:
        p = self._path(task_id)
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            return TaskContext.from_dict(json.load(f))

    def delete(self, task_id: str) -> None:
        p = self._path(task_id)
        if p.exists():
            p.unlink()


# ─── Промпты этапов ──────────────────────────────────────────────────────────

SYSTEM_PLANNING = """\
Ты — опытный планировщик задач. Твоя ЕДИНСТВЕННАЯ роль — составить структурированный план.
НЕ выполняй задачу, НЕ пиши код, НЕ давай готовых решений.
Только план: пронумерованный список шагов с кратким описанием каждого.
Формат ответа — Markdown с заголовками и подпунктами."""

SYSTEM_EXECUTION = """\
Ты — опытный исполнитель. Ты получаешь задачу и готовый план.
Выполни план шаг за шагом. Для каждого шага укажи:
1. Номер шага и его название
2. Конкретный результат выполнения (код, текст, данные)
Работай строго по плану, не отступай от него."""

SYSTEM_VALIDATION = """\
Ты — строгий валидатор результатов. Ты получаешь исходную задачу и результат выполнения.
Твоя роль — проверить и обосновать:
1. Соответствие результата исходному запросу
2. Полноту выполнения каждого шага плана
3. Качество и корректность решения
4. Что можно улучшить (если есть)
В конце явно укажи: ИТОГ: ПРИНЯТО или ИТОГ: ТРЕБУЕТ ДОРАБОТКИ"""


# ─── Основной класс автомата ─────────────────────────────────────────────────

class TaskStateMachine:
    """
    Детерминированный конечный автомат задачи.
    Переходы: planning → execution → validation → done
    Каждый этап = отдельный LLM-вызов с обогащённым промптом.
    """

    def __init__(self, client: OpenAI, model: str = MODEL):
        self.client = client
        self.model = model
        self.storage = TaskStorage()

    def create_task(self, user_id: str, user_task: str) -> TaskContext:
        """Создать новую задачу."""
        task_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ctx = TaskContext(
            task_id=task_id,
            user_task=user_task,
            current_state=TaskState.PLANNING,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        self.storage.save(ctx)
        return ctx

    def run_current_stage(self, ctx: TaskContext) -> str:
        """
        Запустить текущий этап. Возвращает вывод LLM.
        Обогащение промпта зависит от состояния.
        """
        state = ctx.state()

        if state == TaskState.PLANNING:
            return self._run_planning(ctx)
        elif state == TaskState.EXECUTION:
            return self._run_execution(ctx)
        elif state == TaskState.VALIDATION:
            return self._run_validation(ctx)
        else:
            return ""

    def _run_planning(self, ctx: TaskContext) -> str:
        """Этап 1: Запрос + "Составь план, не выполняй" → План"""
        messages = [
            {"role": "system", "content": SYSTEM_PLANNING},
            {"role": "user", "content": (
                f"Задача:\n{ctx.user_task}\n\n"
                f"Составь детальный план выполнения. НЕ выполняй — только план."
            )},
        ]
        result = self._call_llm(messages)
        ctx.plan = result
        self.storage.save(ctx)
        return result

    def _run_execution(self, ctx: TaskContext) -> str:
        """Этап 2: Запрос + План + "Выполни по плану" → Результат"""
        messages = [
            {"role": "system", "content": SYSTEM_EXECUTION},
            {"role": "user", "content": (
                f"Исходная задача:\n{ctx.user_task}\n\n"
                f"Готовый план:\n{ctx.plan}\n\n"
                f"Выполни задачу строго по этому плану, шаг за шагом."
            )},
        ]
        result = self._call_llm(messages)
        ctx.execution_result = result
        ctx.retry_count += 1
        self.storage.save(ctx)
        return result

    def _run_validation(self, ctx: TaskContext) -> str:
        """Этап 3: Запрос + Результат + "Обоснуй и проверь" → Обоснование"""
        messages = [
            {"role": "system", "content": SYSTEM_VALIDATION},
            {"role": "user", "content": (
                f"Исходная задача:\n{ctx.user_task}\n\n"
                f"Результат выполнения:\n{ctx.execution_result}\n\n"
                f"Проверь и обоснуй результат. В конце укажи: "
                f"ИТОГ: ПРИНЯТО или ИТОГ: ТРЕБУЕТ ДОРАБОТКИ"
            )},
        ]
        result = self._call_llm(messages)
        ctx.validation_result = result
        ctx.validation_passed = "ИТОГ: ПРИНЯТО" in result
        self.storage.save(ctx)
        return result

    def _call_llm(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""

    # ── Переходы между состояниями ───────────────────────────────────────────

    def advance(self, ctx: TaskContext) -> TaskState:
        """Перейти к следующему этапу."""
        state = ctx.state()
        next_state = NEXT_STATE.get(state)
        if next_state:
            ctx.current_state = next_state
            self.storage.save(ctx)
        return ctx.state()

    def retry_execution(self, ctx: TaskContext) -> None:
        """Вернуть на этап выполнения (валидация не прошла)."""
        ctx.current_state = TaskState.EXECUTION
        self.storage.save(ctx)

    def pause(self, ctx: TaskContext) -> None:
        """Поставить задачу на паузу."""
        if ctx.state() != TaskState.PAUSED:
            ctx.paused_from = ctx.current_state
            ctx.current_state = TaskState.PAUSED
            self.storage.save(ctx)

    def resume(self, ctx: TaskContext) -> TaskState:
        """Снять с паузы — вернуться к тому этапу, где остановились."""
        if ctx.state() == TaskState.PAUSED and ctx.paused_from:
            ctx.current_state = ctx.paused_from
            ctx.paused_from = None
            self.storage.save(ctx)
        return ctx.state()

    def finish(self, ctx: TaskContext) -> None:
        """Завершить задачу."""
        ctx.current_state = TaskState.DONE
        self.storage.save(ctx)

    # ── Утилиты ──────────────────────────────────────────────────────────────

    def get_stage_intro(self, ctx: TaskContext) -> str:
        """Вводное сообщение перед запуском этапа."""
        state = ctx.state()
        intros = {
            TaskState.PLANNING: (
                "📋 *Этап 1/3 — Планирование*\n\n"
                "LLM получает задачу и инструкцию «составь план, не выполняй».\n"
                "Ждём структурированный план..."
            ),
            TaskState.EXECUTION: (
                "⚙️ *Этап 2/3 — Выполнение*\n\n"
                "LLM получает задачу + утверждённый план.\n"
                "Выполняем шаг за шагом..."
            ),
            TaskState.VALIDATION: (
                "🔍 *Этап 3/3 — Валидация*\n\n"
                "LLM получает задачу + результат выполнения.\n"
                "Проверяем качество и полноту..."
            ),
        }
        return intros.get(state, "")
