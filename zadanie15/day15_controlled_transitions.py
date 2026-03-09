"""
День 15. Контролируемые переходы состояний.

Архитектура:
- Явный allowlist переходов: ALLOWED_TRANSITIONS — множество пар (from, to)
- TransitionError — исключение при попытке нелегального перехода
- ControlledStateMachine наследует InvariantStateMachine (z14)
  и добавляет метод assert_transition(from, to) перед каждым переходом
- Демо: попытка пропустить этап → чёткий отказ с объяснением

Гарантии:
  ✅ Нельзя перейти к выполнению без утверждённого плана
  ✅ Нельзя завершить без валидации
  ✅ Нельзя пропустить этапы в любом порядке
  ✅ Пауза/возобновление работают из любого активного состояния
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from zadanie13.day13_statemachine import TaskState, TaskContext, TaskStorage, MODEL
from zadanie14.day14_invariants import (
    InvariantStateMachine, InvariantTaskContext, InvariantStorage,
    Invariant, _InvariantTaskStorage,
)

# ─── Константы ───────────────────────────────────────────────────────────────

PERSIST_DIR_Z15 = Path(__file__).parent / ".tasks"

# ─── Разрешённые переходы ─────────────────────────────────────────────────────
#
# Явный allowlist пар (from_state, to_state).
# Любой переход, не входящий в этот список → TransitionError.

ALLOWED_TRANSITIONS: set[tuple[str, str]] = {
    # Нормальный поток
    (TaskState.PLANNING,   TaskState.EXECUTION),
    (TaskState.EXECUTION,  TaskState.VALIDATION),
    (TaskState.VALIDATION, TaskState.DONE),
    # Повторная генерация (откат)
    (TaskState.VALIDATION, TaskState.EXECUTION),   # redo execution after failed validation
    # Пауза из любого активного состояния
    (TaskState.PLANNING,   TaskState.PAUSED),
    (TaskState.EXECUTION,  TaskState.PAUSED),
    (TaskState.VALIDATION, TaskState.PAUSED),
    # Возобновление из паузы в любое активное состояние
    (TaskState.PAUSED,     TaskState.PLANNING),
    (TaskState.PAUSED,     TaskState.EXECUTION),
    (TaskState.PAUSED,     TaskState.VALIDATION),
    # Аварийный останов
    (TaskState.PLANNING,   TaskState.FAILED),
    (TaskState.EXECUTION,  TaskState.FAILED),
    (TaskState.VALIDATION, TaskState.FAILED),
}

# Человекочитаемые объяснения, почему переход заблокирован
BLOCKED_REASON: dict[tuple[str, str], str] = {
    (TaskState.PLANNING,   TaskState.VALIDATION): (
        "❌ Нельзя перейти к валидации без выполнения.\n"
        "Сначала утвердите план (PLANNING) и выполните задачу (EXECUTION)."
    ),
    (TaskState.PLANNING,   TaskState.DONE): (
        "❌ Нельзя завершить задачу без выполнения и валидации.\n"
        "Пройдите: PLANNING → EXECUTION → VALIDATION → DONE."
    ),
    (TaskState.EXECUTION,  TaskState.DONE): (
        "❌ Нельзя завершить задачу без валидации.\n"
        "Сначала проверьте результат на этапе VALIDATION."
    ),
    (TaskState.EXECUTION,  TaskState.PLANNING): (
        "❌ Нельзя вернуться к планированию после начала выполнения.\n"
        "Для повторной попытки используйте 'retry' (перегенерация выполнения)."
    ),
    (TaskState.VALIDATION, TaskState.PLANNING): (
        "❌ Нельзя вернуться к планированию с этапа валидации.\n"
        "Если результат не прошёл — используйте 'retry' (повторить выполнение)."
    ),
    (TaskState.DONE,       TaskState.PLANNING):   "❌ Задача завершена. Создайте новую задачу.",
    (TaskState.DONE,       TaskState.EXECUTION):  "❌ Задача завершена. Создайте новую задачу.",
    (TaskState.DONE,       TaskState.VALIDATION): "❌ Задача завершена. Создайте новую задачу.",
    (TaskState.FAILED,     TaskState.PLANNING):   "❌ Задача в состоянии FAILED. Создайте новую задачу.",
    (TaskState.FAILED,     TaskState.EXECUTION):  "❌ Задача в состоянии FAILED. Создайте новую задачу.",
}


# ─── Исключение ──────────────────────────────────────────────────────────────

class TransitionError(Exception):
    """Попытка незаконного перехода между состояниями FSM."""

    def __init__(self, from_state: str, to_state: str, reason: str = ""):
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        super().__init__(
            reason or (
                f"❌ Переход {from_state.upper()} → {to_state.upper()} запрещён.\n"
                f"Допустимые переходы из {from_state.upper()}: "
                + ", ".join(
                    t.upper() for f, t in ALLOWED_TRANSITIONS if f == from_state
                )
            )
        )


# ─── Утилита проверки ─────────────────────────────────────────────────────────

def transition_allowed(from_state: str, to_state: str) -> bool:
    """Вернуть True если переход разрешён."""
    return (from_state, to_state) in ALLOWED_TRANSITIONS


def assert_transition(from_state: str, to_state: str) -> None:
    """Вызвать TransitionError если переход не разрешён."""
    if not transition_allowed(from_state, to_state):
        reason = BLOCKED_REASON.get(
            (from_state, to_state),
            (
                f"❌ Переход {from_state.upper()} → {to_state.upper()} запрещён.\n"
                f"Допустимые переходы из {from_state.upper()}: "
                + ", ".join(
                    t.upper() for f, t in sorted(ALLOWED_TRANSITIONS) if f == from_state
                )
            ),
        )
        raise TransitionError(from_state, to_state, reason)


# ─── FSM с контролируемыми переходами ────────────────────────────────────────

class ControlledStateMachine(InvariantStateMachine):
    """
    z14 InvariantStateMachine + жёсткий контроль переходов через allowlist.

    Каждый вызов advance/retry/pause/resume сначала проверяет
    ALLOWED_TRANSITIONS и кидает TransitionError при нарушении.
    """

    def __init__(self, client: OpenAI, model: str = MODEL, max_retries: int = 3):
        super().__init__(client, model, max_retries)
        self.storage = _ControlledTaskStorage()

    def create_task(self, user_id: str, user_task: str) -> ControlledTaskContext:
        task_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ctx = ControlledTaskContext(
            task_id=task_id,
            user_task=user_task,
            current_state=TaskState.PLANNING,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        self.storage.save(ctx)
        return ctx

    def advance(self, ctx: "ControlledTaskContext") -> TaskState:
        """Перейти к следующему этапу — с проверкой перехода."""
        from zadanie13.day13_statemachine import NEXT_STATE
        current = ctx.state()
        next_s = NEXT_STATE.get(current)
        if next_s is None:
            raise TransitionError(current, "???", f"❌ Нет следующего состояния из {current.upper()}.")
        assert_transition(current, next_s)
        ctx.current_state = next_s
        self.storage.save(ctx)
        ctx.record_transition(current, next_s, success=True)
        return ctx.state()

    def retry_execution(self, ctx: "ControlledTaskContext") -> None:
        """Повторить выполнение (откат валидация → выполнение)."""
        current = ctx.state()
        assert_transition(current, TaskState.EXECUTION)
        ctx.current_state = TaskState.EXECUTION
        self.storage.save(ctx)
        ctx.record_transition(current, TaskState.EXECUTION, success=True)

    def pause(self, ctx: "ControlledTaskContext") -> None:
        """Поставить задачу на паузу — с проверкой перехода."""
        current = ctx.state()
        if current == TaskState.PAUSED:
            return
        assert_transition(current, TaskState.PAUSED)
        ctx.paused_from = ctx.current_state
        ctx.current_state = TaskState.PAUSED
        self.storage.save(ctx)
        ctx.record_transition(current, TaskState.PAUSED, success=True)

    def resume(self, ctx: "ControlledTaskContext") -> TaskState:
        """Снять с паузы — с проверкой перехода."""
        if ctx.state() != TaskState.PAUSED or not ctx.paused_from:
            return ctx.state()
        target = TaskState(ctx.paused_from)
        assert_transition(TaskState.PAUSED, target)
        ctx.current_state = ctx.paused_from
        ctx.paused_from = None
        self.storage.save(ctx)
        ctx.record_transition(TaskState.PAUSED, target, success=True)
        return ctx.state()

    def try_illegal_transition(
        self,
        ctx: "ControlledTaskContext",
        target_state: TaskState,
    ) -> str:
        """
        Попытаться выполнить заведомо нелегальный переход.
        Возвращает строку с описанием ошибки (для демо-режима).
        Не изменяет состояние.
        """
        from_state = ctx.state()
        try:
            assert_transition(from_state, target_state)
            return f"⚠️ Переход {from_state.upper()} → {target_state.upper()} неожиданно разрешён."
        except TransitionError as e:
            ctx.record_transition(from_state, target_state, success=False, error=str(e))
            self.storage.save(ctx)
            return str(e)


# ─── Расширенный контекст ─────────────────────────────────────────────────────

@dataclass
class ControlledTaskContext(InvariantTaskContext):
    """InvariantTaskContext + история попыток переходов."""
    transition_log: list[dict] = field(default_factory=list)

    def record_transition(
        self,
        from_state: str,
        to_state: str,
        success: bool,
        error: str = "",
    ) -> None:
        self.transition_log.append({
            "from": str(from_state),
            "to": str(to_state),
            "success": success,
            "error": error,
            "at": datetime.now().isoformat(),
        })

    def illegal_attempts(self) -> list[dict]:
        return [t for t in self.transition_log if not t["success"]]

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["transition_log"] = self.transition_log
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ControlledTaskContext":
        from zadanie13.day13_statemachine import TaskContext
        from zadanie14.day14_invariants import InvariantTaskContext
        fields = (
            set(TaskContext.__dataclass_fields__)
            | {"invariant_violations", "stage_retry_counts", "transition_log"}
        )
        return cls(**{k: v for k, v in d.items() if k in fields})


class _ControlledTaskStorage(TaskStorage):
    """Хранилище для ControlledTaskContext."""

    def __init__(self):
        super().__init__(PERSIST_DIR_Z15)

    def save(self, ctx: TaskContext) -> None:
        ctx.updated_at = datetime.now().isoformat()
        path = self._path(ctx.task_id)
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ctx.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, task_id: str) -> Optional[ControlledTaskContext]:
        import json
        p = self._path(task_id)
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            return ControlledTaskContext.from_dict(json.load(f))


# ─── Демо-сценарии ───────────────────────────────────────────────────────────

DEMO_TASK_Z15 = (
    "Напиши Python-класс для управления очередью задач с приоритетами."
)

# Нелегальные переходы для демо-теста
ILLEGAL_DEMO_ATTEMPTS: list[tuple[TaskState, TaskState, str]] = [
    (
        TaskState.PLANNING,
        TaskState.VALIDATION,
        "Попытка пропустить EXECUTION и перейти сразу к VALIDATION",
    ),
    (
        TaskState.PLANNING,
        TaskState.DONE,
        "Попытка завершить задачу без выполнения и валидации",
    ),
    (
        TaskState.EXECUTION,
        TaskState.DONE,
        "Попытка завершить задачу без валидации",
    ),
]


def describe_allowlist() -> str:
    """Текстовое описание разрешённых переходов."""
    grouped: dict[str, list[str]] = {}
    for (f, t) in sorted(ALLOWED_TRANSITIONS):
        grouped.setdefault(f, []).append(t)
    lines = ["*Разрешённые переходы:*"]
    for from_s, targets in grouped.items():
        lines.append(f"  `{from_s.upper()}` → " + ", ".join(f"`{t.upper()}`" for t in targets))
    return "\n".join(lines)
