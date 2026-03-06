"""
День 14. Инварианты и ограничения состояния.

Архитектура:
- Инварианты хранятся отдельно от диалога (JSON-файл на пользователя)
- Отдельный агент-чекер делает бинарный вывод: YES / VIOLATION: <причина>
- FSM из Day 13 расширена: после каждого этапа → проверка инвариантов
- Нарушение → откат на тот же этап + перегенерация с объяснением нарушения
- Конфигурируемый max_retries, после превышения → эскалация пользователю

┌──────────┬──────────────┬──────────────────┬──────────────┐
│  Этап    │  LLM output  │  Invariant check │  Результат   │
├──────────┼──────────────┼──────────────────┼──────────────┤
│ Planning │  → План      │  YES / VIOLATION │  advance/retry│
│ Execution│  → Результат │  YES / VIOLATION │  advance/retry│
│Validation│  → Обосн.    │  YES / VIOLATION │  done/retry  │
└──────────┴──────────────┴──────────────────┴──────────────┘

Ключевое: инварианты — жёсткие проверки кодом, не LLM-рекомендации.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from zadanie13.day13_statemachine import (
    TaskStateMachine, TaskContext, TaskState, TaskStorage,
    SYSTEM_PLANNING, SYSTEM_EXECUTION, SYSTEM_VALIDATION, MODEL,
)

# ─── Константы ───────────────────────────────────────────────────────────────

INVARIANTS_DIR = Path(__file__).parent / ".invariants"
MAX_RETRIES_DEFAULT = 3

# ─── Инвариант ───────────────────────────────────────────────────────────────

@dataclass
class Invariant:
    name: str               # Краткое название: "no_javascript"
    description: str        # Для человека: "Не использовать JavaScript"
    rule_text: str          # Для LLM: "Решение НЕ ДОЛЖНО содержать JavaScript-код"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Invariant":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─── Хранилище инвариантов ────────────────────────────────────────────────────

class InvariantStorage:
    """Хранит инварианты пользователя в JSON-файле (отдельно от задачи)."""

    def __init__(self, user_id: str, storage_dir: Path = INVARIANTS_DIR):
        self.user_id = user_id
        self.path = storage_dir / f"{user_id}.json"
        storage_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[Invariant]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        return [Invariant.from_dict(d) for d in data]

    def save(self, invariants: list[Invariant]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([inv.to_dict() for inv in invariants], f, ensure_ascii=False, indent=2)

    def add(self, inv: Invariant) -> None:
        invariants = self.load()
        # Заменить если имя совпадает
        invariants = [i for i in invariants if i.name != inv.name]
        invariants.append(inv)
        self.save(invariants)

    def remove(self, name: str) -> bool:
        invariants = self.load()
        new = [i for i in invariants if i.name != name]
        if len(new) == len(invariants):
            return False
        self.save(new)
        return True

    def clear(self) -> None:
        self.save([])

    def list_all(self) -> list[Invariant]:
        return self.load()


# ─── Агент-чекер инвариантов ─────────────────────────────────────────────────

SYSTEM_INVARIANT_CHECKER = """\
Ты — строгий аудитор соответствия требованиям. Тебе дают:
1. Список инвариантов (жёстких ограничений), которые НЕЛЬЗЯ нарушать
2. Контент для проверки (план, результат или обоснование)

Твоя задача — проверить, нарушает ли контент хотя бы один инвариант.

ФОРМАТ ОТВЕТА — только одна из двух форм:
- YES  (если контент не нарушает ни одного инварианта)
- VIOLATION: <название инварианта>. <конкретное объяснение нарушения>

НЕ добавляй пояснений кроме указанного формата.
НЕ пиши ничего лишнего до или после ответа.
"""


class InvariantChecker:
    """
    Отдельный LLM-агент для бинарной проверки инвариантов.
    Возвращает (passed: bool, violation_reason: str | None).
    """

    def __init__(self, client: OpenAI, model: str = MODEL):
        self.client = client
        self.model = model

    def check(
        self,
        content: str,
        invariants: list[Invariant],
        stage_label: str = "содержимое",
    ) -> tuple[bool, Optional[str]]:
        """
        Возвращает (True, None) если все инварианты соблюдены.
        Возвращает (False, "VIOLATION: ...") если нарушен.
        """
        if not invariants:
            return True, None

        inv_block = "\n".join(
            f"{i+1}. [{inv.name}] {inv.rule_text}"
            for i, inv in enumerate(invariants)
        )

        messages = [
            {"role": "system", "content": SYSTEM_INVARIANT_CHECKER},
            {"role": "user", "content": (
                f"Инварианты:\n{inv_block}\n\n"
                f"Проверяемый контент ({stage_label}):\n{content}"
            )},
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=256,
            temperature=0.0,  # детерминированно
        )
        answer = (resp.choices[0].message.content or "").strip()

        if answer.upper().startswith("YES"):
            return True, None
        else:
            return False, answer  # "VIOLATION: ..."


# ─── Расширенный контекст задачи ─────────────────────────────────────────────

@dataclass
class InvariantTaskContext(TaskContext):
    """TaskContext + история нарушений инвариантов."""
    invariant_violations: list[dict] = field(default_factory=list)
    stage_retry_counts: dict = field(default_factory=dict)

    def record_violation(self, stage: str, reason: str) -> None:
        self.invariant_violations.append({
            "stage": stage,
            "reason": reason,
            "at": datetime.now().isoformat(),
        })
        self.stage_retry_counts[stage] = self.stage_retry_counts.get(stage, 0) + 1

    def stage_retries(self, stage: str) -> int:
        return self.stage_retry_counts.get(stage, 0)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["invariant_violations"] = self.invariant_violations
        d["stage_retry_counts"] = self.stage_retry_counts
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "InvariantTaskContext":
        fields = set(TaskContext.__dataclass_fields__) | {"invariant_violations", "stage_retry_counts"}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ─── FSM с инвариантами ───────────────────────────────────────────────────────

class InvariantStateMachine(TaskStateMachine):
    """
    Расширение z13 FSM: после каждого этапа — жёсткая проверка инвариантов.
    Нарушение → откат + перегенерация с объяснением нарушения в промпте.
    Превышение max_retries → эскалация (возвращаем статус FAILED).
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = MODEL,
        max_retries: int = MAX_RETRIES_DEFAULT,
    ):
        super().__init__(client, model)
        self.checker = InvariantChecker(client, model)
        self.max_retries = max_retries
        # Отдельное хранилище с поддержкой InvariantTaskContext
        self.storage = _InvariantTaskStorage()

    def create_task(self, user_id: str, user_task: str) -> InvariantTaskContext:
        task_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ctx = InvariantTaskContext(
            task_id=task_id,
            user_task=user_task,
            current_state=TaskState.PLANNING,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        self.storage.save(ctx)
        return ctx

    def run_stage_with_invariants(
        self,
        ctx: InvariantTaskContext,
        invariants: list[Invariant],
        violation_hint: Optional[str] = None,
    ) -> tuple[str, bool, Optional[str]]:
        """
        Запустить текущий этап + проверить инварианты.
        Возвращает (output, passed, violation_reason).
        """
        state = ctx.state()

        # Запустить этап с опциональным хинтом о нарушении
        if violation_hint and state == TaskState.PLANNING:
            output = self._run_planning_with_hint(ctx, violation_hint)
        elif violation_hint and state == TaskState.EXECUTION:
            output = self._run_execution_with_hint(ctx, violation_hint)
        elif violation_hint and state == TaskState.VALIDATION:
            output = self._run_validation(ctx)
        else:
            output = self.run_current_stage(ctx)

        # Жёсткая проверка инвариантов
        passed, violation = self.checker.check(
            content=output,
            invariants=invariants,
            stage_label=state.value,
        )
        return output, passed, violation

    def _run_planning_with_hint(self, ctx: InvariantTaskContext, hint: str) -> str:
        """Перегенерация плана с указанием нарушённого инварианта."""
        messages = [
            {"role": "system", "content": SYSTEM_PLANNING},
            {"role": "user", "content": (
                f"Задача:\n{ctx.user_task}\n\n"
                f"⚠️ Предыдущий план нарушил ограничение:\n{hint}\n\n"
                f"Составь новый план, строго соблюдая это ограничение. "
                f"НЕ выполняй — только план."
            )},
        ]
        result = self._call_llm(messages)
        ctx.plan = result
        self.storage.save(ctx)
        return result

    def _run_execution_with_hint(self, ctx: InvariantTaskContext, hint: str) -> str:
        """Перегенерация выполнения с указанием нарушённого инварианта."""
        messages = [
            {"role": "system", "content": SYSTEM_EXECUTION},
            {"role": "user", "content": (
                f"Исходная задача:\n{ctx.user_task}\n\n"
                f"Готовый план:\n{ctx.plan}\n\n"
                f"⚠️ Предыдущий результат нарушил ограничение:\n{hint}\n\n"
                f"Выполни задачу строго по плану, соблюдая указанное ограничение."
            )},
        ]
        result = self._call_llm(messages)
        ctx.execution_result = result
        ctx.retry_count += 1
        self.storage.save(ctx)
        return result


class _InvariantTaskStorage(TaskStorage):
    """Хранилище, которое умеет сохранять/загружать InvariantTaskContext."""

    def save(self, ctx: TaskContext) -> None:
        ctx.updated_at = datetime.now().isoformat()
        path = self._path(ctx.task_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ctx.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, task_id: str) -> Optional[InvariantTaskContext]:
        p = self._path(task_id)
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            return InvariantTaskContext.from_dict(json.load(f))


# ─── Пресеты инвариантов для демо ────────────────────────────────────────────

DEMO_INVARIANTS = [
    Invariant(
        name="no_javascript",
        description="Не использовать JavaScript",
        rule_text="Решение НЕ ДОЛЖНО содержать JavaScript-код. Только Python.",
    ),
    Invariant(
        name="use_type_hints",
        description="Обязательны аннотации типов",
        rule_text="Весь Python-код ДОЛЖЕН содержать аннотации типов (type hints).",
    ),
]

DEMO_TASK_Z14 = "Напиши функцию для подсчёта слов в тексте."

DEMO_VIOLATING_TASK = (
    "Напиши скрипт для веб-интерфейса с кнопкой, которая считает слова. "
    "Можно использовать любой язык."
)
