"""
Day 10: Context Management Strategies
3 approaches to managing conversation history:
  1. SlidingWindowAgent — keeps only last N messages
  2. StickyFactsAgent  — extracts key-value facts after each turn + last N messages
  3. BranchingAgent    — checkpoint + fork into independent branches
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

# ─── pricing ────────────────────────────────────────────────────────────────
INPUT_PRICE_RUB_PER_1M  = 25.0   # ₽ per 1M input tokens
OUTPUT_PRICE_RUB_PER_1M = 37.0   # ₽ per 1M output tokens


# ─── demo scenario ──────────────────────────────────────────────────────────
DEMO_SHORT = [
    "Хочу сделать iOS-приложение для трекинга привычек. Бюджет 300 000 рублей, срок — 3 месяца.",
    "Нужны push-уведомления, виджет на экран блокировки и синхронизация через iCloud.",
    "Что мы уже определили по проекту? Перечисли ключевые параметры.",
]

BRANCH_DEMO_BASE = [
    "Разрабатываем приложение: iOS, трекинг привычек, бюджет 300 000 рублей.",
    "Договорились: push-уведомления обязательны, виджет — желателен, срок 3 месяца.",
]

BRANCH_DEMO_FORK_A = "Нам урезали бюджет до 200 000 рублей. Что придётся исключить из функциональности?"
BRANCH_DEMO_FORK_B = "Нам выделили дополнительные 300 000 рублей. Что можно добавить в проект?"

SYSTEM_PROMPT = (
    "Ты — опытный IT-менеджер. Помогаешь формировать техническое задание для мобильного приложения. "
    "Отвечай лаконично — не более 3 коротких абзацев. Пиши по-русски."
)


# ─── stats ──────────────────────────────────────────────────────────────────
@dataclass
class TurnStats:
    turn: int
    strategy: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extra_tokens: int = 0        # tokens from facts-update call (StickyFacts only)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.extra_tokens

    @property
    def cost_rub(self) -> float:
        inp = (self.prompt_tokens + self.extra_tokens) / 1_000_000 * INPUT_PRICE_RUB_PER_1M
        out = self.completion_tokens / 1_000_000 * OUTPUT_PRICE_RUB_PER_1M
        return round(inp + out, 4)


# ────────────────────────────────────────────────────────────────────────────
# 1. Sliding Window Agent
# ────────────────────────────────────────────────────────────────────────────
class SlidingWindowAgent:
    """Keeps only the last `window_size` messages in context."""

    def __init__(self, client, model_id: str, system_prompt: str, window_size: int = 8):
        self.client       = client
        self.model_id     = model_id
        self.system_prompt = system_prompt
        self.window_size  = window_size
        self.history: list[dict] = []
        self.stats: list[TurnStats] = []

    # ── public ──────────────────────────────────────────────────────────────
    def chat(self, user_message: str) -> tuple[str, TurnStats]:
        self.history.append({"role": "user", "content": user_message})
        window = self.history[-self.window_size:]
        messages = [{"role": "system", "content": self.system_prompt}] + window

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})

        stat = TurnStats(
            turn=len(self.stats) + 1,
            strategy="sliding_window",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        self.stats.append(stat)
        return reply, stat

    def context_size(self) -> int:
        return min(len(self.history), self.window_size)

    def reset(self):
        self.history.clear()
        self.stats.clear()


# ────────────────────────────────────────────────────────────────────────────
# 2. Sticky Facts Agent
# ────────────────────────────────────────────────────────────────────────────
FACTS_EXTRACTION_PROMPT = """\
Из диалога ниже извлеки ключевые факты о проекте в виде JSON-объекта.
Используй только эти поля (оставляй пустую строку "" если информации нет):
{
  "цель": "",
  "платформа": "",
  "бюджет": "",
  "срок": "",
  "команда": "",
  "требования": ""
}
Возвращай ТОЛЬКО JSON, без пояснений."""


class StickyFactsAgent:
    """
    After every turn: asks LLM to extract key-value facts from the full history,
    then injects those facts as a system-level block. Main context uses last `window_size` messages.
    """

    def __init__(self, client, model_id: str, system_prompt: str, window_size: int = 6):
        self.client        = client
        self.model_id      = model_id
        self.system_prompt = system_prompt
        self.window_size   = window_size
        self.history: list[dict] = []
        self.facts: dict[str, str] = {}
        self.stats: list[TurnStats] = []

    # ── public ──────────────────────────────────────────────────────────────
    def chat(self, user_message: str) -> tuple[str, TurnStats]:
        self.history.append({"role": "user", "content": user_message})

        # Build context: facts block + last N messages
        messages = [{"role": "system", "content": self._system_with_facts()}]
        messages += self.history[-self.window_size:]

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})

        # Update facts (extra LLM call)
        extra_tokens = self._update_facts()

        stat = TurnStats(
            turn=len(self.stats) + 1,
            strategy="sticky_facts",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            extra_tokens=extra_tokens,
        )
        self.stats.append(stat)
        return reply, stat

    def get_facts_text(self) -> str:
        if not self.facts:
            return "(факты ещё не извлечены)"
        lines = [f"• {k}: {v}" for k, v in self.facts.items() if v]
        return "\n".join(lines) if lines else "(факты пусты)"

    def context_size(self) -> int:
        return min(len(self.history), self.window_size)

    def reset(self):
        self.history.clear()
        self.facts.clear()
        self.stats.clear()

    # ── private ─────────────────────────────────────────────────────────────
    def _system_with_facts(self) -> str:
        if not self.facts:
            return self.system_prompt
        facts_lines = "\n".join(f"  {k}: {v}" for k, v in self.facts.items() if v)
        facts_block = f"\n\n--- Ключевые факты о проекте ---\n{facts_lines}\n---"
        return self.system_prompt + facts_block

    def _update_facts(self) -> int:
        """Ask LLM to re-extract facts from full history. Returns tokens used."""
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self.history
        )
        messages = [
            {"role": "system", "content": FACTS_EXTRACTION_PROMPT},
            {"role": "user",   "content": history_text},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=200,
            )
            import json
            raw = resp.choices[0].message.content.strip()
            # strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            self.facts = {k: str(v) for k, v in parsed.items()}
            return resp.usage.prompt_tokens + resp.usage.completion_tokens
        except Exception:
            return 0


# ────────────────────────────────────────────────────────────────────────────
# 3. Branching Agent
# ────────────────────────────────────────────────────────────────────────────
BRANCH_A = "ветка-а"
BRANCH_B = "ветка-б"
BRANCH_MAIN = "main"


class BranchingAgent:
    """
    Supports checkpoints and independent branches.
    After a checkpoint you can fork: each branch evolves independently.
    """

    def __init__(self, client, model_id: str, system_prompt: str):
        self.client        = client
        self.model_id      = model_id
        self.system_prompt = system_prompt

        self.branches: dict[str, list[dict]] = {BRANCH_MAIN: []}
        self.active_branch: str = BRANCH_MAIN
        self.checkpoint: Optional[list[dict]] = None
        self.stats: list[TurnStats] = []

    # ── public ──────────────────────────────────────────────────────────────
    def chat(self, user_message: str) -> tuple[str, TurnStats]:
        history = self.branches[self.active_branch]
        history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}] + history
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})

        stat = TurnStats(
            turn=len(self.stats) + 1,
            strategy=f"branching[{self.active_branch}]",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
        self.stats.append(stat)
        return reply, stat

    def create_checkpoint(self):
        """Save current branch history as checkpoint."""
        self.checkpoint = copy.deepcopy(self.branches[self.active_branch])

    def fork(self, branch_name: str, from_checkpoint: bool = True):
        """Create a new branch from checkpoint (or current state) and switch to it."""
        base = self.checkpoint if (from_checkpoint and self.checkpoint is not None) \
               else self.branches[self.active_branch]
        self.branches[branch_name] = copy.deepcopy(base)
        self.active_branch = branch_name

    def switch(self, branch_name: str):
        if branch_name not in self.branches:
            raise KeyError(f"Branch '{branch_name}' does not exist")
        self.active_branch = branch_name

    def branch_names(self) -> list[str]:
        return list(self.branches.keys())

    def context_size(self) -> int:
        return len(self.branches[self.active_branch])

    def reset(self):
        self.branches = {BRANCH_MAIN: []}
        self.active_branch = BRANCH_MAIN
        self.checkpoint = None
        self.stats.clear()
