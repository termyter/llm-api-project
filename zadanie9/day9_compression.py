"""
День 9 — Управление контекстом: сжатие истории.

SummaryAgent:
  - хранит последние N сообщений "как есть"
  - когда история становится длиннее порога — сжимает старые сообщения в summary
  - подставляет summary в запрос вместо полной истории
  - экономит токены при длинных диалогах
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))


# ── Статистика одного хода ────────────────────────────────────────────────────

@dataclass
class TurnStats:
    turn: int
    prompt_tokens: int
    completion_tokens: int
    summary_tokens: int = 0       # токены на само сжатие (если было)
    compressed: bool = False       # был ли сжат контекст на этом ходу

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.summary_tokens

    @property
    def cost_rub(self) -> float:
        inp = (self.prompt_tokens + self.summary_tokens) / 1_000_000 * 25
        out = self.completion_tokens / 1_000_000 * 37
        return inp + out


# ── SummaryAgent ─────────────────────────────────────────────────────────────

class SummaryAgent:
    """
    Агент с компрессией истории.

    Алгоритм:
    1. Ведёт рабочую историю (последние keep_recent сообщений)
    2. Когда history > keep_recent + compress_after — сжимает "старые" сообщения
    3. summary обновляется инкрементально (новые старые факты добавляются к старому summary)
    4. В запрос к API идёт: system + [summary как system] + last keep_recent messages

    Args:
        keep_recent:     сколько последних сообщений оставлять как есть
        compress_after:  через сколько новых сообщений после последнего сжатия — сжать снова
    """

    def __init__(
        self,
        client: OpenAI,
        model_id: str = "deepseek-chat",
        system_prompt: str = "Ты полезный ассистент. Отвечай по делу.",
        keep_recent: int = 6,
        compress_after: int = 8,
    ):
        self.client = client
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.keep_recent = keep_recent
        self.compress_after = compress_after

        self.history: list[dict] = []
        self.summary: str = ""
        self.stats: list[TurnStats] = []
        self._turn = 0

    # ── Сборка запроса ────────────────────────────────────────────────────────

    def _build_messages(self) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            msgs.append({
                "role": "system",
                "content": (
                    "📝 [Краткое содержание предыдущей части диалога]\n"
                    + self.summary
                ),
            })
        msgs.extend(self.history[-self.keep_recent :])
        return msgs

    # ── Сжатие ───────────────────────────────────────────────────────────────

    def _compress(self) -> int:
        """
        Сжать сообщения, которые не войдут в keep_recent.
        Возвращает кол-во потраченных на сжатие токенов.
        """
        to_compress = self.history[: -self.keep_recent]
        if not to_compress:
            return 0

        formatted = "\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in to_compress
        )

        if self.summary:
            user_content = (
                f"Текущее резюме предыдущего диалога:\n{self.summary}\n\n"
                f"Новые сообщения для добавления в резюме:\n{formatted}\n\n"
                "Обнови резюме, добавив ключевые факты из новых сообщений. "
                "Резюме должно быть кратким (до 200 слов)."
            )
        else:
            user_content = (
                f"Создай краткое резюме этих сообщений диалога (до 200 слов):\n{formatted}"
            )

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "Ты помощник для сжатия истории диалога."},
                {"role": "user", "content": user_content},
            ],
            max_tokens=400,
        )

        self.summary = resp.choices[0].message.content.strip()
        self.history = self.history[-self.keep_recent :]
        return resp.usage.total_tokens

    # ── Основной метод ────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> tuple[str, TurnStats]:
        self._turn += 1
        self.history.append({"role": "user", "content": user_message})

        # Сжатие: если история переполнилась
        summary_tokens = 0
        compressed = False
        if len(self.history) > self.keep_recent + self.compress_after:
            summary_tokens = self._compress()
            compressed = True

        # Основной запрос
        messages = self._build_messages()
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )

        answer = resp.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})

        stat = TurnStats(
            turn=self._turn,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            summary_tokens=summary_tokens,
            compressed=compressed,
        )
        self.stats.append(stat)
        return answer, stat

    def reset(self) -> None:
        self.history = []
        self.summary = ""
        self.stats = []
        self._turn = 0

    @property
    def context_size(self) -> int:
        """Текущее кол-во сообщений в рабочей истории."""
        return len(self.history)


# ── Сравнительный агент (без сжатия) ─────────────────────────────────────────

class PlainAgent:
    """Агент без сжатия — хранит всю историю."""

    def __init__(self, client: OpenAI, model_id: str = "deepseek-chat",
                 system_prompt: str = "Ты полезный ассистент. Отвечай по делу."):
        self.client = client
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.history: list[dict] = []
        self.stats: list[TurnStats] = []
        self._turn = 0

    def chat(self, user_message: str) -> tuple[str, TurnStats]:
        self._turn += 1
        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        resp = self.client.chat.completions.create(model=self.model_id, messages=messages)
        answer = resp.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})
        stat = TurnStats(
            turn=self._turn,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
        )
        self.stats.append(stat)
        return answer, stat


# ── Отображение ───────────────────────────────────────────────────────────────

def bar(value: int, max_value: int, width: int = 20) -> str:
    filled = int(width * value / max_value) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_turn(label: str, stat: TurnStats) -> None:
    compress_mark = " 🗜 СЖАТИЕ" if stat.compressed else ""
    print(
        f"  [{label}] Ход {stat.turn:>2}{compress_mark}\n"
        f"    вход:  {stat.prompt_tokens:>6,} т  {bar(stat.prompt_tokens, 4000)}\n"
        f"    выход: {stat.completion_tokens:>6,} т\n"
        f"    итого: {stat.total_tokens:>6,} т  {stat.cost_rub:.4f} ₽"
    )


def print_comparison(plain_stats: list[TurnStats], summary_stats: list[TurnStats]) -> None:
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ: без сжатия vs со сжатием")
    print("=" * 60)

    plain_input = sum(s.prompt_tokens for s in plain_stats)
    summary_input = sum(s.prompt_tokens + s.summary_tokens for s in summary_stats)
    plain_cost = sum(s.cost_rub for s in plain_stats)
    summary_cost = sum(s.cost_rub for s in summary_stats)
    saved_pct = (1 - summary_input / plain_input) * 100 if plain_input > 0 else 0

    print(f"\n{'Метрика':<28} {'Без сжатия':>12} {'Со сжатием':>12} {'Экономия':>10}")
    print("─" * 64)
    print(f"{'Входящих токенов (всего)':<28} {plain_input:>12,} {summary_input:>12,} {saved_pct:>9.1f}%")
    print(f"{'Стоимость сессии (₽)':<28} {plain_cost:>12.4f} {summary_cost:>12.4f} {(1-summary_cost/plain_cost)*100:>9.1f}%")
    print(f"{'Макс. токенов в одном ходу':<28} {max(s.prompt_tokens for s in plain_stats):>12,} {max(s.prompt_tokens for s in summary_stats):>12,}")
    print(f"{'Кол-во сжатий':<28} {'—':>12} {sum(1 for s in summary_stats if s.compressed):>12}")


# ── Демо ─────────────────────────────────────────────────────────────────────

QUESTIONS = [
    "Что такое машинное обучение? Объясни кратко.",
    "Какие бывают типы ML: supervised, unsupervised?",
    "Что такое нейронная сеть и как она работает?",
    "Чем отличается CNN от RNN?",
    "Что такое трансформер (transformer)?",
    "Какова роль attention mechanism?",
    "Что такое fine-tuning модели?",
    "Что такое RAG (Retrieval Augmented Generation)?",
    "Вспомни — что я спрашивал в самом начале?",      # тест памяти
    "Что такое prompt engineering?",
    "Какой вопрос я задавал про нейронные сети?",     # тест памяти
    "Что такое embeddings?",
    "Как работает vector database?",
    "Подведи итог нашего разговора.",
]


def run_demo() -> None:
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    SYSTEM = "Ты эксперт по ML и AI. Отвечай кратко — 2-3 предложения максимум."

    plain = PlainAgent(client, system_prompt=SYSTEM)
    smart = SummaryAgent(client, system_prompt=SYSTEM, keep_recent=6, compress_after=6)

    print("=" * 60)
    print("День 9 — Сжатие истории (компрессия контекста)")
    print("=" * 60)
    print(f"Вопросов: {len(QUESTIONS)} | keep_recent=6 | compress_after=6\n")

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'─'*60}")
        print(f"❓ Вопрос {i}: {q}")

        _, p_stat = plain.chat(q)
        ans, s_stat = smart.chat(q)

        print_turn("БЕЗ", p_stat)
        print_turn("СО ", s_stat)
        print(f"  → Ответ (со сжатием): {ans[:120]}...")

    print_comparison(plain.stats, smart.stats)

    # Тест качества памяти
    print("\n" + "=" * 60)
    print("ТЕСТ ПАМЯТИ — агент со сжатием помнит начало?")
    print("=" * 60)
    for stat in smart.stats:
        if stat.compressed:
            print(f"  Сжатие произошло на ходу {stat.turn}")
    print(f"\nТекущее summary агента со сжатием:\n{smart.summary[:400]}...")


if __name__ == "__main__":
    run_demo()
