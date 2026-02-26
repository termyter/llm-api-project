#!/usr/bin/env python3
"""
День 8 — Работа с токенами.

Демонстрирует:
  1. Подсчёт токенов: текущий запрос / история / ответ модели
  2. Сравнение: короткий / длинный / переполненный диалог
  3. Рост стоимости по мере диалога
  4. Что ломается при переполнении и как это чинить (sliding window)
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID      = "deepseek-chat"
CONTEXT_LIMIT = 64_000   # deepseek-chat: 64 000 токенов контекста
SAFE_LIMIT    = 60_000   # порог предупреждения

# Стоимость deepseek-chat за 1 000 000 токенов (api.deepseek.com)
PRICE_INPUT_RUB  = 25.0  # ₽ за 1M prompt-токенов
PRICE_OUTPUT_RUB = 37.0  # ₽ за 1M completion-токенов


# ─────────────────────────────────────────────────────────────────────────────
# Структура данных для одного хода
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TurnStats:
    turn: int
    user_message_chars: int       # длина сообщения пользователя (символы)
    prompt_tokens: int            # токены ВХОДА  (история + запрос + system)
    completion_tokens: int        # токены ОТВЕТА модели
    total_tokens: int             # prompt + completion
    session_input_total: int      # нарастающий итог входных токенов
    session_output_total: int     # нарастающий итог выходных токенов

    @property
    def session_total(self) -> int:
        return self.session_input_total + self.session_output_total

    @property
    def turn_cost_rub(self) -> float:
        return (
            self.prompt_tokens     * PRICE_INPUT_RUB +
            self.completion_tokens * PRICE_OUTPUT_RUB
        ) / 1_000_000

    @property
    def session_cost_rub(self) -> float:
        return (
            self.session_input_total  * PRICE_INPUT_RUB +
            self.session_output_total * PRICE_OUTPUT_RUB
        ) / 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Исключение при переполнении
# ─────────────────────────────────────────────────────────────────────────────

class ContextOverflowError(RuntimeError):
    """Вызывается когда оценочное число токенов превышает лимит."""


# ─────────────────────────────────────────────────────────────────────────────
# Агент с детальным подсчётом токенов
# ─────────────────────────────────────────────────────────────────────────────

class TokenAwareAgent:
    """
    LLM-агент, который отслеживает токены на каждом ходу.

    После каждого вызова chat() агент сохраняет TurnStats:
      - сколько токенов пошло на вход (вся история + новое сообщение)
      - сколько токенов в ответе
      - нарастающий итог за сессию
      - стоимость в рублях
    """

    def __init__(
        self,
        client: OpenAI,
        model_id: str = MODEL_ID,
        system_prompt: str = "Ты полезный ассистент. Отвечай по делу.",
        context_limit: int = CONTEXT_LIMIT,
        safe_limit: int = SAFE_LIMIT,
    ):
        self.client        = client
        self.model_id      = model_id
        self.system_prompt = system_prompt
        self.context_limit = context_limit
        self.safe_limit    = safe_limit

        self.history: list[dict] = []
        self.stats:   list[TurnStats] = []
        self._session_in:  int = 0
        self._session_out: int = 0

    # ── Вспомогательные ──────────────────────────────────────────────────────

    def _messages(self, user_msg: str) -> list[dict]:
        return (
            [{"role": "system", "content": self.system_prompt}]
            + self.history
            + [{"role": "user", "content": user_msg}]
        )

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Грубая оценка: символов / 4 ≈ токены (приемлемо для кириллицы)."""
        return sum(len(m["content"]) for m in messages) // 4

    # ── Основной метод ───────────────────────────────────────────────────────

    def chat(self, user_message: str) -> tuple[str, TurnStats]:
        """
        Отправить сообщение, получить ответ + детальную статистику токенов.

        Raises:
            ContextOverflowError: если оценочный объём токенов > context_limit
        """
        messages = self._messages(user_message)

        # Предварительная проверка лимита (без API-вызова)
        estimated = self._estimate_tokens(messages)
        if estimated > self.context_limit:
            raise ContextOverflowError(
                f"Оценка: ~{estimated:,} токенов > лимит {self.context_limit:,}. "
                "История слишком длинная — нужен сброс или sliding window."
            )

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )

        answer   = resp.choices[0].message.content
        usage    = resp.usage
        p_tok    = usage.prompt_tokens
        c_tok    = usage.completion_tokens
        tot_tok  = usage.total_tokens

        self._session_in  += p_tok
        self._session_out += c_tok

        stat = TurnStats(
            turn                = len(self.stats) + 1,
            user_message_chars  = len(user_message),
            prompt_tokens       = p_tok,
            completion_tokens   = c_tok,
            total_tokens        = tot_tok,
            session_input_total = self._session_in,
            session_output_total= self._session_out,
        )
        self.stats.append(stat)

        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant",  "content": answer})

        # Предупреждение при приближении к лимиту
        if p_tok >= self.safe_limit:
            print(f"  ⚠️  ВНИМАНИЕ: {p_tok:,} из {self.context_limit:,} "
                  f"токенов ({p_tok / self.context_limit:.0%}) — скоро переполнение!")

        return answer, stat

    def reset(self) -> None:
        self.history        = []
        self.stats          = []
        self._session_in    = 0
        self._session_out   = 0

    def trim_history(self, keep_pairs: int = 3) -> int:
        """
        Sliding window: удалить старые пары user/assistant из истории.
        Возвращает кол-во удалённых пар.
        """
        total_pairs  = len(self.history) // 2
        to_remove    = max(0, total_pairs - keep_pairs)
        self.history = self.history[to_remove * 2:]
        return to_remove


# ─────────────────────────────────────────────────────────────────────────────
# Вывод статистики
# ─────────────────────────────────────────────────────────────────────────────

BAR_FULL  = 40   # ширина прогресс-бара

def _bar(value: int, maximum: int, width: int = BAR_FULL) -> str:
    filled = min(width, int(width * value / max(maximum, 1)))
    return "█" * filled + "░" * (width - filled)


def print_turn_stats(stat: TurnStats, context_limit: int = CONTEXT_LIMIT) -> None:
    """Подробная статистика одного хода."""
    bar = _bar(stat.prompt_tokens, context_limit)
    pct = stat.prompt_tokens / context_limit * 100

    print(f"  ┌─ ХОД {stat.turn} {'─' * 52}┐")
    print(f"  │  Сообщение пользователя  : {stat.user_message_chars:>6} символов")
    print(f"  │  Токены ВХОДА (prompt)   : {stat.prompt_tokens:>8,}   ← вся история + запрос")
    print(f"  │  Токены ОТВЕТА           : {stat.completion_tokens:>8,}   ← только ответ модели")
    print(f"  │  Итого за ход            : {stat.total_tokens:>8,}")
    print(f"  │  ─────────────────────────────────────────────────────")
    print(f"  │  Нарастающий итог        : {stat.session_total:>8,}   токенов с начала сессии")
    print(f"  │  Стоимость этого хода    : {stat.turn_cost_rub:>10.5f} ₽")
    print(f"  │  Стоимость сессии        : {stat.session_cost_rub:>10.5f} ₽")
    print(f"  │  Заполнено контекста     : [{bar}] {pct:.1f}%")
    print(f"  └{'─' * 58}┘")


def print_session_summary(agent: TokenAwareAgent, title: str) -> None:
    """Итоговая таблица с ростом токенов по ходам."""
    if not agent.stats:
        return

    print(f"\n{'═' * 70}")
    print(f"  ИТОГ СЕССИИ: {title}")
    print(f"{'═' * 70}")
    print(f"  {'Ход':>3}  {'Вход':>8}  {'Выход':>7}  {'Нарастающий':>12}  {'Стоимость':>10}  Прогресс")
    print(f"  {'─'*3}  {'─'*8}  {'─'*7}  {'─'*12}  {'─'*10}  {'─'*20}")

    max_total = agent.stats[-1].session_total if agent.stats else 1
    for s in agent.stats:
        bar = "▓" * min(20, int(20 * s.session_total / max(max_total, 1)))
        print(
            f"  {s.turn:>3}  {s.prompt_tokens:>8,}  "
            f"{s.completion_tokens:>7,}  {s.session_total:>12,}  "
            f"{s.turn_cost_rub:>10.5f} ₽  {bar}"
        )

    last = agent.stats[-1]
    print(f"  {'─'*3}  {'─'*8}  {'─'*7}  {'─'*12}  {'─'*10}")
    print(
        f"  ΣΣΣ  вход={agent._session_in:,}  "
        f"выход={agent._session_out:,}  "
        f"всего={last.session_total:,}  "
        f"итого={last.session_cost_rub:.5f} ₽"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Сценарий 1 — Короткий диалог
# ─────────────────────────────────────────────────────────────────────────────

def demo_short(client: OpenAI) -> None:
    """3 коротких хода — минимум токенов."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  СЦЕНАРИЙ 1: КОРОТКИЙ ДИАЛОГ (3 хода, простые сообщения)          ║")
    print("╚" + "═" * 68 + "╝")

    agent = TokenAwareAgent(client)

    exchanges = [
        "Привет! Как тебя зовут?",
        "Сколько будет 2 + 2?",
        "Спасибо, пока!",
    ]

    for msg in exchanges:
        print(f"\n  👤 {msg}")
        answer, stat = agent.chat(msg)
        print(f"  🤖 {answer[:100]}{'…' if len(answer) > 100 else ''}")
        print_turn_stats(stat)

    print_session_summary(agent, "Короткий диалог")


# ─────────────────────────────────────────────────────────────────────────────
# Сценарий 2 — Длинный диалог
# ─────────────────────────────────────────────────────────────────────────────

def demo_long(client: OpenAI) -> None:
    """8 ходов с длинными вопросами — наблюдаем экспоненциальный рост."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  СЦЕНАРИЙ 2: ДЛИННЫЙ ДИАЛОГ (8 ходов, длинные вопросы)           ║")
    print("║  Ключевой эффект: prompt-токены РАСТУТ с каждым ходом,           ║")
    print("║  т.к. вся история добавляется к каждому запросу.                 ║")
    print("╚" + "═" * 68 + "╝")

    agent = TokenAwareAgent(client)

    exchanges = [
        "Объясни, как работает машинное обучение. Начни с самых основ и дай развёрнутый ответ.",
        "Расскажи подробнее про нейронные сети. Как они обучаются? Что такое backpropagation и gradient descent?",
        "Чем supervised learning отличается от unsupervised? Приведи по 3 примера каждого с описанием.",
        "Что такое reinforcement learning? Как AI учится играть в шахматы и видеоигры?",
        "Объясни архитектуру трансформеров. Почему они стали революцией в NLP?",
        "Что такое attention mechanism и self-attention? Как они работают внутри GPT?",
        "Какие проблемы у LLM: галлюцинации, bias, alignment? Как их решают?",
        "Подведи итог нашего разговора: что самое важное ты объяснил?",
    ]

    for msg in exchanges:
        preview = msg[:70] + "…" if len(msg) > 70 else msg
        print(f"\n  👤 {preview}")
        answer, stat = agent.chat(msg)
        print(f"  🤖 {answer[:100]}{'…' if len(answer) > 100 else ''}")
        print_turn_stats(stat)

    print_session_summary(agent, "Длинный диалог")


# ─────────────────────────────────────────────────────────────────────────────
# Сценарий 3 — Переполнение и sliding window
# ─────────────────────────────────────────────────────────────────────────────

def demo_overflow(client: OpenAI) -> None:
    """
    Заполняем историю, пока не сработает ContextOverflowError.
    Затем показываем sliding window как решение.
    """
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  СЦЕНАРИЙ 3: ПЕРЕПОЛНЕНИЕ КОНТЕКСТА                              ║")
    print("╠" + "═" * 68 + "╣")
    print("║  Используем маленький искусственный лимит (3 000 токенов)        ║")
    print("║  для наглядной демонстрации без лишних затрат.                   ║")
    print("╚" + "═" * 68 + "╝")

    DEMO_LIMIT = 3_000   # намеренно маленький лимит для демо

    agent = TokenAwareAgent(
        client,
        context_limit=DEMO_LIMIT,
        safe_limit=2_500,
    )

    # Длинный текст-наполнитель (~300 символов → ~75 токенов за раз)
    filler = (
        "Подробно объясни следующую концепцию: "
        + "алгоритм Дейкстры для нахождения кратчайшего пути в графе. "
        + "Опиши каждый шаг алгоритма, приведи псевдокод и пример на графе из 5 вершин. "
    )

    print(f"\n  Лимит контекста: {DEMO_LIMIT:,} токенов (для демо).")
    print(f"  Каждое сообщение ≈ {len(filler) // 4} токенов.\n")

    crashed_at = None

    for turn in range(1, 20):
        print(f"  ── Ход {turn} ─────────────────────────────────────────────────")
        try:
            answer, stat = agent.chat(filler)
            print(f"  ✅ OK: вход={stat.prompt_tokens:,}  выход={stat.completion_tokens:,}  "
                  f"нарастающий={stat.session_total:,}")
        except ContextOverflowError as e:
            crashed_at = turn
            print(f"\n  💥 ПЕРЕПОЛНЕНИЕ НА ХОДУ {turn}!")
            print(f"  Ошибка: {e}")
            break

    print_session_summary(agent, "До переполнения")

    # ── Демонстрация sliding window ───────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  РЕШЕНИЕ: SLIDING WINDOW")
    print("  Удаляем старые пары из истории, чтобы оставаться в лимите.")
    print(f"{'─' * 70}")

    agent_sw = TokenAwareAgent(
        client,
        context_limit=DEMO_LIMIT,
        safe_limit=2_500,
        system_prompt="Ты ассистент. Отвечай кратко (1-2 предложения).",
    )

    KEEP_PAIRS = 2   # сколько пар user/assistant оставлять в окне

    for turn in range(1, 8):
        # Перед отправкой — обрезаем историю если нужно
        estimated = agent_sw._estimate_tokens(agent_sw._messages(filler))

        if estimated > DEMO_LIMIT:
            removed = agent_sw.trim_history(keep_pairs=KEEP_PAIRS)
            new_est = agent_sw._estimate_tokens(agent_sw._messages(filler))
            print(f"\n  🗑️  Ход {turn}: удалено {removed} старых пар, "
                  f"оценка {estimated:,} → {new_est:,} токенов")

        try:
            answer, stat = agent_sw.chat(filler)
            print(f"  ✅ Ход {turn}: вход={stat.prompt_tokens:,}  "
                  f"выход={stat.completion_tokens:,}  "
                  f"история={len(agent_sw.history)//2} пар")
        except ContextOverflowError as e:
            print(f"  ❌ Ход {turn}: {e}")

    print_session_summary(agent_sw, "Sliding window — токены под контролем")


# ─────────────────────────────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    print()
    print("╔" + "═" * 68 + "╗")
    print("║           ДЕНЬ 8: РАБОТА С ТОКЕНАМИ                             ║")
    print("║  Модель: deepseek-chat  │  Контекст: 64 000 токенов             ║")
    print("║  Цена: 25 ₽/1M вход  │  37 ₽/1M выход  (api.deepseek.com)     ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("  Что показываем:")
    print("  1. prompt_tokens  — всё, что идёт в запрос (история + сообщение)")
    print("  2. completion_tokens — только ответ модели")
    print("  3. Нарастающий итог — суммарный расход токенов за сессию")
    print("  4. Как токены НАКАПЛИВАЮТСЯ и цена РАСТЁТ с каждым ходом")
    print("  5. Что происходит при переполнении и как его чинить")

    input("\n  [Enter → Сценарий 1: Короткий диалог]")
    demo_short(client)

    input("\n  [Enter → Сценарий 2: Длинный диалог]")
    demo_long(client)

    input("\n  [Enter → Сценарий 3: Переполнение + Sliding Window]")
    demo_overflow(client)

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  КЛЮЧЕВЫЕ ВЫВОДЫ                                                 ║")
    print("╠" + "═" * 68 + "╣")
    print("║  ▸ prompt_tokens растут с каждым ходом — вся история в запросе  ║")
    print("║  ▸ Короткий диалог: ~100–300 tok/ход ≈ 0.001 ₽                 ║")
    print("║  ▸ Длинный диалог: 2 000–10 000 tok/ход ≈ 0.05–0.3 ₽          ║")
    print("║  ▸ Переполнение → ошибка API (или обрезка начала истории)       ║")
    print("║  ▸ Sliding window: удаляем старые пары, сохраняем N последних   ║")
    print("║  ▸ Альтернатива: summary-compression (сжать старое в резюме)    ║")
    print("╚" + "═" * 68 + "╝")
    print()


if __name__ == "__main__":
    main()
