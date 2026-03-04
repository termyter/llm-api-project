"""
Day 12: Персонализация ассистента

Поверх 3-слойной памяти (Day 11) добавляем UserProfile:
  - стиль общения (технический / деловой / дружеский)
  - формат ответов (краткий / подробный / маркированный)
  - уровень экспертизы (новичок / средний / эксперт)
  - ограничения (без эмодзи, без англицизмов, ...)
  - интересы (автоматически подтягиваются из LTM)

PersonalizedAgent:
  - строит персонализированный system prompt из профиля + LTM
  - обновляет профиль на основе новых LTM-фактов
  - поддерживает быстрое переключение профиля
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from openai import OpenAI

from zadanie11.day11_memory import (
    MemoryAgent, MemoryTurnStats,
    LongTermMemory, ShortTermMemory, WorkingMemory,
    INPUT_PRICE_RUB_PER_1M, OUTPUT_PRICE_RUB_PER_1M,
)

# ─── константы ────────────────────────────────────────────────────────────────
PROFILES_DIR = os.path.join(os.path.dirname(__file__), ".profiles")

STYLE_OPTIONS = {
    "технический":  "Используй технические термины, примеры кода, точные формулировки.",
    "деловой":      "Формальный деловой стиль. Структурировано, без лирики.",
    "дружеский":    "Неформально, с юмором, как общение с другом.",
    "нейтральный":  "Обычный разговорный стиль без особенностей.",
}

FORMAT_OPTIONS = {
    "краткий":       "Отвечай максимально кратко — 1-3 предложения. Только суть.",
    "подробный":     "Давай развёрнутые ответы с примерами и объяснениями.",
    "маркированный": "Структурируй ответ маркированными списками (•, -, 1. 2. 3.).",
    "авто":          "Выбирай формат под вопрос самостоятельно.",
}

EXPERTISE_OPTIONS = {
    "новичок":  "Пользователь — новичок. Объясняй каждый термин, используй аналогии из жизни.",
    "средний":  "Пользователь имеет базовые знания. Не объясняй очевидное.",
    "эксперт":  "Пользователь — эксперт. Без лишних объяснений, сразу к делу.",
}

# ─── предустановленные профили для демо ──────────────────────────────────────
PRESET_PROFILES = {
    "техник": {
        "display_name": "👨‍💻 Технарь",
        "style": "технический",
        "format": "подробный",
        "expertise": "эксперт",
        "restrictions": ["без объяснения базовых понятий"],
        "interests": ["программирование", "архитектура", "алгоритмы"],
    },
    "менеджер": {
        "display_name": "📊 Менеджер",
        "style": "деловой",
        "format": "маркированный",
        "expertise": "средний",
        "restrictions": ["без технического жаргона", "только выводы и риски"],
        "interests": ["управление", "бизнес", "метрики"],
    },
    "новичок": {
        "display_name": "🌱 Новичок",
        "style": "дружеский",
        "format": "подробный",
        "expertise": "новичок",
        "restrictions": ["объяснять все термины", "использовать аналогии"],
        "interests": ["обучение", "практика"],
    },
}

# Один вопрос для сравнения профилей
DEMO_QUESTION = "Объясни, что такое async/await и зачем это нужно."


# ─── UserProfile ──────────────────────────────────────────────────────────────
@dataclass
class UserProfile:
    user_id: str
    name: str = ""
    style: str = "нейтральный"
    format: str = "авто"
    expertise: str = "средний"
    restrictions: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def build_system_prompt(self, ltm_facts: list[str] | None = None) -> str:
        """Собирает персонализированный system prompt."""
        lines = [
            "Ты — персональный ассистент с долгосрочной памятью.",
            "",
        ]

        # Имя пользователя
        if self.name:
            lines.append(f"Пользователя зовут {self.name}. Обращайся к нему по имени.")

        # Стиль
        style_hint = STYLE_OPTIONS.get(self.style, "")
        if style_hint:
            lines.append(f"Стиль общения: {style_hint}")

        # Формат
        format_hint = FORMAT_OPTIONS.get(self.format, "")
        if format_hint:
            lines.append(f"Формат ответов: {format_hint}")

        # Уровень
        expertise_hint = EXPERTISE_OPTIONS.get(self.expertise, "")
        if expertise_hint:
            lines.append(f"Уровень пользователя: {expertise_hint}")

        # Ограничения
        if self.restrictions:
            lines.append("Ограничения: " + "; ".join(self.restrictions) + ".")

        # Интересы
        if self.interests:
            lines.append("Интересы пользователя: " + ", ".join(self.interests) + ".")

        # LTM факты
        if ltm_facts:
            lines.append("")
            lines.append("Что ты знаешь о пользователе:")
            for f in ltm_facts:
                lines.append(f"  • {f}")

        lines.append("")
        lines.append("Отвечай на русском языке.")
        return "\n".join(lines)

    def as_summary(self) -> str:
        """Короткое описание профиля для отображения."""
        parts = [
            f"👤 Имя: {self.name or '(не задано)'}",
            f"🎨 Стиль: {self.style}",
            f"📄 Формат: {self.format}",
            f"🎓 Уровень: {self.expertise}",
        ]
        if self.restrictions:
            parts.append(f"🚫 Ограничения: {', '.join(self.restrictions)}")
        if self.interests:
            parts.append(f"💡 Интересы: {', '.join(self.interests)}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_preset(cls, user_id: str, preset_key: str) -> "UserProfile":
        p = PRESET_PROFILES[preset_key]
        return cls(
            user_id=user_id,
            name="",
            style=p["style"],
            format=p["format"],
            expertise=p["expertise"],
            restrictions=list(p["restrictions"]),
            interests=list(p["interests"]),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )


# ─── ProfileManager ───────────────────────────────────────────────────────────
class ProfileManager:
    """Сохраняет и загружает профили пользователей (JSON-файлы)."""

    def __init__(self, profiles_dir: str = PROFILES_DIR):
        self.dir = profiles_dir
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, user_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(user_id))
        return os.path.join(self.dir, f"{safe}.json")

    def save(self, profile: UserProfile) -> None:
        profile.updated_at = datetime.now().isoformat()
        with open(self._path(profile.user_id), "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, user_id: str) -> UserProfile:
        path = self._path(user_id)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return UserProfile.from_dict(data)
        # Профиль не найден — создаём дефолтный
        profile = UserProfile(
            user_id=user_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        self.save(profile)
        return profile

    def delete(self, user_id: str) -> None:
        path = self._path(user_id)
        if os.path.exists(path):
            os.remove(path)


# ─── PersonalizedAgent ────────────────────────────────────────────────────────
class PersonalizedAgent:
    """
    Надстройка над MemoryAgent.
    Добавляет персонализированный system prompt из UserProfile + LTM.
    """

    def __init__(
        self,
        client: OpenAI,
        user_id: str,
        profile: Optional[UserProfile] = None,
        model: str = "deepseek-chat",
        stm_window: int = 10,
        persist_dir: str | None = None,
        demo_mode: bool = False,
    ):
        self.client = client
        self.user_id = user_id
        self.model = model
        self.demo_mode = demo_mode  # в demo_mode пропускаем LTM и факт-экстракцию

        self.stm = ShortTermMemory(window=stm_window)
        self.wm = WorkingMemory()
        # В demo_mode не инициализируем ChromaDB — быстрее
        self.ltm = None if demo_mode else LongTermMemory(user_id=user_id, persist_dir=persist_dir)

        self.profile_manager = ProfileManager()
        self.profile = profile or self.profile_manager.load(user_id)

        self._turn = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    # ── обновить имя из LTM ───────────────────────────────────────────────────
    def _sync_name_from_ltm(self) -> None:
        """Если профиль не имеет имени — ищем в LTM."""
        if self.profile.name or self.demo_mode or self.ltm is None:
            return
        facts = self.ltm.search("как зовут пользователя имя", n=3)
        for f in facts:
            # Простая эвристика: если факт содержит "зовут" или "имя"
            if any(w in f.lower() for w in ["зовут", "имя", "меня"]):
                self.profile.name = f
                break

    # ── построить system prompt ───────────────────────────────────────────────
    def _build_system(self, ltm_facts: list[str]) -> str:
        return self.profile.build_system_prompt(ltm_facts)

    # ── извлечь факты из сообщения ────────────────────────────────────────────
    def _extract_and_save_facts(self, user_message: str) -> int:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты — экстрактор фактов. Из сообщения пользователя извлеки "
                            "только факты о нём самом (имя, профессия, город, увлечения, "
                            "предпочтения, питомцы и т.п.). "
                            "Верни JSON: {\"facts\": [\"факт1\", \"факт2\"]}. "
                            "Если фактов нет — верни {\"facts\": []}."
                        ),
                    },
                    {"role": "user", "content": user_message},
                ],
                max_tokens=256,
                temperature=0,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            facts: list[str] = data.get("facts", [])
            for f in facts:
                self.ltm.save_fact(f)

            # Обновить интересы в профиле из LTM
            self._update_interests_from_facts(facts)
            return len(facts)
        except Exception:
            return 0

    def _update_interests_from_facts(self, facts: list[str]) -> None:
        """Добавляем новые темы в profile.interests если их ещё нет."""
        keywords = ["программирован", "разработ", "дизайн", "маркетинг",
                    "финансы", "спорт", "музыка", "игр", "путешеств"]
        for fact in facts:
            for kw in keywords:
                if kw in fact.lower():
                    topic = fact[:40]
                    if topic not in self.profile.interests:
                        self.profile.interests.append(topic)
        if facts:
            self.profile_manager.save(self.profile)

    # ── основной метод ────────────────────────────────────────────────────────
    def chat(self, user_message: str) -> tuple[str, MemoryTurnStats]:
        self._turn += 1
        self._sync_name_from_ltm()

        # 1. Загрузить релевантные факты из LTM (пропускаем в demo_mode)
        ltm_facts = self.ltm.search(user_message, n=5) if self.ltm is not None else []

        # 2. Персонализированный system prompt
        system = self._build_system(ltm_facts)

        # 3. Сформировать messages
        messages = [{"role": "system", "content": system}]
        messages += self.stm.get()
        messages.append({"role": "user", "content": user_message})

        # 4. LLM-вызов
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        reply = resp.choices[0].message.content or ""
        prompt_tokens = resp.usage.prompt_tokens
        completion_tokens = resp.usage.completion_tokens

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        # 5. Обновить STM
        self.stm.add("user", user_message)
        self.stm.add("assistant", reply)

        # 6. Извлечь и сохранить факты (пропускаем в demo_mode — нет LTM)
        facts_saved = 0 if self.demo_mode else self._extract_and_save_facts(user_message)

        stats = MemoryTurnStats(
            turn=self._turn,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ltm_facts_loaded=len(ltm_facts),
            ltm_facts_saved=facts_saved,
        )
        return reply, stats

    # ── утилиты ───────────────────────────────────────────────────────────────
    def reset_session(self) -> None:
        """Очистить STM и WM (LTM и профиль сохраняются)."""
        self.stm = ShortTermMemory(window=self.stm._window)
        self.wm.clear()
        self._turn = 0

    def forget_all(self) -> None:
        """Очистить LTM."""
        if self.ltm is not None:
            self.ltm.clear()

    def set_profile(self, profile: UserProfile) -> None:
        self.profile = profile
        self.profile_manager.save(profile)

    def stats_str(self) -> str:
        cost = (
            self._total_prompt_tokens / 1_000_000 * INPUT_PRICE_RUB_PER_1M
            + self._total_completion_tokens / 1_000_000 * OUTPUT_PRICE_RUB_PER_1M
        )
        return (
            f"Ходов: {self._turn} | "
            f"Токены: {self._total_prompt_tokens}↑ {self._total_completion_tokens}↓ | "
            f"LTM фактов: {self.ltm.count()} | "
            f"Стоимость: ~{cost:.4f} ₽"
        )


# ─── Demo scenarios ───────────────────────────────────────────────────────────
DEMO_INTRO_MESSAGES = [
    "Привет! Меня зовут Саша, я iOS-разработчик из Питера.",
    "Я люблю минималистичные ответы — без воды, только суть.",
]
