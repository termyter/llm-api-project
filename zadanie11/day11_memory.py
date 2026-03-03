"""
Day 11: Assistant Memory Model
3 explicit memory layers:

  ShortTermMemory  — last N messages of current dialog (in-memory list)
  WorkingMemory    — entities for current task (in-memory dict, reset per session)
  LongTermMemory   — persistent facts about user (ChromaDB local SQLite)

MemoryAgent combines all three: reads from LTM on startup, writes to LTM after
each turn, uses working memory to track current task context.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any
from openai import OpenAI

# ─── pricing ────────────────────────────────────────────────────────────────
INPUT_PRICE_RUB_PER_1M  = 25.0
OUTPUT_PRICE_RUB_PER_1M = 37.0

SYSTEM_PROMPT = (
    "Ты — персональный ассистент с долгосрочной памятью. "
    "Ты помнишь факты о пользователе между сессиями. "
    "Если знаешь что-то о пользователе — используй это в разговоре. "
    "Отвечай по-русски, кратко и по делу."
)

# Demo scenarios
DEMO_INTRO = [
    "Привет! Меня зовут Алексей, я iOS-разработчик из Москвы.",
    "У меня есть питомец — кот Борис, рыжий.",
    "Я сейчас работаю над приложением для трекинга привычек.",
]

DEMO_RECALL = [
    "Как меня зовут?",
    "Расскажи про моего питомца.",
    "Что я разрабатываю?",
]


# ─── stats ───────────────────────────────────────────────────────────────────
@dataclass
class MemoryTurnStats:
    turn: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ltm_facts_loaded: int = 0
    ltm_facts_saved: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_rub(self) -> float:
        inp = self.prompt_tokens / 1_000_000 * INPUT_PRICE_RUB_PER_1M
        out = self.completion_tokens / 1_000_000 * OUTPUT_PRICE_RUB_PER_1M
        return round(inp + out, 4)


# ─── Layer 1: ShortTermMemory ─────────────────────────────────────────────────
class ShortTermMemory:
    """Keeps last N messages of the current dialog session."""

    def __init__(self, window: int = 10):
        self._window = window
        self._messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def get(self) -> list[dict]:
        return self._messages[-self._window:]

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


# ─── Layer 2: WorkingMemory ───────────────────────────────────────────────────
class WorkingMemory:
    """Tracks entities relevant to the current task. Resets per session."""

    def __init__(self):
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def all(self) -> dict:
        return dict(self._store)

    def clear(self) -> None:
        self._store.clear()

    def as_context_str(self) -> str:
        if not self._store:
            return ""
        lines = [f"  • {k}: {v}" for k, v in self._store.items()]
        return "Текущий контекст задачи:\n" + "\n".join(lines)


# ─── Layer 3: LongTermMemory ──────────────────────────────────────────────────
class LongTermMemory:
    """
    Persistent facts about user stored in ChromaDB (local SQLite backend).
    Each fact is embedded and retrieved by semantic similarity.
    """

    def __init__(self, user_id: str, persist_dir: str = ".chroma_db"):
        import chromadb
        self._user_id = str(user_id)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=f"user_{self._user_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def save_fact(self, fact: str) -> str:
        """Store a fact. Returns the fact id."""
        fact_id = str(uuid.uuid4())
        self._collection.add(
            documents=[fact],
            ids=[fact_id],
            metadatas=[{"user_id": self._user_id}],
        )
        return fact_id

    def search(self, query: str, n: int = 5) -> list[str]:
        """Find semantically similar facts for a given query."""
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n, count),
        )
        return results["documents"][0] if results["documents"] else []

    def all_facts(self) -> list[str]:
        """Return all stored facts (for display)."""
        if self._collection.count() == 0:
            return []
        results = self._collection.get()
        return results["documents"]

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        """Delete all facts for this user."""
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)


# ─── extract facts from assistant response ────────────────────────────────────
def _extract_facts(client: OpenAI, user_message: str, model: str = "deepseek-chat") -> list[str]:
    """
    Ask LLM to extract storable facts from user's message.
    Returns list of short fact strings (may be empty).
    """
    prompt = (
        "Из сообщения пользователя извлеки факты, которые стоит запомнить "
        "(имя, профессия, город, питомцы, проекты, предпочтения и т.п.). "
        "Верни JSON-массив коротких строк-фактов на русском. "
        "Если фактов нет — верни пустой массив [].\n\n"
        f"Сообщение: {user_message}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        # handle both {"facts": [...]} and plain [...]
        if isinstance(data, list):
            return [str(f) for f in data if f]
        for v in data.values():
            if isinstance(v, list):
                return [str(f) for f in v if f]
    except Exception:
        pass
    return []


# ─── MemoryAgent ──────────────────────────────────────────────────────────────
class MemoryAgent:
    """
    Combines all three memory layers:
      - ShortTermMemory for recent dialog
      - WorkingMemory for current task context
      - LongTermMemory for persistent user facts (ChromaDB)
    """

    def __init__(
        self,
        client: OpenAI,
        user_id: str | int,
        model: str = "deepseek-chat",
        stm_window: int = 10,
        persist_dir: str = ".chroma_db",
    ):
        self._client = client
        self._model = model
        self.stm = ShortTermMemory(window=stm_window)
        self.wm = WorkingMemory()
        self.ltm = LongTermMemory(user_id=str(user_id), persist_dir=persist_dir)
        self.stats: list[MemoryTurnStats] = []

    def chat(self, user_message: str) -> tuple[str, MemoryTurnStats]:
        """
        Process one user turn:
        1. Search LTM for relevant facts
        2. Build system message with LTM facts + WM context
        3. Call LLM with STM (recent history)
        4. Extract + save new facts to LTM
        5. Update STM
        """
        turn_num = len(self.stats) + 1
        stat = MemoryTurnStats(turn=turn_num)

        # 1. Retrieve relevant facts from LTM
        relevant_facts = self.ltm.search(user_message, n=5)
        stat.ltm_facts_loaded = len(relevant_facts)

        # 2. Build augmented system prompt
        system_parts = [SYSTEM_PROMPT]
        if relevant_facts:
            facts_str = "\n".join(f"  • {f}" for f in relevant_facts)
            system_parts.append(f"\nЧто я знаю о тебе:\n{facts_str}")
        wm_ctx = self.wm.as_context_str()
        if wm_ctx:
            system_parts.append(f"\n{wm_ctx}")
        system_content = "\n".join(system_parts)

        # 3. Build messages: system + STM history + new user message
        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.stm.get())
        messages.append({"role": "user", "content": user_message})

        # 4. LLM call
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.7,
        )
        assistant_reply = resp.choices[0].message.content or ""
        usage = resp.usage
        stat.prompt_tokens = usage.prompt_tokens
        stat.completion_tokens = usage.completion_tokens

        # 5. Extract + save new facts
        new_facts = _extract_facts(self._client, user_message, self._model)
        for fact in new_facts:
            self.ltm.save_fact(fact)
        stat.ltm_facts_saved = len(new_facts)

        # 6. Update STM
        self.stm.add("user", user_message)
        self.stm.add("assistant", assistant_reply)

        self.stats.append(stat)
        return assistant_reply, stat

    def reset_session(self) -> None:
        """Clear STM and WM (keep LTM — it's persistent)."""
        self.stm.clear()
        self.wm.clear()
        self.stats.clear()

    def forget_all(self) -> None:
        """Clear everything including LTM."""
        self.reset_session()
        self.ltm.clear()
