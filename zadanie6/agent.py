"""
День 6/7 — Агент с сохранением контекста между запусками.

Agent — отдельная сущность, которая:
- хранит историю диалога (messages[])
- инкапсулирует логику запроса/ответа к LLM
- сохраняет историю в JSON-файл после каждого обмена
- загружает историю при создании (переживает перезапуск)
"""

import json
import os
from openai import OpenAI


class Agent:
    """
    LLM-агент с персистентной памятью диалога.

    История сохраняется в data/history_{user_id}.json
    и загружается автоматически при создании агента.

    Использование:
        agent = Agent(client, model_id="deepseek-chat", user_id=12345)
        answer, tokens = agent.chat("Привет!")
        # перезапуск...
        agent2 = Agent(client, model_id="deepseek-chat", user_id=12345)
        answer2, tokens2 = agent2.chat("Что ты сказал?")  # помнит!
    """

    def __init__(
        self,
        client: OpenAI,
        model_id: str = "deepseek-chat",
        system_prompt: str = "Ты полезный ассистент. Отвечай по делу.",
        user_id: int | None = None,
        data_dir: str = "data",
    ):
        self.client = client
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.user_id = user_id
        self.data_dir = data_dir
        self.history: list[dict] = self._load()

    # ── Персистентность ──────────────────────────────────────────────

    def _history_path(self) -> str | None:
        if self.user_id is None:
            return None
        return os.path.join(self.data_dir, f"history_{self.user_id}.json")

    def _load(self) -> list[dict]:
        """Загрузить историю из JSON-файла. Если файла нет — пустой список."""
        path = self._history_path()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        """Сохранить историю в JSON-файл."""
        path = self._history_path()
        if path is None:
            return
        os.makedirs(self.data_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    # ── Основные методы ──────────────────────────────────────────────

    def chat(self, user_message: str) -> tuple[str, int]:
        """
        Отправить сообщение агенту и получить ответ.
        История сохраняется в файл автоматически после каждого обмена.

        Returns:
            (ответ, кол-во токенов)
        """
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
        )

        answer = resp.choices[0].message.content
        tokens = resp.usage.total_tokens

        self.history.append({"role": "assistant", "content": answer})
        self._save()

        return answer, tokens

    def reset(self) -> None:
        """Сбросить историю диалога и удалить файл."""
        self.history = []
        self._save()

    @property
    def turn_count(self) -> int:
        """Количество завершённых обменов (вопрос + ответ)."""
        return len(self.history) // 2

    @property
    def is_restored(self) -> bool:
        """True если история была загружена из файла (не пустая при старте)."""
        return self.turn_count > 0
