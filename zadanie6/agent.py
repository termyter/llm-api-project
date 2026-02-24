"""
День 6 — Простой агент с сохранением контекста.

Agent — отдельная сущность, которая:
- хранит историю диалога (messages[])
- инкапсулирует логику запроса/ответа к LLM
- позволяет вести многоходовой разговор
"""

from openai import OpenAI


class Agent:
    """
    LLM-агент с памятью диалога.

    Использование:
        agent = Agent(client, model_id="deepseek-chat")
        answer, tokens = agent.chat("Привет!")
        answer2, tokens2 = agent.chat("Что ты сказал?")  # помнит предыдущее
    """

    def __init__(
        self,
        client: OpenAI,
        model_id: str = "deepseek-chat",
        system_prompt: str = "Ты полезный ассистент. Отвечай по делу.",
    ):
        self.client = client
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.history: list[dict] = []  # [{role, content}, ...]

    def chat(self, user_message: str) -> tuple[str, int]:
        """
        Отправить сообщение агенту и получить ответ.
        История диалога сохраняется автоматически.

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

        return answer, tokens

    def reset(self):
        """Сбросить историю диалога (начать новый чат)."""
        self.history = []

    @property
    def turn_count(self) -> int:
        """Количество завершённых обменов (вопрос + ответ)."""
        return len(self.history) // 2
