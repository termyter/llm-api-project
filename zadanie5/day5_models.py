#!/usr/bin/env python3
"""
День 5: Сравнение версий моделей
Один и тот же запрос на слабой, средней и сильной моделях через RouterAI (routerai.ru).
Замеряем: время ответа, токены, стоимость, качество.
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ROUTERAI_API_KEY"),
    base_url="https://routerai.ru/api/v1"
)

# Один запрос для всех моделей
TASK = (
    "Объясни простыми словами, как работает нейронная сеть. "
    "Приведи пример из реальной жизни."
)

MODELS = [
    {
        "id": "amazon/nova-micro-v1",
        "label": "Слабая",
        "desc": "Amazon Nova Micro (лёгкая, только текст, дешёвая)"
    },
    {
        "id": "deepseek/deepseek-chat",
        "label": "Средняя",
        "desc": "DeepSeek V3 (671B MoE, хорошее качество)"
    },
    {
        "id": "anthropic/claude-opus-4.6",
        "label": "Сильная",
        "desc": "Claude Opus 4.6 (самая мощная модель Anthropic)"
    }
]

# Стоимость за 1M токенов (в рублях), из routerai.ru/models
PRICES_RUB = {
    "amazon/nova-micro-v1":       {"input": 9.0,   "output": 29.0},
    "deepseek/deepseek-chat":     {"input": 25.0,  "output": 37.0},
    "anthropic/claude-opus-4.6":  {"input": 498.0, "output": 2494.0},
}


def query_model(model_id):
    """Отправляет запрос и возвращает результат с замером времени."""
    start = time.time()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": TASK}]
    )
    elapsed = time.time() - start

    answer = response.choices[0].message.content
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0

    price = PRICES_RUB.get(model_id, {"input": 0.0, "output": 0.0})
    cost_rub = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

    return {
        "answer": answer,
        "elapsed": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_rub": cost_rub
    }


def main():
    print("=" * 65)
    print("🤖  ДЕНЬ 5: СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 65)
    print(f"\n📝 Запрос: {TASK}\n")

    results = []

    for model in MODELS:
        print("─" * 65)
        print(f"\n📌 {model['label'].upper()}: {model['desc']}")
        print(f"   Модель: {model['id']}\n")

        result = query_model(model["id"])

        print(f"📥 Ответ:")
        print(result["answer"])
        print(f"\n⏱️  Время ответа: {result['elapsed']:.2f} сек")
        print(f"📊 Токены: {result['input_tokens']} вход / {result['output_tokens']} выход / {result['total_tokens']} всего")
        print(f"💰 Стоимость: {result['cost_rub']:.4f} ₽")

        results.append({**model, **result})

    # --- Сводная таблица ---
    print("\n" + "=" * 65)
    print("📈 СВОДНАЯ ТАБЛИЦА:")
    print(f"  {'Модель':<10} {'Время':>8} {'Токены':>8} {'Стоимость':>12}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*12}")
    for r in results:
        print(f"  {r['label']:<10} {r['elapsed']:>7.2f}с {r['total_tokens']:>8} {r['cost_rub']:>10.4f} ₽")

    # --- Автоматический анализ ---
    print("\n" + "=" * 65)
    print("🤖 АВТОМАТИЧЕСКИЙ АНАЛИЗ: Claude сравнивает все ответы")
    print("=" * 65 + "\n")

    analysis = client.chat.completions.create(
        model="anthropic/claude-opus-4.6",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты — аналитик качества LLM. Оцени три ответа на один вопрос "
                    "от разных моделей (слабой, средней, сильной). "
                    "Сравни по критериям:\n"
                    "1. Полнота — всё ли объяснено?\n"
                    "2. Точность — нет ли ошибок?\n"
                    "3. Понятность — легко ли читать?\n"
                    "4. Пример — насколько хорош пример из реальной жизни?\n\n"
                    "Дай оценку каждой модели и итоговую рекомендацию: "
                    "когда стоит использовать слабую, а когда сильную модель."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {TASK}\n\n"
                    f"--- СЛАБАЯ модель ({results[0]['id']}) ---\n{results[0]['answer']}\n\n"
                    f"--- СРЕДНЯЯ модель ({results[1]['id']}) ---\n{results[1]['answer']}\n\n"
                    f"--- СИЛЬНАЯ модель ({results[2]['id']}) ---\n{results[2]['answer']}"
                )
            }
        ]
    )

    print(analysis.choices[0].message.content)

    analysis_tokens = analysis.usage.total_tokens if analysis.usage else 0
    analysis_cost = (analysis_tokens * 2494.0) / 1_000_000
    print(f"\n📊 Токенов на анализ: {analysis_tokens} (≈ {analysis_cost:.4f} ₽)")
    print("=" * 65)


if __name__ == "__main__":
    main()
