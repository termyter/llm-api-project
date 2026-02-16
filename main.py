#!/usr/bin/env python3
"""
Минимальный скрипт для работы с LLM через API
Задание: отправить запрос, получить ответ, вывести в консоль
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Загружаем переменные окружения из .env файла
load_dotenv()

def main():
    # Получаем API ключ из переменных окружения
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ Ошибка: API ключ не найден!")
        print("📝 Создайте файл .env и добавьте: OPENAI_API_KEY=your_key")
        return

    # Инициализируем клиент OpenAI
    client = OpenAI(api_key=api_key)

    print("🚀 Отправляем запрос в LLM...")

    try:
        # Отправляем запрос в API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Привет! Расскажи кратко, что такое нейронная сеть?"}
            ]
        )

        # Получаем ответ
        answer = response.choices[0].message.content

        # Выводим в консоль
        print("\n✅ Ответ получен:\n")
        print(answer)
        print("\n" + "="*50)
        print(f"Использовано токенов: {response.usage.total_tokens}")

    except Exception as e:
        print(f"❌ Ошибка при запросе: {e}")

if __name__ == "__main__":
    main()
