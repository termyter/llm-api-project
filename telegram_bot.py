#!/usr/bin/env python3
"""
Telegram бот для работы с LLM через API
Задание: отправляет запрос в LLM, получает ответ, выводит в интерфейс
"""

import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация OpenAI клиента (работает с DeepSeek API)
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com"
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    await update.message.reply_text(
        "👋 Привет! Я бот с LLM.\n\n"
        "Просто напиши мне любой вопрос, и я отвечу используя нейросеть!\n\n"
        "Команды:\n"
        "/start - это сообщение\n"
        "/help - помощь"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help"""
    await update.message.reply_text(
        "❓ Как использовать:\n\n"
        "1. Просто напиши мне любой вопрос\n"
        "2. Я отправлю его в LLM (DeepSeek)\n"
        "3. Получу ответ и пришлю тебе\n\n"
        "Пример: 'Что такое нейронная сеть?'"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений"""
    user_message = update.message.text
    user_name = update.effective_user.first_name

    logger.info(f"Получено сообщение от {user_name}: {user_message}")

    # Отправляем уведомление о начале обработки
    await update.message.reply_text("🤔 Думаю...")

    try:
        # Отправляем запрос в DeepSeek API
        response = openai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Ты полезный ассистент. Отвечай кратко и по делу."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500
        )

        # Получаем ответ
        ai_answer = response.choices[0].message.content

        # Отправляем ответ пользователю
        await update.message.reply_text(ai_answer)

        logger.info(f"Ответ отправлен. Токенов использовано: {response.usage.total_tokens}")

    except Exception as e:
        logger.error(f"Ошибка при обработке: {e}")
        await update.message.reply_text(
            "❌ Произошла ошибка при обработке запроса.\n"
            f"Детали: {str(e)}"
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка ошибок"""
    logger.error(f"Update {update} caused error {context.error}")


def main():
    """Запуск бота"""
    # Проверяем наличие токенов
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not bot_token:
        print("❌ Ошибка: TELEGRAM_BOT_TOKEN не найден в .env файле!")
        print("📝 Получите токен у @BotFather и добавьте в .env")
        return

    if not openai_key:
        print("❌ Ошибка: OPENAI_API_KEY не найден в .env файле!")
        print("📝 Добавьте ваш OpenAI ключ в .env")
        return

    # Создаем приложение
    application = Application.builder().token(bot_token).build()

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)

    # Запускаем бота
    print("🤖 Бот запущен! Нажмите Ctrl+C для остановки.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
