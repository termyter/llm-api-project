# Модуль для работы с памятью агента: хранение и получение истории диалогов
# через SQLite базу данных (aiosqlite для async I/O).

from __future__ import annotations

import aiosqlite
from datetime import datetime
from pathlib import Path
import yaml


def get_db_path() -> str:
    """Читает путь к базе данных из config.yaml.

    Returns:
        Путь к файлу SQLite базы данных.
    """
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["memory"]["db_path"]


async def init_db() -> None:
    """Инициализирует базу данных: создаёт директорию и таблицу messages если не существует.

    Создаёт директорию myai/data/ если она отсутствует, затем создаёт таблицу
    для хранения сообщений истории диалогов.
    """
    db_path = get_db_path()
    # Создаём директорию если не существует
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


async def save_message(
    role: str,
    content: str,
    session_id: str = "default",
) -> None:
    """Сохраняет сообщение в историю диалога.

    Args:
        role: Роль отправителя ("user", "assistant", "system" и т.д.).
        content: Текст сообщения.
        session_id: Идентификатор сессии. По умолчанию "default".
    """
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        await db.commit()


async def get_history(
    session_id: str = "default",
    limit: int = 20,
) -> list[dict]:
    """Возвращает историю диалога для указанной сессии.

    Args:
        session_id: Идентификатор сессии. По умолчанию "default".
        limit: Максимальное количество сообщений. По умолчанию 20.

    Returns:
        Список словарей вида [{"role": "user", "content": "..."}],
        отсортированных по времени от старых к новым.
    """
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC, id ASC
            LIMIT ?
            """,
            (session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


async def clear_session(session_id: str = "default") -> None:
    """Удаляет все сообщения указанной сессии из базы данных.

    Args:
        session_id: Идентификатор сессии для очистки. По умолчанию "default".
    """
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (session_id,),
        )
        await db.commit()


async def list_sessions() -> list[str]:
    """Возвращает список всех уникальных идентификаторов сессий.

    Returns:
        Список уникальных session_id, отсортированных по алфавиту.
    """
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT DISTINCT session_id FROM messages ORDER BY session_id ASC"
        ) as cursor:
            rows = await cursor.fetchall()
    return [row[0] for row in rows]
