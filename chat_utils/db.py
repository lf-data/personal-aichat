import os
import json
import logging
import aiosqlite
import sqlite3

CHAT_MEMORY_DB = os.environ["CHAT_MEMORY_DB"]

_db: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    """
    Get the database connection, initializing it if necessary.
    Returns:
        aiosqlite.Connection: The database connection.
    """
    global _db
    if _db is None:
        _db = await aiosqlite.connect(CHAT_MEMORY_DB)
        _db.row_factory = aiosqlite.Row
        await _db.execute("PRAGMA foreign_keys = ON;")
        await init_db(_db)
    return _db


async def init_db(db: aiosqlite.Connection) -> None:
    """
    Initialize the database with required tables if they do not exist.
    Args:
        db (aiosqlite.Connection): The database connection.
    """
    await db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            mail TEXT,
            name TEXT,
            surname TEXT,
            role TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            user_id TEXT PRIMARY KEY,
            messages TEXT,         -- JSON list
            telegram_msgs TEXT,    -- JSON list
            memories TEXT,         -- JSON list
            shopping_lists TEXT,   -- JSON dict
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )
    await db.commit()

def init_db_sync():
    """
    Inizialize database in a synchronous way.
    """
    db = sqlite3.connect(CHAT_MEMORY_DB)

    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            mail TEXT,
            name TEXT,
            surname TEXT,
            role TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            user_id TEXT PRIMARY KEY,
            messages TEXT,         -- JSON list
            telegram_msgs TEXT,    -- JSON list
            memories TEXT,         -- JSON list
            shopping_lists TEXT,   -- JSON dict
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )

    db.commit()
    db.close()

def add_user_sync(user_id, mail, name, surname, role):
    """
    Non-async version of add_user.
    Args:
        user_id: str
        mail: str
        name: str
        surname: str
        role: str
    """
    init_db_sync()
    conn = sqlite3.connect(CHAT_MEMORY_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO users (id, mail, name, surname, role)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            mail=excluded.mail,
            name=excluded.name,
            surname=excluded.surname,
            role=excluded.role;
        """,
        (user_id, mail, name, surname, role),
    )

    conn.commit()
    conn.close()

async def add_user(user_id, mail, name, surname, role):
    """
    Add a new user or update an existing one.
    Args:
        user_id: str
        mail: str
        name: str
        surname: str
        role: str
    """
    db = await get_db()
    data = (user_id, mail, name, surname, role)

    # UPSERT
    await db.execute(
        """
        INSERT INTO users (id, mail, name, surname, role)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            mail=excluded.mail,
            name=excluded.name,
            surname=excluded.surname,
            role=excluded.role;
        """,
        data,
    )
    await db.commit()


async def check_user_exist(user_id) -> bool:
    """
    Check if a user exists in the database.
    Args:
        user_id: str
    Returns:
        bool: True if the user exists, False otherwise.
    """
    db = await get_db()
    cur = await db.execute("SELECT 1 FROM users WHERE id = ? LIMIT 1;", (user_id,))
    row = await cur.fetchone()
    return row is not None


async def user_is_admin(user_id) -> bool:
    """
    Check if a user has admin role.
    Args:
        user_id: str
    Returns:
        bool: True if the user is admin, False otherwise.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT role FROM users WHERE id = ? LIMIT 1;",
        (user_id,),
    )
    row = await cur.fetchone()
    if not row:
        return False
    role = row["role"]
    return role == "admin"


def _load_json(value, default):
    """
    Load JSON from a string, returning a default value on failure.
    Args:
        value (str): The JSON string.
        default: The default value to return on failure.
    Returns:
        The parsed JSON object or the default value."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except Exception:
        logging.exception("Errore nel parsing JSON, uso default.")
        return default


def _dump_json(value) -> str:
    """
    Dump a value to a JSON string.
    Args:
        value: The value to dump.
    Returns:
        str: The JSON string.
    """
    return json.dumps(value, ensure_ascii=False)


async def _ensure_messages_row(db: aiosqlite.Connection, user_id: str):
    """
    Ensure that a messages row exists for the given user_id.
    Args:
        db (aiosqlite.Connection): The database connection.
        user_id (str): The user ID.
    """
    cur = await db.execute(
        "SELECT 1 FROM messages WHERE user_id = ? LIMIT 1;",
        (user_id,),
    )
    row = await cur.fetchone()
    if row is None:
        await db.execute(
            """
            INSERT INTO messages (user_id, messages, telegram_msgs, memories, shopping_lists)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                user_id,
                _dump_json([]),
                _dump_json([]),
                _dump_json([]),
                _dump_json({}),
            ),
        )
        await db.commit()


async def create_user_if_not_exists(user_id):
    """
    Create a messages row for the user if it does not exist.
    Args:
        user_id (str): The user ID.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)


async def add_permanent_memory(user_id, memory):
    """
    Add a permanent memory for the given user.
    Args:
        user_id (str): The user ID.
        memory (str): The memory to add.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)

    cur = await db.execute(
        "SELECT memories FROM messages WHERE user_id = ?;",
        (user_id,),
    )
    row = await cur.fetchone()
    memories = _load_json(row["memories"], [])

    memories.append(memory)

    await db.execute(
        "UPDATE messages SET memories = ? WHERE user_id = ?;",
        (_dump_json(memories), user_id),
    )
    await db.commit()


async def reset_permanent_memories(user_id):
    """
    Reset permanent memories for the given user.
    Args:
        user_id (str): The user ID.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)
    await db.execute(
        "UPDATE messages SET memories = ? WHERE user_id = ?;",
        (_dump_json([]), user_id),
    )
    await db.commit()


async def get_permanent_memories(user_id):
    """
    Get permanent memories for the given user.
    Args:
        user_id (str): The user ID.
    Returns:
        list: The list of permanent memories.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)

    cur = await db.execute(
        "SELECT memories FROM messages WHERE user_id = ?;",
        (user_id,),
    )
    row = await cur.fetchone()
    return _load_json(row["memories"], [])


async def get_shopping_list(user_id):
    """
    Get shopping list for the given user.
    Args:
        user_id (str): The user ID.
    Returns:
        dict: The shopping list.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)

    cur = await db.execute(
        "SELECT shopping_lists FROM messages WHERE user_id = ?;",
        (user_id,),
    )
    row = await cur.fetchone()
    return _load_json(row["shopping_lists"], {})


async def add_or_update_shopping_list(user_id, list_name, product_id, quantity, metric):
    """
    Add or update a shopping list item for the given user.
    Args:
        user_id (str): The user ID.
        list_name (str): The name of the product.
        product_id (str): The product ID.
        quantity (float): The quantity of the product.
        metric (str): The metric of the product.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)

    cur = await db.execute(
        "SELECT shopping_lists FROM messages WHERE user_id = ?;",
        (user_id,),
    )
    row = await cur.fetchone()
    shopping_lists = _load_json(row["shopping_lists"], {})

    shopping_lists[product_id] = {"product_name": list_name, "quantity": quantity, "metric": metric}

    await db.execute(
        "UPDATE messages SET shopping_lists = ? WHERE user_id = ?;",
        (_dump_json(shopping_lists), user_id),
    )
    await db.commit()


async def delete_shopping_list(user_id, product_id):
    """
    Delete a shopping list item for the given user.
    Args:
        user_id (str): The user ID.
        product_id (str): The product ID to delete.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)

    cur = await db.execute(
        "SELECT shopping_lists FROM messages WHERE user_id = ?;",
        (user_id,),
    )
    row = await cur.fetchone()
    shopping_lists = _load_json(row["shopping_lists"], {})

    if product_id in shopping_lists:
        del shopping_lists[product_id]
        await db.execute(
            "UPDATE messages SET shopping_lists = ? WHERE user_id = ?;",
            (_dump_json(shopping_lists), user_id),
        )
        await db.commit()


async def get_user_memory(user_id):
    """
    Get user memory messages for the given user.
    Args:
        user_id (str): The user ID.
    Returns:
        list: The list of user memory messages.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)

    cur = await db.execute(
        "SELECT messages FROM messages WHERE user_id = ?;",
        (user_id,),
    )
    row = await cur.fetchone()
    return _load_json(row["messages"], [])


async def save_user_memory(user_id, messages):
    """
    Save user memory messages for the given user.
    Args:
        user_id (str): The user ID.
        messages (list): The list of messages to save.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)

    await db.execute(
        "UPDATE messages SET messages = ? WHERE user_id = ?;",
        (_dump_json(messages), user_id),
    )
    await db.commit()


async def reset_user_memory(user_id):
    """
    Reset user memory messages for the given user.
    Args:
        user_id (str): The user ID.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)

    await db.execute(
        "UPDATE messages SET messages = ? WHERE user_id = ?;",
        (_dump_json([]), user_id),
    )
    await db.commit()

async def save_user_message(user_id, message_id):
    """
    Save a Telegram message ID for the given user.
    Args:
        user_id (str): The user ID.
        message_id (int): The Telegram message ID to save.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)

    cur = await db.execute(
        "SELECT telegram_msgs FROM messages WHERE user_id = ?;",
        (user_id,),
    )
    row = await cur.fetchone()
    msg_ids = _load_json(row["telegram_msgs"], [])

    if message_id not in msg_ids:
        msg_ids.append(message_id)
        await db.execute(
            "UPDATE messages SET telegram_msgs = ? WHERE user_id = ?;",
            (_dump_json(msg_ids), user_id),
        )
        await db.commit()


async def delete_user_message(user_id, message, context, messages_id):
    """
    Delete Telegram messages for the given user.
    Args:
        user_id (str): The user ID.
        message: The Telegram message object.
        context: The Telegram context object.
        messages_id (list): The list of Telegram message IDs to delete.
    """
    # first delete from Telegram
    for mid in messages_id:
        if mid > message.message_id:
            try:
                await context.bot.delete_message(
                    chat_id=message.chat_id, message_id=mid
                )
            except Exception:
                pass

    # update DB
    db = await get_db()
    await _ensure_messages_row(db, user_id)

    cur = await db.execute(
        "SELECT telegram_msgs FROM messages WHERE user_id = ?;",
        (user_id,),
    )
    row = await cur.fetchone()
    msg_ids = _load_json(row["telegram_msgs"], [])

    for mid in messages_id:
        if mid in msg_ids:
            msg_ids.remove(mid)

    await db.execute(
        "UPDATE messages SET telegram_msgs = ? WHERE user_id = ?;",
        (_dump_json(msg_ids), user_id),
    )
    await db.commit()


async def get_user_messages(user_id):
    """
    Get Telegram message IDs for the given user.
    Args:
        user_id (str): The user ID.
    Returns:
        list: The list of Telegram message IDs.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)

    cur = await db.execute(
        "SELECT telegram_msgs FROM messages WHERE user_id = ?;",
        (user_id,),
    )
    row = await cur.fetchone()
    return _load_json(row["telegram_msgs"], [])


async def reset_user_messages(user_id):
    """
    Reset Telegram message IDs for the given user.
    Args:
        user_id (str): The user ID.
    """
    db = await get_db()
    await _ensure_messages_row(db, user_id)

    await db.execute(
        "UPDATE messages SET telegram_msgs = ? WHERE user_id = ?;",
        (_dump_json([]), user_id),
    )
    await db.commit()
