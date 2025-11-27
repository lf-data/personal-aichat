from dotenv import load_dotenv

load_dotenv()

import logging
import os
from functools import wraps
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from langchain.agents import create_agent
from langchain_core.messages import messages_to_dict, messages_from_dict, HumanMessage
import telegramify_markdown
from chat_utils.prompt import SYSTEM_PROMPT
from chat_utils.utils import (
    build_content_blocks,
    split_markdown,
    init_db,
    is_valid_email,
)

from tool import load_tools
from chat_utils.openai_provider import model
from chat_utils.db import (
    get_user_memory,
    save_user_memory,
    reset_user_memory,
    save_user_message,
    delete_user_message,
    get_user_messages,
    reset_user_messages,
    create_user_if_not_exists,
    check_user_exist,
    user_is_admin,
    add_user,
)
from time import time
import json

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "AiChat")
TELEGRAM_CONNECTION_TIMEOUT = int(os.getenv("TELEGRAM_CONNECTION_TIMEOUT", "60"))
TELEGRAM_POOL_TIMEOUT = int(os.getenv("TELEGRAM_POOL_TIMEOUT", "60"))
TELEGRAM_READ_TIMEOUT = int(os.getenv("TELEGRAM_READ_TIMEOUT", "60"))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "30"))
DEBUG = int(os.getenv("DEBUG_TELEGRAM", "0"))

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


# --- Decorator auth ---
def authorized(func):
    """
    Decorator to check if the user is authorized to use the bot.
    If not authorized, sends a message to the user.
    """
    @wraps(func)
    async def wrapper(
        update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs
    ):
        try:
            user_id = update.effective_user.id
            if not await check_user_exist(user_id):
                await update.effective_message.reply_text(
                    "❌ Questo bot è ad accesso limitato e non sei autorizzato a utilizzarlo.\n"
                    "I messaggi che invii non vengono inoltrati all'intelligenza artificiale "
                    "e sono usati solo per verificare l'accesso.\n"
                    "Ti chiediamo di non usare questo bot."
                )
                return
            await create_user_if_not_exists(user_id)
            return await func(update, context, *args, **kwargs)
        except Exception as e:
            logging.error(
                f"Si è verificato un errore nella fase di autenticazione dell'utente: {e}",
                exc_info=DEBUG,
            )
            sent_msg = await update.effective_message.reply_text(
                "❌ Si è verificato un errore nella fase di autenticazione dell'utente."
            )
            await save_user_message(update.effective_user.id, sent_msg.message_id)
            if await user_is_admin(update.effective_user.id):
                sent_msg_error = await update.effective_message.reply_text(str(e))
                await save_user_message(
                    update.effective_user.id, sent_msg_error.message_id
                )

    return wrapper


def authorized_admin(func):
    """
    Decorator to check if the user is an admin.
    If not an admin, sends a message to the user.
    """
    @wraps(func)
    async def wrapper(
        update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs
    ):
        try:
            user_id = update.effective_user.id
            if not await check_user_exist(user_id):
                await update.effective_message.reply_text(
                    "❌ Questo bot è ad accesso limitato e non sei autorizzato a utilizzarlo.\n"
                    "I messaggi che invii non vengono inoltrati all'intelligenza artificiale "
                    "e sono usati solo per verificare l'accesso.\n"
                    "Ti chiediamo di non usare questo bot."
                )
                return

            if not await user_is_admin(user_id):
                await update.effective_message.reply_text(
                    "❌ Non sei autorizzato ad usare questo comando.\nSolo gli utenti amministratori possono aggiungere nuovi utenti."
                )
                return
            await create_user_if_not_exists(user_id)
            return await func(update, context, *args, **kwargs)
        except Exception as e:
            logging.error(
                f"Si è verificato un errore nella fase di autenticazione dell'utente: {e}",
                exc_info=DEBUG,
            )
            sent_msg = await update.effective_message.reply_text(
                "❌ Si è verificato un errore nella fase di autenticazione dell'utente."
            )
            await save_user_message(update.effective_user.id, sent_msg.message_id)
            if await user_is_admin(update.effective_user.id):
                sent_msg_error = await update.effective_message.reply_text(str(e))
                await save_user_message(
                    update.effective_user.id, sent_msg_error.message_id
                )

    return wrapper


async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Provide information about the bot, data handling, and disclaimers.
    Args:
        update (Update): The Telegram update object.
        context (ContextTypes.DEFAULT_TYPE): The context of the Telegram bot.
    """
    try:
        await save_user_message(user_id, update.effective_message.message_id)
        user_id = update.effective_user.id
        if not check_user_exist(user_id):
            sent_msg = await update.effective_message.reply_text(
                "❌ Questo bot è ad accesso limitato e non sei autorizzato a utilizzarlo.\n"
                "I messaggi che invii non vengono inoltrati all'intelligenza artificiale "
                "e sono usati solo per verificare l'accesso.\n"
                "Ti chiediamo di non usare questo bot."
            )
        else:
            sent_msg = await update.effective_message.reply_text(
                "Questo bot utilizza un servizio di intelligenza artificiale per generare le risposte.\n"
                "Le risposte possono contenere errori e non sostituiscono il parere di un professionista.\n"
                "Non inviare dati sensibili (salute, password, dati bancari, documenti, ecc.).\n"
                "Il bot è ad uso privato/sperimentale e l'accesso è limitato a utenti autorizzati."
            )
        await save_user_message(user_id, sent_msg.message_id)
    except Exception as e:
        logging.error(
            f"Si è verificato un errore nella fase di verifica dell'esistenza dell'utente: {e}",
            exc_info=DEBUG,
        )
        sent_msg = await update.effective_message.reply_text(
            "❌ Si è verificato un errore nella fase di verifica dell'esistenza dell'utente"
        )
        await save_user_message(update.effective_user.id, sent_msg.message_id)
        if await user_is_admin(update.effective_user.id):
            sent_msg_error = await update.effective_message.reply_text(str(e))
            await save_user_message(update.effective_user.id, sent_msg_error.message_id)


@authorized
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Initialize a new chat session for the user.
    Args:
        update (Update): The Telegram update object.
        context (ContextTypes.DEFAULT_TYPE): The context of the Telegram bot.
    """
    try:
        await reset_user_memory(update.effective_user.id)
        await reset_user_messages(update.effective_user.id)
        await save_user_message(
            update.effective_user.id, update.effective_message.message_id
        )  # save user message
        msg = await update.effective_message.reply_text(
            f"Ciao! Sono {ASSISTANT_NAME}, il tuo assistente personale 🤖\n"
            "Puoi farmi domande liberamente.\n"
            "Per sapere come funziona il bot, come vengono trattati i dati "
            "e alcune avvertenze importanti, usa il comando /info."
        )
        await save_user_message(
            update.effective_user.id, msg.message_id
        )  # salva msg bot
    except Exception as e:
        logging.error(
            f"Si è verificato un errore nella fase di inizializzazione del bot: {e}",
            exc_info=DEBUG,
        )
        sent_msg = await update.effective_message.reply_text(
            "❌ Si è verificato un errore nella fase di inizializzazione del bot"
        )
        await save_user_message(update.effective_user.id, sent_msg.message_id)
        if await user_is_admin(update.effective_user.id):
            sent_msg_error = await update.effective_message.reply_text(str(e))
            await save_user_message(update.effective_user.id, sent_msg_error.message_id)


@authorized_admin
async def add_user_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Add a new user to the bot.
    Args:
        update (Update): The Telegram update object.
        context (ContextTypes.DEFAULT_TYPE): The context of the Telegram bot.
    """
    try:
        txt = update.effective_message.text
        txt = txt.replace("/new_user", "").replace("new_user", "").strip()
        values = txt.split("-")
        values = [x.strip() for x in values]
        if len(values) == 5:
            if not is_valid_email(values[1]):
                raise Exception("la mail inserita non è corretta")
            if values[4] not in ["admin", "user"]:
                raise Exception("il ruolo deve essere o 'admin' o 'user'")
            await add_user(
                user_id=values[0],
                mail=values[1],
                name=values[2],
                surname=values[3],
                role=values[4],
            )
            await update.effective_message.reply_text("Utente aggiunto correttamente")
        else:
            raise Exception(
                "Il testo inserito non rispetta il formato corretto: {user_id - mail - name - surname - role}"
            )
    except Exception as e:
        logging.error(
            f"Si è verificato un errore nell'aggiunta di un nuovo utente: {e}",
            exc_info=DEBUG,
        )
        sent_msg = await update.effective_message.reply_text(
            "❌ Si è verificato un errore nell'aggiunta di un nuovo utente"
        )
        await save_user_message(update.effective_user.id, sent_msg.message_id)
        if await user_is_admin(update.effective_user.id):
            sent_msg_error = await update.effective_message.reply_text(str(e))
            await save_user_message(update.effective_user.id, sent_msg_error.message_id)


@authorized
async def new_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Start a new chat session by resetting user memory and messages.
    Args:
        update (Update): The Telegram update object.
        context (ContextTypes.DEFAULT_TYPE): The context of the Telegram bot.
    """
    try:
        user_id = update.effective_user.id

        # Reset memoria AI
        await reset_user_memory(user_id)
        await reset_user_messages(user_id)

        # Messaggio finale (nuova conversazione)
        msg = await update.effective_message.reply_text(
            f"Ciao! Sono {ASSISTANT_NAME}, il tuo assistente personale 🤖\n"
            "Puoi farmi domande liberamente.\n"
            "Per sapere come funziona il bot, come vengono trattati i dati "
            "e alcune avvertenze importanti, usa il comando /info."
        )
        await save_user_message(user_id, msg.message_id)
    except Exception as e:
        logging.error(
            f"Si è verificato un errore nella creazione di una nuova chat: {e}",
            exc_info=DEBUG,
        )
        sent_msg = await update.effective_message.reply_text(
            "❌ Si è verificato un errore nella creazione di una nuova chat"
        )
        await save_user_message(user_id, sent_msg.message_id)
        if await user_is_admin(user_id):
            sent_msg_error = await update.effective_message.reply_text(str(e))
            await save_user_message(user_id, sent_msg_error.message_id)


@authorized
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle user messages, interact with the AI model, and manage chat sessions.
    Args:
        update (Update): The Telegram update object.
        context (ContextTypes.DEFAULT_TYPE): The context of the Telegram bot.
    """
    start_time = time()
    try:
        message = update.effective_message
        user_id = update.effective_user.id
        await save_user_message(user_id, message.message_id)
        all_messages = await get_user_messages(user_id)
        # if the message is an edit, delete all messages after the edited one
        if update.edited_message:
            await delete_user_message(
                user_id,
                message,
                context,
                [mid for mid in all_messages if mid > message.message_id],
            )

        # Send waiting message
        waiting_msg = await message.reply_text("⏳ Sto pensando...")
        await save_user_message(user_id, waiting_msg.message_id)

        content_blocks = await build_content_blocks(message, context)

        # Carica memoria
        messages = await get_user_memory(user_id)
        client = create_agent(
            model,
            tools=load_tools(user_id=user_id),
            system_prompt=SYSTEM_PROMPT,
        )
        max_messages = MAX_MESSAGES
        # Assure that the first message is from human
        if len(messages) > 0:
            while True:
                if messages[-max_messages:][0]["type"] == "human":
                    break
                else:
                    max_messages -= 1
        # Convert messages from dict to LangChain message objects
        messages = messages_from_dict(messages[-max_messages:]) + [HumanMessage(content=content_blocks)]

        # invoke AI
        result = await client.ainvoke({"messages": messages})
        ai_reply = result["messages"][-1].content

        # Save messages to DB
        messages = messages_to_dict(result["messages"])

        await save_user_memory(user_id, messages)

        # Divide message in different parts if too long (telegram has a limit of 4096 characters)
        parts = split_markdown(telegramify_markdown.markdownify(ai_reply))

        # Delete waiting message
        await delete_user_message(
            user_id,
            message,
            context,
            [waiting_msg.message_id],
        )

        # Send message parts
        for part in parts:
            if part.strip() == "":
                continue
            try:
                sent_msg = await message.reply_text(part, parse_mode="MarkdownV2")
            except:
                sent_msg = await message.reply_text(part)
            await save_user_message(user_id, sent_msg.message_id)

        # Check for tool messages with files to send
        for x in messages:
            if x["type"] == "tool":
                try:
                    data_doc: dict = json.loads(x["data"]["content"])
                    key_names = list(data_doc.keys())
                    check_variable = all([name in key_names for name in ["path", "file_name", "creation_time"]])
                    if check_variable:
                        if data_doc["creation_time"] > start_time:
                            with open(data_doc["path"], "rb") as f:
                                await message.reply_document(
                                    document=f,
                                    filename=data_doc["file_name"]
                                )
                            try:
                                os.remove(data_doc["path"])
                            except OSError:
                                pass
                except Exception as e:
                    logging.warning(e, exc_info=True)
                    continue
    except Exception as e:
        logging.error(
            f"Si è verificato un errore durante l'elaborazione del tuo messaggio: {e}",
            exc_info=DEBUG,
        )
        await delete_user_message(
            user_id,
            message,
            context,
            [waiting_msg.message_id],
        )
        sent_msg = await message.reply_text(
            "❌ Si è verificato un errore durante l'elaborazione del tuo messaggio."
        )
        await save_user_message(user_id, sent_msg.message_id)
        if await user_is_admin(user_id):
            sent_msg_error = await message.reply_text(str(e))
            await save_user_message(user_id, sent_msg_error.message_id)


# --- Main ---
def main():
    init_db()
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .connect_timeout(TELEGRAM_CONNECTION_TIMEOUT)
        .pool_timeout(TELEGRAM_POOL_TIMEOUT)
        .read_timeout(TELEGRAM_READ_TIMEOUT)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(CommandHandler("new_chat", new_chat))
    app.add_handler(CommandHandler("new_user", add_user_chat))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, chat))
    logging.info("Bot avviato 🚀")
    app.run_polling(timeout=TELEGRAM_POOL_TIMEOUT)


if __name__ == "__main__":
    main()
