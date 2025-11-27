import os

LANGUAGE = os.getenv("LANGUAGE", "en")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "AiChat")

if LANGUAGE == "it":
    NOT_AUTHENTICATED_MESSAGE = (
        "❌ Questo bot è ad accesso limitato e non sei autorizzato a utilizzarlo.\n"
        "I messaggi che invii non vengono inoltrati all'intelligenza artificiale "
        "e sono usati solo per verificare l'accesso.\n"
        "Ti chiediamo di non usare questo bot."
    )
    NOT_AUTHENTICATED_MESSAGE_ADMIN = "❌ Non sei autorizzato ad usare questo comando.\nSolo gli utenti amministratori possono aggiungere nuovi utenti."
    INFO_MESSAGE = (
        "Questo bot utilizza un servizio di intelligenza artificiale per generare le risposte.\n"
        "Le risposte possono contenere errori e non sostituiscono il parere di un professionista.\n"
        "Non inviare dati sensibili (salute, password, dati bancari, documenti, ecc.).\n"
        "Il bot è ad uso privato/sperimentale e l'accesso è limitato a utenti autorizzati."
    )
    WELCOME_MESSAGE = (
        f"Ciao! Sono {ASSISTANT_NAME}, il tuo assistente personale 🤖\n"
        "Puoi farmi domande liberamente.\n"
        "Per sapere come funziona il bot, come vengono trattati i dati "
        "e alcune avvertenze importanti, usa il comando /info."
    )
    WAITING_MESSAGE = "⏳ Sto pensando..."
    ERROR_AUTH_MESSAGE = "❌ Si è verificato un errore nella fase di autenticazione dell'utente."
    ERROR_INFO_MESSAGE = "❌ Si è verificato un errore nel fornire informazioni sul bot all'utente."
    ERROR_START_MESSAGE = "❌ Si è verificato un errore nella fase di inizializzazione del bot."
    ERROR_CHAT_MESSAGE = "❌ Si è verificato un errore durante l'elaborazione del tuo messaggio."
else:
    NOT_AUTHENTICATED_MESSAGE = (
        "❌ This bot is restricted access and you are not authorized to use it.\n"
        "The messages you send are not forwarded to the AI "
        "and are only used to verify access.\n"
        "We kindly ask you not to use this bot."
    )
    NOT_AUTHENTICATED_MESSAGE_ADMIN = "❌ You are not authorized to use this command.\nOnly admin users can add new users."
    INFO_MESSAGE = (
        "This bot uses an AI service to generate responses.\n"
        "Responses may contain errors and do not replace professional advice.\n"
        "Do not send sensitive data (health, passwords, banking data, documents, etc.).\n"
        "The bot is for private/experimental use and access is limited to authorized users."
    )
    WELCOME_MESSAGE = (
        f"Hello! I'm {ASSISTANT_NAME}, your personal assistant 🤖\n"
        "Feel free to ask me questions.\n"
        "To learn how the bot works, how data is handled, "
        "and some important warnings, use the /info command."
    )
    WAITING_MESSAGE = "⏳ I'm thinking..."
    ERROR_AUTH_MESSAGE = "❌ An error occurred during user authentication."
    ERROR_INFO_MESSAGE = "❌ An error occurred while providing bot information to the user."
    ERROR_START_MESSAGE = "❌ An error occurred during bot initialization."
    ERROR_CHAT_MESSAGE = "❌ An error occurred while processing your message."
