# Telegram AI Copilot with Extensible Tools

This project is a private-but-extensible Telegram assistant powered by OpenAI GPT-5.1 models, LangChain agents, and a toolbox of domain-specific actions. It handles text, media, office documents, memory, shopping lists, and dynamic file generation while enforcing an allow-list of Telegram users.

## Table of Contents
- [Key Features](#key-features)
- [High-Level Architecture](#high-level-architecture)
- [Supported Attachments & Tools](#supported-attachments--tools)
- [Requirements](#requirements)
- [Environment Variables](#environment-variables)
- [Setup & Local Run](#setup--local-run)
- [Telegram Commands](#telegram-commands)
- [Extending the Tooling Layer](#extending-the-tooling-layer)
- [Customizing System Prompts](#customizing-system-prompts)
- [Data Handling & Persistence](#data-handling--persistence)
- [Troubleshooting Tips](#troubleshooting-tips)
- [Changelog & License](#changelog--license)

## Key Features
- **AI-first chat** powered by `gpt-5.1` (general agent) and a separate high-reasoning math agent.
- **Strict access control**: only Telegram IDs stored in the SQLite database can interact with the bot; admins can onboard new users live.
- **Memory & shopping lists**: dedicated tools persist user preferences and grocery items per chat.
- **Rich attachment support**: text, photos, images, audio/voice notes (auto-transcribed), PDF, Word, Excel, and CSV documents are normalized before reaching the agent.
- **File generation workflow**: the agent can create PDF, DOCX, and XLSX files on demand and send them back to the chat.
- **Modular tool loader**: every module ending with `_tool.py` inside `tool/` is auto-discovered, so new capabilities ship without touching the agent code.

## High-Level Architecture

| Layer | Entry Points | Responsibilities |
| --- | --- | --- |
| Telegram Bot | `main.py` | Loads env vars, configures `python-telegram-bot`, guards commands with `authorized` / `authorized_admin`, orchestrates conversations, chunked replies, and cleanup of temporary files. |
| AI Provider | `chat_utils/openai_provider.py` | Exposes `model` (general assistant) and `math_model` (reasoning-intensive math agent) plus Whisper-based `transcribe_audio_bytes` for audio attachments. |
| Persistence | `chat_utils/db.py` | Async SQLite layer storing users, Telegram message IDs, chat memory, permanent memories, and shopping lists. |
| Content Normalization | `chat_utils/utils.py` | `build_content_blocks` extracts text from media/documents (PDF, DOCX, XLSX, CSV), converts audio into text, and seeds the agent with structured content blocks. Also seeds the DB with the initial admin via `init_db()`. |
| Prompts | `chat_utils/prompt.py` | Houses `SYSTEM_PROMPT` for the main agent and `SYSTEM_PROMPT_MATH` for the math specialist, including tool policies and formatting rules. |
| Tools | `tool/` | Each `_tool.py` module exposes a `load_tools` hook. The dynamic loader (`tool/__init__.py`) inspects signatures to pass `user_id` or other context automatically. |

## Supported Attachments & Tools

**Attachments**
- Text or captions (plain Markdown input)
- Photos / images (Telegram photos or image documents)
- Audio / voice / video notes (transcribed to text via Whisper)
- Documents: PDF (first N pages), Word (.docx/.doc), Excel (.xlsx/.xls), CSV
- Unsupported formats return a short “file received” notice so users know the bot saw the upload

**Built-in tools**
- `tool/utils_tool.py`: time & timezone helpers, date math, language detection, text cleaning, web scraping/metadata, optional Tavily search when `TAVILY_API_KEY` is set.
- `tool/math_tool.py`: a dedicated `math_expert_agent` that switches to `math_model` and can call dozens of algebra/geometry/statistics helpers.
- `tool/memories_tool.py`: persist, list, or reset user memories for long-lived personalization.
- `tool/shopping_tool.py`: manage per-user shopping lists through CRUD operations backed by SQLite.
- `tool/generate_file_tool.py`: structured PDF/DOCX/XLSX generation (reports, templates, tables) with automatic cleanup once delivered.

## Requirements
- Python 3.11+
- A Telegram bot token (`@BotFather`)
- OpenAI API key with access to GPT-5.1 and Whisper
- (Optional) Tavily API key for web search augmentation
- Dependencies listed in `requirements.txt` (LangChain, python-telegram-bot, ReportLab, python-docx, openpyxl, etc.)

## Environment Variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `TELEGRAM_TOKEN` | ✅ | Token of the Telegram bot. |
| `OPENAI_API_KEY` | ✅ | OpenAI credential for chat completions and Whisper transcription. |
| `CHAT_MEMORY_DB` | ✅ | Absolute or relative path to the SQLite file (e.g., `chat_memory.db`). |
| `ADMIN_USER_ID` | ✅ | Telegram ID promoted to admin at boot; seeded via `init_db()`. |
| `ADMIN_MAIL`, `ADMIN_NAME`, `ADMIN_SURNAME` | ➖ | Optional metadata stored in the users table for the admin. |
| `ASSISTANT_NAME` | ➖ | Alias injected into the system prompt (defaults to `AiChat`). |
| `MAX_MESSAGES` | ➖ | Sliding window of messages kept in context (default `30`). |
| `DEBUG_TELEGRAM` | ➖ | `1` to include stack traces in logs and admin replies. |
| `TELEGRAM_CONNECTION_TIMEOUT`, `TELEGRAM_POOL_TIMEOUT`, `TELEGRAM_READ_TIMEOUT` | ➖ | Fine-tune bot network timeouts (seconds). |
| `TAVILY_API_KEY` | ➖ | Enables the Tavily search tool. |
| `LANGUAGE` | ➖ | The languages allowed are `it`(italian) and `eng` (english). Default is `eng`|

> Tip: store secrets in a `.env` file and let `python-dotenv` (already imported in `main.py`) load them automatically.

## Setup & Local Run
1. **Clone & install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Configure environment** using `.env` or your favorite secret manager.
3. **Seed the admin user**: on first launch the bot invokes `init_db()` which reads `ADMIN_*` env vars and ensures the admin row exists.
4. **Run the bot**
   ```bash
   python main.py
   ```
5. **Onboard additional users** directly from Telegram via `/new_user {id - mail - name - surname - role}` (role is `admin` or `user`).

## Telegram Commands

| Command | Audience | Effect |
| --- | --- | --- |
| `/start` | Any authorized user | Clears chat memory/messages and sends the welcome text. |
| `/info` | Any authorized user | Displays privacy & usage disclaimers. |
| `/new_chat` | Any authorized user | Resets the AI memory window for a “fresh” conversation. |
| `/new_user {id - mail - name - surname - role}` | Admins only | Adds or updates a row in the users table so new Telegram IDs can chat. |

## Extending the Tooling Layer
1. **Create a module** named `tool/<something>_tool.py`.
2. **Expose a `load_tools` function** returning either a list of LangChain tools or a single tool. Parameters like `user_id` are injected automatically when present in the signature thanks to `tool/__init__.py`.
3. **Decorate each callable** with `@tool()` (LangChain) and type-annotate the arguments to improve agent planning.
4. **Return structured payloads** (dicts/JSON-serializable) if the tool must send files back to the user. Include `path`, `file_name`, and `creation_time` so `main.py` knows whether to forward or delete.
5. **Restart the bot**—new modules are auto-discovered at startup; no extra wiring needed.

_Minimal example_
```python
from langchain.tools import tool

def load_tools():
	@tool()
	async def hello(name: str) -> str:
		return f"Ciao {name}!"

	return [hello]
```

## Customizing System Prompts
- Modify `chat_utils/prompt.py` to tweak tone, allowed tools, Markdown policies, or math instructions.
- `SYSTEM_PROMPT` governs the general assistant; `SYSTEM_PROMPT_MATH` affects the math expert agent invoked by `tool/math_tool.py`.
- After editing prompts, simply restart `main.py`; no additional build steps are required.

## Data Handling & Persistence
- **Database**: SQLite file defined by `CHAT_MEMORY_DB` stores users, Telegram message IDs (for cleanup), serialized LangChain memories, permanent memories, and shopping lists.
- **File lifecycle**: generated documents live in temporary directories and are deleted right after the bot delivers them to Telegram.
- **Attachment truncation**: PDF and Excel extractors limit pages/rows (`max_pages`, `max_rows`) to stay within token and Telegram size limits.
- **Privacy**: `/info` reminds users not to send sensitive data; the repository ships as a private assistant by default (no broad access).

## Troubleshooting Tips
- **Auth issues**: if users see an access denied message, ensure their Telegram ID exists in the `users` table (use `/new_user`).
- **Stuck conversations**: `/new_chat` plus deleting the corresponding row in `messages` (or the entire DB file) resets the AI state.
- **Attachment parsing errors**: verify the runtime has `pypdf`, `python-docx`, `openpyxl`, and `ffmpeg` (for Telegram voice conversions handled by Telegram servers).
- **Tool discovery**: confirm your custom module name ends with `_tool.py` and exports `load_tools`.
- **OpenAI errors**: set `DEBUG_TELEGRAM=1` to receive stack traces as an admin when the model call fails.

## Changelog & License
- See [`CHANGELOG.md`](CHANGELOG.md) for release notes.
- Licensed under the [MIT License](LICENSE). Feel free to fork, extend, and deploy your own instance while keeping credentials safe.

Made with ❤️ for private, high-trust Telegram copilots.
