from langchain.tools import tool
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
from typing import Annotated
from langdetect import detect
import re
import requests
from bs4 import BeautifulSoup
from langchain_tavily import TavilySearch
import os

@tool()
async def current_time() -> str:
    """
    Get the current date and time in the Europe/Rome timezone.
    """
    now = datetime.now(ZoneInfo("Europe/Rome"))
    return json.dumps(
        {"time": now.strftime("%Y-%m-%d %H:%M:%S"), "timezone": "Europe/Rome"}
    )


@tool()
async def add_days(
    date_str: Annotated[str, "Date string in YYYY-MM-DD format"],
    days: Annotated[int, "Number of days to add"],
) -> str:
    """
    Add a number of days to a given date string (YYYY-MM-DD) and return the new date as a string.
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date + timedelta(days=days)
    return new_date.strftime("%Y-%m-%d")


@tool()
async def days_between(
    date_str1: Annotated[str, "First date string in YYYY-MM-DD format"],
    date_str2: Annotated[str, "Second date string in YYYY-MM-DD format"],
) -> int:
    """
    Calculate the number of days between two date strings (YYYY-MM-DD).
    """
    date1 = datetime.strptime(date_str1, "%Y-%m-%d")
    date2 = datetime.strptime(date_str2, "%Y-%m-%d")
    return abs((date2 - date1).days)


@tool()
async def convert_timezone(
    datetime_str: Annotated[str, "Date in ISO 8601 format, e.g. 2025-03-01T15:30:00"],
    from_tz: Annotated[str, "Timezone to convert from, e.g. Europe/Rome"],
    to_tz: Annotated[str, "Timezone to convert to es. America/New_York"],
) -> str:
    """
    Convert a date/time from one timezone to another and return an ISO 8601 string with offset.
    """
    # Se datetime_str è senza timezone, viene interpretato nel fuso from_tz
    dt = datetime.fromisoformat(datetime_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(from_tz))
    else:
        # Se ha già tzinfo, lo si converte prima a from_tz per coerenza
        dt = dt.astimezone(ZoneInfo(from_tz))

    dt_converted = dt.astimezone(ZoneInfo(to_tz))
    return dt_converted.isoformat()


@tool()
async def next_weekday(
    start_date: Annotated[str, "Start date in YYYY-MM-DD format"],
    weekday: Annotated[int, "Day of the week (0=Monday ... 6=Sunday)"],
) -> str:
    """
    Return the next date (EXCLUDING the start date) that falls on the specified weekday.
    """
    date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
    days_ahead = (weekday - date_obj.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    next_date = date_obj + timedelta(days=days_ahead)
    return next_date.isoformat()


@tool()
async def detect_language(
    text: Annotated[str, "Text to detect the language from"],
) -> str:
    """Detect the main language of the text and return the ISO 639-1 code (e.g., 'en', 'it')."""
    return detect(text)


@tool()
async def text_cleaner(
    raw_text: Annotated[str, "Input text to be cleaned"],
    remove_html: Annotated[bool, "If True, remove HTTML tags"] = True,
    normalize_spaces: Annotated[bool, "If True, normalize \\n"] = True,
) -> str:
    """Cleans text from basic HTML and normalizes spaces."""
    text = raw_text

    if remove_html:
        # Rimozione molto semplice dei tag HTML
        text = re.sub(r"<[^>]+>", " ", text)

    if normalize_spaces:
        text = re.sub(r"\s+", " ", text).strip()

    return text


@tool()
async def fetch_url(
    url: Annotated[str, "URL of page to fetch"],
    timeout_sec: Annotated[int, "Timeout in seconds"] = 10,
) -> dict:
    """
    Download the content of a web page and return title, status, and main text (paragraphs).
    In case of HTTP error, returns a dictionary with the 'error' key.
    """
    try:
        resp = requests.get(url, timeout=timeout_sec)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {
            "url": url,
            "error": str(e),
        }

    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else None
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    main_text = "\n\n".join(p for p in paragraphs if p)

    return {
        "url": url,
        "status_code": resp.status_code,
        "title": title,
        "text": main_text or resp.text,
    }


@tool()
async def get_website_metadata(
    url: Annotated[str, "URL of website to extract metadata from"],
    timeout_sec: Annotated[int, "Timeout for HTTP request"] = 10,
) -> dict:
    """
    Extract basic metadata (title, description, keywords, favicon) from a web page.
    """
    try:
        resp = requests.get(url, timeout=timeout_sec)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {
            "url": url,
            "error": str(e),
        }

    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else None
    description_tag = soup.find("meta", attrs={"name": "description"})
    keywords_tag = soup.find("meta", attrs={"name": "keywords"})
    icon_tag = soup.find("link", rel=lambda v: v and "icon" in v.lower())

    return {
        "url": url,
        "title": title,
        "meta_description": (
            description_tag.get("content", "").strip()
            if description_tag and description_tag.get("content")
            else None
        ),
        "meta_keywords": (
            keywords_tag.get("content", "").strip()
            if keywords_tag and keywords_tag.get("content")
            else None
        ),
        "favicon": icon_tag.get("href") if icon_tag and icon_tag.get("href") else None,
    }


TOOLS = [
    current_time,
    add_days,
    days_between,
    convert_timezone,
    next_weekday,
    detect_language,
    text_cleaner,
    fetch_url,
    get_website_metadata,
]


def load_tools():
    """
    Load tools.
    """
    tools = [
        current_time,
        add_days,
        days_between,
        convert_timezone,
        next_weekday,
        detect_language,
        text_cleaner,
        fetch_url,
        get_website_metadata,
    ]
    if "TAVILY_API_KEY" in os.environ:
        tavily_search = TavilySearch(max_results=5, topic="general")
        tools.append(tavily_search)
    return tools
