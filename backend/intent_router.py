import re

TEMPORAL_MARKERS = [
    "today", "now", "current", "currently", "latest", "recent", "live",
    "upcoming", "present", "this week", "this month", "real time", "real-time"
]

WEATHER_TRIGGERS = [
    "weather", "temperature", "forecast", "rain", "aqi", "air quality", "humidity"
]

FINANCIAL_TRIGGERS = [
    "stock price", "share price", "market cap", "nasdaq", "sensex", "nifty",
    "crypto", "bitcoin", "ethereum", "exchange rate", "dollar rate", "inr rate"
]

POLITICAL_LEADERSHIP_TRIGGERS = [
    "chief minister", "prime minister", "president", "governor", "mayor",
    "ceo", "cabinet minister", "who is the current"
]

SPORTS_MEDIA_TRIGGERS = [
    "score", "match", "game result", "points table", "standings",
    "release date", "schedule"
]

ALL_VOLATILE_TRIGGERS = (
    WEATHER_TRIGGERS +
    FINANCIAL_TRIGGERS +
    POLITICAL_LEADERSHIP_TRIGGERS +
    SPORTS_MEDIA_TRIGGERS
)

RECENCY_PATTERNS = [
    re.compile(r"\b(202[4-9]|203[0-9])\b", re.IGNORECASE),
    re.compile(r"\bwho is the\b", re.IGNORECASE),
    re.compile(r"\bwhat happened to\b", re.IGNORECASE),
    re.compile(r"\bbreaking news\b", re.IGNORECASE),
]

DOC_OVERRIDE_PATTERN = re.compile(
    r"\b(in|from|according to|based on) (?:the|this) (?:document|pdf|file|paper|uploaded document|context)\b",
    re.IGNORECASE
)


def should_web_search_directly(query: str) -> bool:
    if not query:
        return False

    q = query.lower().strip()

    if DOC_OVERRIDE_PATTERN.search(q):
        return False

    for trigger in ALL_VOLATILE_TRIGGERS:
        if trigger in q:
            return True

    for marker in TEMPORAL_MARKERS:
        if re.search(r"\b" + re.escape(marker) + r"\b", q):
            return True

    for pattern in RECENCY_PATTERNS:
        if pattern.search(q):
            return True

    return False
