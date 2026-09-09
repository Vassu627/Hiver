from __future__ import annotations

import hashlib
import re
import unicodedata

MENTION = re.compile(r"@\w+")
TCO = re.compile(r"https://t\.co/\S+")
HTML_AMP = re.compile(r"&amp;")
WS = re.compile(r"\s+")
HASHTAG = re.compile(r"#\w+")

# I + variation selector / replacement-character keyboard bug (iOS 11 era)
KEYBOARD_BUG = re.compile(
    r"(I️|I\uFE0F|\uFFFD|question mark|weird char|boxed \?|letter [\"']i[\"']|"
    r"autocorrect|auto correct|Illumin)",
    re.I,
)

ANGER = re.compile(
    r"\b(fuck|shit|sucks|hate|worst|garbage|trash|useless|horrible|awful|"
    r"pissed|ridiculous|wtf|stupid)\b",
    re.I,
)

REPEAT_HELP = re.compile(
    r"(tried .{0,50}(everything|all( these| those)?|those steps|uninstall|reinstall|"
    r"reboot|restart|restore)|still (happening|not working|have no)|"
    r"contacted .{0,20}(times|already)|hours on the phone|dozens of calls|"
    r"5 times|30 times)",
    re.I,
)

LEGAL = re.compile(r"\b(lawyer|lawsuit|sue|legal action|attorney)\b", re.I)

PASSWORD_FISH = re.compile(
    r"\b(password|ssn|social security|credit card number|cvv)\b", re.I
)

NON_LATIN = re.compile(
    r"[\u0400-\u04FF\u0600-\u06FF\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"
)

ES = re.compile(
    r"\b(qué|cuando|cuándo|estoy|tengo|gracias|actualizaci[oó]n|bater[ií]a|"
    r"problema|teclado|ayuda|por favor|celular|descontenta|soluci[oó]n|"
    r"lentis[ií]mo|agot|vais a|para cuando)\b",
    re.I,
)
PT = re.compile(
    r"\b(n[aã]o|meu|minha|celular|atualiza|bateria|obrigad|estou|voc[eê]|"
    r"rolando|impedido|estragar)\b",
    re.I,
)
IT = re.compile(
    r"\b(surriscalda|batteria durata|meno reattivo|andava meglio)\b", re.I
)
FR = re.compile(r"\b(bonjour|merci|s'il vous pla[iî]t|j'ai|probl[eè]me)\b", re.I)
DE = re.compile(r"\b(nicht|danke|bitte| entschuldigung|aktualisierung)\b", re.I)


def uid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def strip_handles(text: str) -> str:
    text = MENTION.sub(" ", text)
    text = HTML_AMP.sub("&", text)
    return WS.sub(" ", text).strip()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = HTML_AMP.sub("&", text)
    text = text.replace("&gt;", ">").replace("&lt;", "<")
    text = MENTION.sub(" ", text)
    text = HASHTAG.sub(" ", text)
    text = TCO.sub(" ", text)
    return WS.sub(" ", text).strip()


def looks_non_english(text: str) -> bool:
    if NON_LATIN.search(text):
        return True
    # macOS "El Capitan" is English
    lowered = text.lower()
    if "el cap" in lowered:
        return False
    hits = sum(bool(p.search(text)) for p in (ES, PT, IT, FR, DE))
    return hits >= 1 and len(normalize(text).split()) <= 40


def is_keyboard_bug(text: str) -> bool:
    if "I️" in text or "\uFE0F" in text and re.search(r"I\uFE0F", text):
        return True
    if KEYBOARD_BUG.search(text):
        return True
    # classic "I" → boxed question mark complaints
    if re.search(r"\bI\s*[?\uFFFD]", text) or "I.T" in text or "I️" in text:
        return True
    return bool(re.search(r"(question marks?|weird charcter|boxed \?|A boxed)", text, re.I))


def is_screenshot_only(text: str) -> bool:
    cleaned = strip_handles(text)
    cleaned = TCO.sub("", cleaned).strip()
    return len(cleaned) < 8


def is_too_short(text: str) -> bool:
    return len(normalize(text)) < 28 or len(normalize(text).split()) < 4


def anger_score(text: str) -> int:
    return len(ANGER.findall(text))


def extract_device(text: str) -> str | None:
    m = re.search(
        r"\b(iphone\s*(?:se|x|xr|xs|\d\s*(?:plus|s|s plus)?)?|ipad(?:\s*pro)?|"
        r"macbook(?:\s*pro)?|imac|apple watch|airpods|iphone)\b",
        text,
        re.I,
    )
    return m.group(0) if m else None


def extract_ios(text: str) -> str | None:
    m = re.search(r"\bios\s*(\d+(?:\.\d+){0,2})\b", text, re.I)
    return m.group(1) if m else None


def extract_urls(text: str) -> list[str]:
    return TCO.findall(text or "")
