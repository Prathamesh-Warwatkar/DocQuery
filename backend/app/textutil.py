import re

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it
    its me my of on or our shall should so than that the their them then there these they this to us was we
    were what when where which who whom why will with would you your many much any also after before
    between within""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")
_SUFFIXES = ("ations", "ation", "ings", "ing", "edly", "ed", "es", "s", "ly")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def stem(word: str) -> str:
    """Tiny suffix stripper, good enough for term-overlap features."""
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def content_stems(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]
