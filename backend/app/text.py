"""Tiny, dependency-free text normalisation shared by the offline providers."""

import re

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for
    from further had has have having he her here hers herself him himself his how i if in
    into is it its itself just me more most my myself no nor not now of off on once only or
    other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up
    us very was we were what when where which while who whom why will with would you your
    yours yourself yourselves also may might must shall within without per via
    """.split()  # noqa: SIM905 -- a block of words reads better than a 140-line list
)


def stem(word: str) -> str:
    """Very small suffix stripper (a crude subset of Porter): good enough to match
    "chunks"/"chunk" or "indexing"/"index" without pulling in NLTK."""
    if len(word) <= 3 or word.isdigit():
        return word
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""), ("ed", ""), ("ly", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + repl
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def content_terms(text: str) -> list[str]:
    """Lower-cased, stop-word-free, stemmed tokens in document order."""
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
