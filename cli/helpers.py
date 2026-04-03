import string

STOPWORDS = {"a", "an", "the", "of", "on", "in", "to", "and"}

def normalize(text: str) -> str:
    translator = str.maketrans("","", string.punctuation)
    return text.translate(translator).lower()

def tokenize(text: str) -> list[str]:
    tokens = normalize(text).split()
    return [
        t for t in tokens
        if t and len(t) >= 3 and t not in STOPWORDS
    ]

def is_match(query: str, title: str) -> bool:
    query_tokens = tokenize(query)
    title_tokens = tokenize(title)

    for q in query_tokens:
        for t in title_tokens:
            if len(q) >= 3 and (q in t or t in q):
                return True
    return False
