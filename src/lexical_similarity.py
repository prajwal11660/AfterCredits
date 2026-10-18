import math
import re
from collections import Counter
from difflib import SequenceMatcher


_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at",     "be", "been", "between", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i",
    "in", "is", "it", "its", "of", "on", "or", "our", "she", "that",
    "the", "their", "them", "there", "they", "this", "through", "to",
    "was", "we", "were", "who", "went", "with", "you", "your",
})


def _normalize_token(token):
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        return token[:-1]
    return token


def _tokens(text):
    return [
        _normalize_token(token)
        for token in _TOKEN_PATTERN.findall(text.casefold())
        if token not in _STOP_WORDS
    ]


def _cosine(vector_a, vector_b):
    dot_product = sum(
        value * vector_b.get(term, 0.0)
        for term, value in vector_a.items()
    )
    magnitude_a = math.sqrt(sum(value * value for value in vector_a.values()))
    magnitude_b = math.sqrt(sum(value * value for value in vector_b.values()))
    if not magnitude_a or not magnitude_b:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def lexical_similarities(query, documents):
    """Return deterministic TF-IDF cosine scores against all documents."""
    tokenized = [_tokens(query)] + [_tokens(document) for document in documents]
    document_frequency = Counter(
        token
        for tokens in tokenized
        for token in set(tokens)
    )
    document_count = len(tokenized)
    idf = {
        token: math.log(
            (document_count + 1) / (frequency + 1)
        ) + 1.0
        for token, frequency in document_frequency.items()
    }

    vectors = []
    for tokens in tokenized:
        counts = Counter(tokens)
        vector = {
            token: (1.0 + math.log(count)) * idf[token]
            for token, count in counts.items()
        }
        vectors.append(vector)

    return [
        _cosine(vectors[0], document_vector)
        for document_vector in vectors[1:]
    ]


def lexical_match_terms(query, document):
    """Return query terms that also occur in a candidate's searchable text."""
    query_terms = set(_tokens(query))
    document_terms = set(_tokens(document))
    return sorted(query_terms & document_terms)


def title_similarity(query, candidate_title):
    """Compare title tokens while tolerating small spelling errors."""
    query_tokens = _tokens(query)
    title_tokens = _tokens(candidate_title)
    if not query_tokens or not title_tokens:
        return 0.0

    matched = 0.0
    for query_token in query_tokens:
        best = 0.0
        for title_token in title_tokens:
            if query_token == title_token:
                best = 1.0
            elif min(len(query_token), len(title_token)) >= 6:
                best = max(
                    best,
                    SequenceMatcher(
                        None,
                        query_token,
                        title_token,
                    ).ratio(),
                )
        matched += best
    return matched / len(query_tokens)


def token_coverage(query, document, max_query_terms=5):
    """Measure how many distinct query concepts appear in candidate metadata."""
    query_tokens = set(_tokens(query))
    document_tokens = set(_tokens(document))
    if not query_tokens:
        return 0.0
    denominator = min(max_query_terms, len(query_tokens))
    return min(1.0, len(query_tokens & document_tokens) / denominator)
