import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import requests

from lexical_similarity import (
    lexical_similarities,
    title_similarity,
    token_coverage,
)


logger = logging.getLogger(__name__)
OPEN_LIBRARY_ROOT = "https://openlibrary.org"
REQUEST_TIMEOUT_SECONDS = 5
REQUEST_HEADERS = {
    "User-Agent": "AfterCredits/1.0 (book recommendation app)",
}
MAX_DISCOVERY_TOPICS = 4
MAX_DESCRIPTION_LOOKUPS = 24

EXCLUDED_TERMS = (
    "graphic novel",
    "cocktail",
    "study guide",
    "gradesaver",
    "cliffsnotes",
    "notes on",
    "summary",
    "workbook",
    "teacher",
    "textbook",
)

_TOPIC_HINTS = (
    (
        re.compile(r"\b(prison|jail|inmate|prisoner|convict|incarcerat)\w*\b"),
        ("prisoners", "prisons"),
        "prison jail inmate prisoner convict incarceration captivity",
    ),
    (
        re.compile(r"\b(escap|fugitiv|break out)\w*\b"),
        ("escapes", "prison_escapes"),
        "escape prison break fugitive freedom breakout",
    ),
    (
        re.compile(r"\b(redemption|forgiv|second chance)\w*\b"),
        ("redemption", "forgiveness"),
        "redemption forgiveness second chance rehabilitation",
    ),
    (
        re.compile(r"\b(war|battle|clan|kingdom|throne|dynast)\w*\b"),
        ("war", "kings_and_rulers"),
        "war battle clan kingdom throne dynasty political intrigue",
    ),
    (
        re.compile(r"\b(crime|criminal|detective|murder)\w*\b"),
        ("crime", "murder", "criminal_justice"),
        "crime murder criminal investigation justice wrongful conviction",
    ),
    (
        re.compile(r"\b(friend|loyal|companions|connection)\w*\b"),
        ("friendship", "loyalty"),
        "friendship loyalty companionship connection",
    ),
    (
        re.compile(r"\b(surviv|wilderness|apocalypse|disaster)\w*\b"),
        ("survival", "survival_fiction"),
        "survival endurance hardship resilience",
    ),
    (
        re.compile(r"\b(magic|wizard|dragon|fantasy|myth)\w*\b"),
        ("fantasy", "fantasy_fiction"),
        "fantasy magic mythology dragons adventure",
    ),
    (
        re.compile(r"\b(love|romance)\b|\b(relationship|heartbreak)\w*\b"),
        ("love", "romance"),
        "love romance relationships heartbreak",
    ),
    (
        re.compile(r"\b(meaning|purpose|hope|rediscovers|finds meaning)\w*\b"),
        ("hope", "purpose"),
        "hope purpose meaning of life second chance",
    ),
    (
        re.compile(r"\b(grief|loss|mourning|bereav)\w*\b"),
        ("grief", "loss"),
        "grief loss mourning bereavement",
    ),
)
_NONFICTION_HINTS = (
    "autobiography",
    "biography",
    "memoir",
    "nonfiction",
    "non-fiction",
    "annual report",
    "social science",
    "political science",
    "criminal justice",
    "political aspects",
    "prisons, administration",
)


class OpenLibraryError(RuntimeError):
    """Raised when Open Library returns invalid catalog data."""


def _get_json(url, params=None):
    response = requests.get(
        url,
        params=params,
        headers=REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise OpenLibraryError("Open Library returned an invalid response.")
    return payload


def _description_text(value):
    if isinstance(value, dict):
        value = value.get("value", "")
    if isinstance(value, list):
        value = " ".join(item for item in value if isinstance(item, str))
    return value.strip()[:2500] if isinstance(value, str) else ""


def get_description(work_key):
    if not isinstance(work_key, str) or not work_key.startswith("/works/"):
        return ""

    try:
        payload = _get_json(f"{OPEN_LIBRARY_ROOT}{work_key}.json")
    except requests.RequestException:
        logger.info("Could not retrieve Open Library work %s.", work_key)
        return ""
    except OpenLibraryError:
        logger.warning("Open Library returned invalid work data for %s.", work_key)
        return ""
    return _description_text(payload.get("description"))


def _first_sentence(doc):
    return _description_text(doc.get("first_sentence"))


def _author(doc):
    authors = doc.get("authors", [])
    if isinstance(authors, list) and authors:
        first_author = authors[0]
        if isinstance(first_author, dict):
            name = first_author.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

    authors = doc.get("author_name", [])
    if isinstance(authors, list) and authors:
        first_author = authors[0]
        if isinstance(first_author, str) and first_author.strip():
            return first_author.strip()
    return "Unknown"


def _subjects(doc):
    subjects = doc.get("subject", [])
    if not isinstance(subjects, list):
        return []
    return [
        subject.strip()
        for subject in subjects
        if isinstance(subject, str) and subject.strip()
        and len(subject) <= 120
    ][:60]


def _valid_candidate(doc):
    if not isinstance(doc, dict):
        return False
    title = doc.get("title")
    work_key = doc.get("key")
    if (
        not isinstance(title, str)
        or not title.strip()
        or not isinstance(work_key, str)
        or not work_key.startswith("/works/")
    ):
        return False
    title = title.casefold()
    return not any(term in title for term in EXCLUDED_TERMS)


def _query_topics(movie_title, movie_description):
    normalized = f"{movie_title} {movie_description}".casefold()
    subjects = []
    expansions = []
    for pattern, subject_names, expansion in _TOPIC_HINTS:
        if pattern.search(normalized):
            for subject in subject_names:
                if subject not in subjects:
                    subjects.append(subject)
            expansions.append(expansion)

    return subjects[:MAX_DISCOVERY_TOPICS], " ".join(expansions)


def _is_likely_nonfiction(doc):
    subjects = " ".join(_subjects(doc)).casefold()
    has_fiction_marker = bool(
        re.search(r"\bfiction\b|\bnovel\b|\bnovella\b", subjects)
    )
    return (
        any(hint in subjects for hint in _NONFICTION_HINTS)
        and not has_fiction_marker
    )


def _search_title(query, limit):
    payload = _get_json(
        f"{OPEN_LIBRARY_ROOT}/search.json",
        params={
            "q": query,
            "limit": limit,
            "fields": "key,title,author_name,first_sentence,subject",
        },
    )
    docs = payload.get("docs", [])
    if not isinstance(docs, list):
        raise OpenLibraryError("Open Library title search returned invalid data.")
    return [doc for doc in docs if _valid_candidate(doc)]


def _search_subject(subject, limit):
    payload = _get_json(
        f"{OPEN_LIBRARY_ROOT}/subjects/{quote(subject, safe='')}.json",
        params={"limit": limit},
    )
    works = payload.get("works", [])
    if not isinstance(works, list):
        raise OpenLibraryError(
            "Open Library subject search returned invalid data."
        )
    return [work for work in works if _valid_candidate(work)]


def _title_key(title):
    return re.sub(r"[^\w]", "", title.casefold())


def discover_books(movie_title, movie_description, limit=24):
    """Retrieve topical works from Open Library for recommendation ranking."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    subjects, query_expansion = _query_topics(
        movie_title,
        movie_description,
    )
    if not subjects:
        subjects = ["fiction"]

    search_tasks = (
        [("title", movie_title.strip()[:200])]
        if movie_title.strip()
        else []
    )
    keyword_query = " ".join(movie_description.split())[:300]
    if keyword_query:
        search_tasks.append(("keywords", keyword_query))
    search_tasks.extend(("subject", subject) for subject in subjects)
    errors = []
    documents = []

    with ThreadPoolExecutor(max_workers=min(5, len(search_tasks))) as executor:
        futures = {
            executor.submit(
                _search_subject if kind == "subject" else _search_title,
                query,
                min(limit * 4, 100) if kind == "subject" else min(limit, 40),
            ): (kind, query)
            for kind, query in search_tasks
        }
        for future in as_completed(futures):
            kind, query = futures[future]
            try:
                documents.extend(future.result())
            except requests.RequestException as error:
                logger.warning(
                    "Open Library %s search failed for %r: %s",
                    kind,
                    query,
                    error,
                )
                errors.append(f"{kind} search unavailable")
            except OpenLibraryError as error:
                logger.warning(
                    "Open Library %s search returned invalid data: %s",
                    kind,
                    error,
                )
                errors.append(f"{kind} search returned invalid data")

    candidates_by_key = {}
    for doc in documents:
        if _is_likely_nonfiction(doc):
            continue
        work_key = doc["key"]
        candidate = candidates_by_key.setdefault(
            work_key,
            {
                "title": doc["title"].strip(),
                "author": _author(doc),
                "description": _first_sentence(doc),
                "search_terms": _subjects(doc),
                "source": "Open Library",
                "source_url": f"{OPEN_LIBRARY_ROOT}{work_key}",
                "_work_key": work_key,
            },
        )
        for subject in _subjects(doc):
            if subject not in candidate["search_terms"]:
                candidate["search_terms"].append(subject)
        if candidate["author"] == "Unknown":
            candidate["author"] = _author(doc)
        if not candidate["description"]:
            candidate["description"] = _first_sentence(doc)

    candidates = list(candidates_by_key.values())
    if not candidates:
        return [], query_expansion, errors

    preliminary_documents = [
        f"{book['title']} {book['description']} "
        f"{' '.join(book['search_terms'])}"
        for book in candidates
    ]
    preliminary_plot_scores = lexical_similarities(
        movie_description,
        [
            f"{book['title']} {book['description']}"
            for book in candidates
        ],
    )
    candidates = sorted(
        zip(preliminary_plot_scores, preliminary_documents, candidates),
        key=lambda item: (
            0.55 * item[0]
            + 0.30 * token_coverage(query_expansion, item[1])
            + 0.15 * title_similarity(movie_title, item[2]["title"])
        ),
        reverse=True,
    )[:MAX_DESCRIPTION_LOOKUPS]
    candidates = [candidate for _, _, candidate in candidates]

    with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as executor:
        futures = {
            executor.submit(get_description, book["_work_key"]): book
            for book in candidates
        }
        for future in as_completed(futures):
            book = futures[future]
            description = future.result()
            if description:
                book["description"] = description
            elif not book["description"]:
                book["description"] = (
                    "Catalog subjects: " + ", ".join(book["search_terms"][:12])
                )

    for book in candidates:
        book.pop("_work_key", None)

    return candidates[:limit], query_expansion, errors


def merge_book_candidates(local_books, discovered_books):
    """Merge local and discovered titles while preserving useful metadata."""
    merged = {}
    for book in local_books:
        candidate = dict(book)
        candidate.setdefault("source", "Local catalog")
        candidate.setdefault("source_url", None)
        candidate.setdefault("search_terms", [])
        merged[_title_key(candidate["title"])] = candidate

    for book in discovered_books:
        key = _title_key(book["title"])
        if key not in merged:
            merged[key] = dict(book)
            continue

        existing = merged[key]
        existing["source"] = "Local catalog and Open Library"
        if not existing.get("source_url"):
            existing["source_url"] = book.get("source_url")
        terms = existing.setdefault("search_terms", [])
        for term in book.get("search_terms", []):
            if term not in terms:
                terms.append(term)
        if existing.get("author") == "Unknown":
            existing["author"] = book.get("author", "Unknown")

    return list(merged.values())


def search_books(query, limit=20):
    """Search Open Library by title or keywords."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer.")
    documents = _search_title(query, min(limit, 100))
    books = []
    for doc in documents:
        work_key = doc["key"]
        description = _first_sentence(doc)
        if not description:
            description = get_description(work_key)
        if not description:
            description = "Catalog subjects: " + ", ".join(_subjects(doc)[:12])
        books.append({
            "title": doc["title"].strip(),
            "author": _author(doc),
            "description": description,
            "search_terms": _subjects(doc),
            "source": "Open Library",
            "source_url": f"{OPEN_LIBRARY_ROOT}{work_key}",
        })
    return books


def get_fiction_candidates(limit=20):
    return search_books("fiction", limit)
