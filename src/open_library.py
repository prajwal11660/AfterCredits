import requests

EXCLUDED_TERMS = [
    "graphic novel",
    "cocktail",
    "study guide",
    "gradesaver",
    "cliffsnotes",
    "notes on",
    "summary",
    "workbook",
    "teacher",
    "textbook"
]


def get_description(work_key):
    if not work_key:
        return ""

    try:
        url = f"https://openlibrary.org{work_key}.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        description = response.json().get("description", "")

        if isinstance(description, dict):
            description = description.get("value", "")

        return description if isinstance(description, str) else ""

    except requests.RequestException:
        return ""


def search_books(query, limit=20):
    url = "https://openlibrary.org/search.json"

    params = {
        "q": query,
        "limit": limit
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    docs = response.json()["docs"]

    books = []

    for doc in docs:
        title = doc.get("title", "")
        title_lower = title.lower()

        if not title:
            continue

        if any(term in title_lower for term in EXCLUDED_TERMS):
            continue

        authors = doc.get("author_name", [])
        author = authors[0] if authors else "Unknown"

        work_key = doc.get("key", "")
        description = get_description(work_key)

        if not description.strip():
            continue
        books.append({
            "title": title,
            "author": author,
            "description": description
        })

    return books


def get_fiction_candidates(limit=20):
    return search_books("fiction", limit)