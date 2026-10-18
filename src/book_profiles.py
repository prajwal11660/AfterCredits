import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

from book_data import load_books
from book_validation import log_book_issues, validate_books
from story_profile import (
    PROFILE_VERSION,
    StoryProfile,
    create_story_profile,
    is_valid_story_profile,
)


PROFILE_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "book_profiles.json"
)

API_PROFILE_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "api_book_profiles.json"
)

PROFILE_FORMAT_VERSION = 1
logger = logging.getLogger(__name__)


class ProfileCacheError(ValueError):
    """Raised when a cached profile artifact is missing or incompatible."""


def _source_digest(books):
    source = [
        {
            "title": book["title"],
            "author": book["author"],
            "description": book["description"],
        }
        for book in books
    ]
    serialized = json.dumps(
        source,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_prepared_books(books):
    if not isinstance(books, list):
        raise ProfileCacheError("Profile artifact books must be a list.")

    valid_books, issues = validate_books(books)
    if issues or len(valid_books) != len(books):
        raise ProfileCacheError(
            "Profile artifact contains invalid book records: "
            + "; ".join(issues)
        )

    for book in valid_books:
        story_profile = book.get("profile")
        if not is_valid_story_profile(story_profile):
            raise ProfileCacheError(
                f"Profile for {book['title']!r} is missing or invalid."
            )
        if story_profile.title != book["title"]:
            raise ProfileCacheError(
                f"Profile title does not match book {book['title']!r}."
            )
    embedding_sizes = {
        len(book["profile"].embedding)
        for book in valid_books
    }
    if len(embedding_sizes) != 1:
        raise ProfileCacheError(
            "Profile artifact contains inconsistent embedding dimensions."
        )
    return valid_books


def _build_profiles(books, output_path):
    normalized_books, issues = validate_books(books)
    log_book_issues(issues)
    if not normalized_books:
        raise ValueError("At least one valid book record is required.")

    prepared_books = [
        {
            "title": book["title"],
            "author": book["author"],
            "description": book["description"],
            "profile": create_story_profile(
                book["title"],
                book["description"],
            ),
        }
        for book in normalized_books
    ]

    artifact = {
        "format_version": PROFILE_FORMAT_VERSION,
        "profile_version": PROFILE_VERSION,
        "source_digest": _source_digest(normalized_books),
        "books": [
            {
                **{key: value for key, value in book.items() if key != "profile"},
                "profile": asdict(book["profile"]),
            }
            for book in prepared_books
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(artifact, file, ensure_ascii=False)
        temporary_path.replace(output_path)
    except OSError:
        logger.exception("Could not write profile artifact to %s", output_path)
        raise
    return prepared_books


def build_book_profiles(books):
    return _build_profiles(books, PROFILE_FILE)


def build_api_book_profiles(books):
    return _build_profiles(books, API_PROFILE_FILE)


def _load_profiles(path, source_books=None):
    try:
        with path.open("r", encoding="utf-8") as file:
            artifact = json.load(file)
    except FileNotFoundError as error:
        raise ProfileCacheError(
            f"Profile artifact does not exist: {path}"
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProfileCacheError(
            f"Could not read profile artifact {path}: {error}"
        ) from error

    if not isinstance(artifact, dict):
        raise ProfileCacheError(
            f"Profile artifact {path} uses an unsupported legacy format."
        )
    if artifact.get("format_version") != PROFILE_FORMAT_VERSION:
        raise ProfileCacheError(
            f"Profile artifact {path} has an unsupported format version."
        )
    if artifact.get("profile_version") != PROFILE_VERSION:
        raise ProfileCacheError(
            f"Profile artifact {path} was built with an outdated profile version."
        )
    source_digest = artifact.get("source_digest")
    if (
        not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_digest
        )
    ):
        raise ProfileCacheError(
            f"Profile artifact {path} has an invalid source fingerprint."
        )

    serialized_books = artifact.get("books")
    if not isinstance(serialized_books, list):
        raise ProfileCacheError("Profile artifact books must be a list.")

    profile_fields = {
        "title",
        "emotions",
        "emotional_experiences",
        "themes",
        "character_traits",
        "tone",
        "embedding",
    }
    prepared_books = []
    for item in serialized_books:
        if not isinstance(item, dict):
            raise ProfileCacheError(
                "Profile artifact contains a non-object book record."
            )
        serialized_profile = item.get("profile")
        if (
            not isinstance(serialized_profile, dict)
            or set(serialized_profile) != profile_fields
        ):
            raise ProfileCacheError(
                "Profile artifact contains an invalid story profile shape."
            )
        try:
            story_profile = StoryProfile(**serialized_profile)
        except (TypeError, ValueError) as error:
            raise ProfileCacheError(
                f"Profile artifact contains an invalid story profile: {error}"
            ) from error
        prepared_books.append({
            "title": item.get("title"),
            "author": item.get("author"),
            "description": item.get("description"),
            "profile": story_profile,
        })

    prepared_books = _validate_prepared_books(prepared_books)
    if _source_digest(prepared_books) != source_digest:
        raise ProfileCacheError(
            f"Profile artifact {path} does not match its stored fingerprint."
        )
    if source_books is not None:
        expected_books, issues = validate_books(source_books)
        log_book_issues(issues)
        if issues or _source_digest(expected_books) != artifact.get(
            "source_digest"
        ):
            raise ProfileCacheError(
                f"Profile artifact {path} does not match its source books."
            )
    return prepared_books


def load_book_profiles(books=None):
    if books is None:
        books = load_books()
    return _load_profiles(PROFILE_FILE, source_books=books)


def load_api_book_profiles():
    return _load_profiles(API_PROFILE_FILE)


def load_profiles_or_local_books():
    """Use cached profiles when available; otherwise use local books."""
    try:
        return load_api_book_profiles(), None
    except ProfileCacheError as error:
        books = load_books()
        reason = (
            f"{error} Local book data will be ranked with TF-IDF instead."
        )
        logger.info(reason)
        return books, reason
