import json
import logging
from pathlib import Path

from book_validation import log_book_issues, validate_books


logger = logging.getLogger(__name__)


def load_books():
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "books.json"

    with open(file_path, "r", encoding="utf-8") as file:
        raw_books = json.load(file)

    books, issues = validate_books(raw_books)
    log_book_issues(issues)
    if not books:
        raise ValueError(
            f"No valid book records were found in {file_path}."
        )
    if issues:
        logger.warning(
            "Loaded %d valid books; skipped %d invalid records.",
            len(books),
            len(issues),
        )
    return books