import logging


logger = logging.getLogger(__name__)


def validate_books(books):
    """Normalize usable book records and return validation issues."""
    if not isinstance(books, list):
        raise TypeError("Books must be provided as a list of records.")

    valid_books = []
    issues = []

    for index, book in enumerate(books):
        identifier = (
            book.get("title", f"record {index + 1}")
            if isinstance(book, dict)
            else f"record {index + 1}"
        )
        errors = []

        if not isinstance(book, dict):
            errors.append("record must be an object")
        else:
            title = book.get("title")
            description = book.get("description")
            author = book.get("author", "Unknown")

            if not isinstance(title, str) or not title.strip():
                errors.append("title must be a non-empty string")
            if not isinstance(description, str) or not description.strip():
                errors.append("description must be a non-empty string")
            if author is None or (
                isinstance(author, str) and not author.strip()
            ):
                author = "Unknown"
            elif not isinstance(author, str):
                errors.append("author must be a string when provided")

            if not errors:
                valid_books.append({
                    **book,
                    "title": title.strip(),
                    "description": description.strip(),
                    "author": author.strip(),
                })

        if errors:
            issue = f"{identifier!r}: {', '.join(errors)}"
            issues.append(issue)

    return valid_books, issues


def log_book_issues(issues):
    for issue in issues:
        logger.warning("Skipping invalid book record: %s", issue)
