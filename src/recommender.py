import logging
import math

from book_validation import log_book_issues, validate_books
from lexical_similarity import (
    lexical_match_terms,
    lexical_similarities,
    title_similarity,
    token_coverage,
)
from model_runtime import ModelUnavailableError
from story_profile import (
    StoryProfile,
    create_story_profile,
    is_valid_story_profile,
)
from similarity import dictionary_similarity
from embedding_similarity import cosine_similarity
from scorer import calculate_score_no_tone


logger = logging.getLogger(__name__)
MIN_LEXICAL_SCORE = 0.05


def calculate_book_score(movie_profile, book_profile):
    """
    Calculate the AfterCredits similarity score between
    a movie and a book.
    """

    theme_similarity = dictionary_similarity(
        movie_profile.themes,
        book_profile.themes
    )

    character_similarity = dictionary_similarity(
        movie_profile.character_traits,
        book_profile.character_traits
    )

    emotional_experience_similarity = dictionary_similarity(
        movie_profile.emotional_experiences,
        book_profile.emotional_experiences
    )

    embedding_similarity = cosine_similarity(
        movie_profile.embedding,
        book_profile.embedding
    )

    result = calculate_score_no_tone(
        theme_similarity,
        character_similarity,
        emotional_experience_similarity,
        embedding_similarity
    )

    return {
        "score": result["score"],
        "contributions": result["contributions"],
        "similarities": {
            "themes": theme_similarity,
            "characters": character_similarity,
            "emotional_experience": emotional_experience_similarity,
            "embedding": embedding_similarity
        }
    }


def recommend_books(
    movie_title,
    movie_description,
    books,
    top_k=5,
    model_unavailable_reason=None,
    query_expansion="",
):
    """
    Recommend books based on the emotional and narrative
    experience of a movie.

    Parameters:
        movie_title: Title of the movie.
        movie_description: Description of the movie.
        books: List of dictionaries containing:
               title and description.
        top_k: Number of books to return.

    Returns:
        Ranked list of recommended books.
    """

    if not isinstance(movie_title, str) or not movie_title.strip():
        raise ValueError("Movie title must be a non-empty string.")
    if not isinstance(movie_description, str) or not movie_description.strip():
        raise ValueError("Movie description must be a non-empty string.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    valid_books, issues = validate_books(books)
    log_book_issues(issues)
    if not valid_books:
        raise ValueError("At least one valid book record is required.")
    if issues:
        logger.warning(
            "Using %d valid books; skipped %d invalid records.",
            len(valid_books),
            len(issues),
        )

    movie_title = movie_title.strip()
    movie_description = movie_description.strip()

    if model_unavailable_reason is not None:
        return _recommend_lexically(
            movie_title,
            movie_description,
            valid_books,
            top_k,
            str(model_unavailable_reason),
            query_expansion,
        )

    try:
        movie_profile = create_story_profile(
            movie_title,
            movie_description,
        )
        book_profiles = []
        for book in valid_books:
            profile = book.get("profile")
            if (
                not is_valid_story_profile(profile)
                or profile.title != book["title"]
            ):
                profile = create_story_profile(
                    book["title"],
                    book["description"],
                )
            book_profiles.append(profile)

        if any(
            len(profile.embedding) != len(movie_profile.embedding)
            for profile in book_profiles
        ):
            raise ModelUnavailableError(
                "Story profiles have inconsistent embedding dimensions."
            )

        recommendations = []
        for book, book_profile in zip(valid_books, book_profiles):
            result = calculate_book_score(movie_profile, book_profile)
            score = result["score"]
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ModelUnavailableError(
                    "Model-based scoring returned a non-finite score."
                )
            recommendations.append({
                "title": book["title"],
                "author": book["author"],
                "description": book["description"],
                "score": score,
                "contributions": result["contributions"],
                "similarities": result["similarities"],
                "profile": book_profile,
                "method": "model",
            })

        recommendations.sort(key=lambda item: item["score"], reverse=True)
        return {
            "movie_profile": movie_profile,
            "recommendations": recommendations[:top_k],
            "method": "model",
            "fallback_reason": None,
        }
    except ModelUnavailableError as error:
        logger.warning(
            "Model recommendation path unavailable; using TF-IDF fallback: %s",
            error,
        )
        return _recommend_lexically(
            movie_title,
            movie_description,
            valid_books,
            top_k,
            str(error),
            query_expansion,
        )


def _recommend_lexically(
    movie_title,
    movie_description,
    books,
    top_k,
    fallback_reason,
    query_expansion="",
):
    query = movie_description
    plot_documents = [
        f"{book['title']} {book['description']}"
        for book in books
    ]
    metadata_documents = [
        f"{book['title']} {' '.join(book.get('search_terms', []))}"
        for book in books
    ]
    plot_scores = lexical_similarities(query, plot_documents)
    title_scores = lexical_similarities(movie_title, plot_documents)
    topic_query = query_expansion if query_expansion.strip() else query
    metadata_scores = [
        token_coverage(topic_query, document)
        for document in metadata_documents
    ]
    fuzzy_title_scores = [
        title_similarity(movie_title, book["title"])
        for book in books
    ]
    scores = []
    for book, plot_score, metadata_score, title_score, fuzzy_score in zip(
        books,
        plot_scores,
        metadata_scores,
        title_scores,
        fuzzy_title_scores,
    ):
        if (
            plot_score < 0.035
            and metadata_score < 0.2
            and fuzzy_score < 0.7
        ):
            continue
        scores.append(
            (
                0.65 * plot_score
                + 0.20 * metadata_score
                + 0.05 * title_score
                + 0.10 * fuzzy_score,
                book,
            )
        )

    movie_profile = StoryProfile(title=movie_title)
    recommendations = []

    evidence_query = f"{query} {query_expansion}"
    for score, book in scores:
        if score < MIN_LEXICAL_SCORE:
            continue
        searchable_text = (
            f"{book['title']} {book['description']} "
            f"{' '.join(book.get('search_terms', []))}"
        )
        book_profile = StoryProfile(title=book["title"])
        recommendations.append({
            "title": book["title"],
            "author": book["author"],
            "description": book["description"],
            "score": score,
            "contributions": {"lexical_tfidf": score},
            "similarities": {
                "themes": 0.0,
                "characters": 0.0,
                "emotional_experience": 0.0,
                "embedding": 0.0,
                "lexical_tfidf": score,
            },
            "profile": book_profile,
            "method": "lexical_tfidf",
            "matched_terms": lexical_match_terms(
                evidence_query,
                searchable_text,
            ),
            "source": book.get("source", "Local catalog"),
            "source_url": book.get("source_url"),
        })

    recommendations.sort(key=lambda item: item["score"], reverse=True)
    return {
        "movie_profile": movie_profile,
        "recommendations": recommendations[:top_k],
        "method": "lexical_tfidf",
        "fallback_reason": fallback_reason,
    }