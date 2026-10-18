import logging
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from book_profiles import load_profiles_or_local_books
from open_library import discover_books, merge_book_candidates
from recommender import recommend_books


logger = logging.getLogger(__name__)
app = FastAPI(
    title="AfterCredits API",
    description="Find books that match the story and emotional experience of a movie.",
    version="1.0.0",
)


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    movie_title: str = Field(min_length=1, max_length=200)
    movie_description: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)


class RecommendedBook(BaseModel):
    title: str
    author: str
    description: str
    score: float
    method: Literal["model", "lexical_tfidf"]
    matched_terms: list[str] = Field(default_factory=list)
    source: str
    source_url: str | None
    contributions: dict[str, float]
    similarities: dict[str, float]


class RecommendationResponse(BaseModel):
    method: Literal["model", "lexical_tfidf"]
    fallback_reason: str | None
    recommendations: list[RecommendedBook]


@lru_cache(maxsize=1)
def get_book_catalog():
    """Load the active profile cache or local book catalog once per process."""
    return load_profiles_or_local_books()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/recommendations",
    response_model=RecommendationResponse,
)
def create_recommendations(
    request: RecommendationRequest,
) -> RecommendationResponse:
    try:
        books, profile_error = get_book_catalog()
    except (OSError, ValueError) as error:
        logger.exception("Could not load the book catalog.")
        raise HTTPException(
            status_code=503,
            detail="The book catalog is unavailable.",
        ) from error

    discovered_books, query_expansion, _search_errors = discover_books(
        request.movie_title,
        request.movie_description,
    )
    books = merge_book_candidates(books, discovered_books)

    try:
        result = recommend_books(
            request.movie_title,
            request.movie_description,
            books,
            top_k=request.top_k,
            model_unavailable_reason=(
                "Movie-specific candidates are ranked with TF-IDF."
                if discovered_books
                else profile_error
            ),
            query_expansion=query_expansion,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    fallback_reason = None
    if result["method"] == "lexical_tfidf":
        fallback_reason = (
            "Neural profiles are unavailable; recommendations were ranked "
            "with local TF-IDF matching."
        )

    return RecommendationResponse(
        method=result["method"],
        fallback_reason=fallback_reason,
        recommendations=[
            RecommendedBook(
                title=book["title"],
                author=book["author"],
                description=book["description"],
                score=book["score"],
                method=book["method"],
                matched_terms=book.get("matched_terms", []),
                source=book.get("source", "Local catalog"),
                source_url=book.get("source_url"),
                contributions=book["contributions"],
                similarities=book["similarities"],
            )
            for book in result["recommendations"]
        ],
    )
