from story_profile import create_story_profile
from similarity import dictionary_similarity
from embedding_similarity import cosine_similarity
from scorer import calculate_score_no_tone


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


def recommend_books(movie_title, movie_description, books, top_k=5):
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

    movie_profile = create_story_profile(
        movie_title,
        movie_description
    )

    recommendations = []

    for book in books:
        book_profile = create_story_profile(
            book["title"],
            book["description"]
        )

        result = calculate_book_score(
            movie_profile,
            book_profile
        )

        recommendations.append({
        "title": book["title"],
        "description": book["description"],
        "score": result["score"],
        "contributions": result["contributions"],
        "similarities": result["similarities"],
        "profile": book_profile
        })

    recommendations.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return {
    "movie_profile": movie_profile,
    "recommendations": recommendations[:top_k]
    }