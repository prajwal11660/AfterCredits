
from story_profile import create_story_profile
from similarity import dictionary_similarity, tone_similarity
from scorer import calculate_score
from embedding_similarity import cosine_similarity


movie_text = """
A socially awkward teenager enters high school feeling lonely
and isolated. He forms meaningful friendships, experiences love,
struggles with painful memories, and slowly discovers a sense
of belonging and his own identity.
"""

book_a_text = """
A young student arrives at college feeling completely alone.
An unexpected friendship helps him overcome isolation,
understand himself, and find a place where he belongs.
"""

book_b_text = """
A detective investigates a series of murders in a futuristic
city controlled by artificial intelligence.
"""

book_c_text = """
Two emotionally damaged people slowly fall in love while
struggling with painful experiences from their past.
"""


# Create Story Profiles

movie = create_story_profile("Movie", movie_text)
book_a = create_story_profile("Book A", book_a_text)
book_b = create_story_profile("Book B", book_b_text)
book_c = create_story_profile("Book C", book_c_text)

print("\nTone Similarity:")

print(
    f"Book A: {tone_similarity(movie.tone, book_a.tone):.3f}"
)

print(
    f"Book B: {tone_similarity(movie.tone, book_b.tone):.3f}"
)

print(
    f"Book C: {tone_similarity(movie.tone, book_c.tone):.3f}"
)


# Theme Similarity

print("Theme Similarity:")

print(
    "Book A:",
    dictionary_similarity(movie.themes, book_a.themes)
)

print(
    "Book B:",
    dictionary_similarity(movie.themes, book_b.themes)
)

print(
    "Book C:",
    dictionary_similarity(movie.themes, book_c.themes)
)


# Character Similarity

print("\nCharacter Similarity:")

print(
    "Book A:",
    dictionary_similarity(
        movie.character_traits,
        book_a.character_traits
    )
)

print(
    "Book B:",
    dictionary_similarity(
        movie.character_traits,
        book_b.character_traits
    )
)

print(
    "Book C:",
    dictionary_similarity(
        movie.character_traits,
        book_c.character_traits
    )
)


# Basic Emotion Similarity

print("\nBasic Emotion Similarity:")

print(
    "Book A:",
    dictionary_similarity(
        movie.emotions,
        book_a.emotions
    )
)

print(
    "Book B:",
    dictionary_similarity(
        movie.emotions,
        book_b.emotions
    )
)

print(
    "Book C:",
    dictionary_similarity(
        movie.emotions,
        book_c.emotions
    )
)


# Emotional Experience Similarity

print("\nEmotional Experience Similarity:")

print(
    "Book A:",
    dictionary_similarity(
        movie.emotional_experiences,
        book_a.emotional_experiences
    )
)

print(
    "Book B:",
    dictionary_similarity(
        movie.emotional_experiences,
        book_b.emotional_experiences
    )
)

print(
    "Book C:",
    dictionary_similarity(
        movie.emotional_experiences,
        book_c.emotional_experiences
    )
)


# Embedding Similarity

print("\nEmbedding Similarity:")

print(
    "Book A:",
    round(
        cosine_similarity(
            movie.embedding,
            book_a.embedding
        ),
        3
    )
)

print(
    "Book B:",
    round(
        cosine_similarity(
            movie.embedding,
            book_b.embedding
        ),
        3
    )
)

print(
    "Book C:",
    round(
        cosine_similarity(
            movie.embedding,
            book_c.embedding
        ),
        3
    )
)


# Final Scores

print("\nFinal Scores:")

book_a_result = calculate_score(
    dictionary_similarity(movie.themes, book_a.themes),
    dictionary_similarity(movie.character_traits, book_a.character_traits),
    dictionary_similarity(
        movie.emotional_experiences,
        book_a.emotional_experiences
    ),
    cosine_similarity(movie.embedding, book_a.embedding),
    tone_similarity(movie.tone, book_a.tone)
)
print("Book A:", round(book_a_result["score"], 3))

book_b_result = calculate_score(
    dictionary_similarity(movie.themes, book_b.themes),
    dictionary_similarity(
        movie.character_traits,
        book_b.character_traits
    ),
    dictionary_similarity(
        movie.emotional_experiences,
        book_b.emotional_experiences
    ),
    cosine_similarity(
        movie.embedding,
        book_b.embedding
    ),
    tone_similarity(movie.tone, book_b.tone)
)

book_c_result = calculate_score(
    dictionary_similarity(movie.themes, book_c.themes),
    dictionary_similarity(
        movie.character_traits,
        book_c.character_traits
    ),
    dictionary_similarity(
        movie.emotional_experiences,
        book_c.emotional_experiences
    ),
    cosine_similarity(
        movie.embedding,
        book_c.embedding
    ),
    tone_similarity(movie.tone, book_c.tone)
)

print("\nFinal Scores:")

for name, result in [
    ("Book A", book_a_result),
    ("Book B", book_b_result),
    ("Book C", book_c_result)
]:
    print(f"\n{name}: {result['score']:.3f}")

    print("  Contributions:")

    for component, value in result["contributions"].items():
        print(f"    {component:25} {value:.3f}")


print("Book A:", round(book_a_result["score"], 3))
print("Book B:", round(book_b_result["score"], 3))
print("Book C:", round(book_c_result["score"], 3))