from story_profile import create_story_profile


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


print("Creating movie profile...")
movie_profile = create_story_profile(
    "The Perks of Being a Wallflower",
    movie_text
)

print("Creating Book A profile...")
book_a_profile = create_story_profile(
    "Book A",
    book_a_text
)

print("Creating Book B profile...")
book_b_profile = create_story_profile(
    "Book B",
    book_b_text
)

print("Creating Book C profile...")
book_c_profile = create_story_profile(
    "Book C",
    book_c_text
)


print("\n==============================")
print("PROFILES CREATED SUCCESSFULLY")
print("==============================")

print(f"\nMovie: {movie_profile.title}")
print(f"  Themes: {len(movie_profile.themes)}")
print(f"  Character traits: {len(movie_profile.character_traits)}")
print(f"  Emotions: {len(movie_profile.emotions)}")
print(f"  Emotional experiences: {len(movie_profile.emotional_experiences)}")
print(f"  Embedding dimensions: {len(movie_profile.embedding)}")

print(f"\nBook A: {book_a_profile.title}")
print(f"  Themes: {len(book_a_profile.themes)}")
print(f"  Character traits: {len(book_a_profile.character_traits)}")
print(f"  Emotions: {len(book_a_profile.emotions)}")
print(f"  Embedding dimensions: {len(book_a_profile.embedding)}")

print(f"\nBook B: {book_b_profile.title}")
print(f"  Themes: {len(book_b_profile.themes)}")
print(f"  Character traits: {len(book_b_profile.character_traits)}")
print(f"  Emotions: {len(book_b_profile.emotions)}")
print(f"  Embedding dimensions: {len(book_b_profile.embedding)}")

print(f"\nBook C: {book_c_profile.title}")
print(f"  Themes: {len(book_c_profile.themes)}")
print(f"  Character traits: {len(book_c_profile.character_traits)}")
print(f"  Emotions: {len(book_c_profile.emotions)}")
print(f"  Embedding dimensions: {len(book_c_profile.embedding)}")


from similarity import dictionary_similarity

print("\n==============================")
print("Emotional Experience Similarity")
print("==============================")

for book in [book_a_profile, book_b_profile, book_c_profile]:

    score = dictionary_similarity(
        movie_profile.emotional_experiences,
        book.emotional_experiences
    )

    print(
        f"{book.title}: {score:.3f}"
    )