from emotional_experience import extract_emotional_experiences
from similarity import dictionary_similarity

stories = {
    "Movie": """
    A socially awkward teenager enters high school feeling lonely
    and isolated. He forms meaningful friendships, experiences love,
    struggles with painful memories, and slowly discovers a sense
    of belonging and his own identity.
    """,

    "Book A": """
    A young student arrives at college feeling completely alone.
    An unexpected friendship helps him overcome isolation,
    understand himself, and find a place where he belongs.
    """,

    "Book B": """
    A detective investigates a series of murders in a futuristic
    city controlled by artificial intelligence.
    """,

    "Book C": """
    Two emotionally damaged people slowly fall in love while
    struggling with painful experiences from their past.
    """
}

profiles = {}
for title, text in stories.items():

    print(f"\n{'=' * 40}")
    print(title)
    print('=' * 40)

    experiences = extract_emotional_experiences(text)

    profiles[title] = experiences

    experiences = extract_emotional_experiences(text)

    for emotion, score in experiences.items():
        print(f"{emotion:25} {score:.3f}")

movie = profiles["Movie"]

print("\n" + "=" * 40)
print("Emotional Experience Similarity")
print("=" * 40)

for book in ["Book A", "Book B", "Book C"]:

    similarity = dictionary_similarity(
        movie,
        profiles[book]
    )

    print(f"{book}: {similarity:.3f}")        