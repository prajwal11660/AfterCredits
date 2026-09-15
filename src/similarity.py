def dictionary_similarity(profile_a, profile_b):
    """
    Compare two feature dictionaries using cosine similarity.

    Missing features are treated as 0.
    """

    all_keys = set(profile_a) | set(profile_b)

    if not all_keys:
        return 0.0

    vector_a = []
    vector_b = []

    for key in all_keys:
        vector_a.append(profile_a.get(key, 0.0))
        vector_b.append(profile_b.get(key, 0.0))

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)

def tone_similarity(tone_a, tone_b):
    """
    Compare two tone profiles.

    Tone is divided into:
    - atmosphere
    - narrative feel

    Each part is compared separately and then averaged.
    """

    atmosphere_similarity = dictionary_similarity(
        tone_a.get("atmosphere", {}),
        tone_b.get("atmosphere", {})
    )

    narrative_similarity = dictionary_similarity(
        tone_a.get("narrative_feel", {}),
        tone_b.get("narrative_feel", {})
    )

    return (
        atmosphere_similarity + narrative_similarity
    ) / 2