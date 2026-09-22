def strength_label(score):
    if score >= 0.80:
        return "Very strong match"
    elif score >= 0.65:
        return "Strong match"
    elif score >= 0.50:
        return "Moderate match"
    else:
        return "Light match"


def shared_concepts(movie_profile, book_profile):
    def get_shared(movie_data, book_data, limit=5):
        shared = []

        for key in movie_data:
            if key in book_data:
                strength = min(movie_data[key], book_data[key])
                shared.append((key, strength))

        shared.sort(key=lambda x: x[1], reverse=True)

        return [item[0] for item in shared[:limit]]

    return {
        "themes": get_shared(
            movie_profile.themes,
            book_profile.themes
        ),
        "characters": get_shared(
            movie_profile.character_traits,
            book_profile.character_traits
        ),
        "emotional_experience": get_shared(
            movie_profile.emotional_experiences,
            book_profile.emotional_experiences
        )
    }


def explain_recommendation(
    recommendation,
    movie_profile,
    book_profile
):
    similarities = recommendation["similarities"]

    shared = shared_concepts(
        movie_profile,
        book_profile
    )

    return {
        "title": recommendation["title"],
        "score": recommendation["score"],
        "signals": {
            "emotional_experience": {
                "score": similarities["emotional_experience"],
                "label": strength_label(
                    similarities["emotional_experience"]
                ),
                "shared": shared["emotional_experience"]
            },
            "characters": {
                "score": similarities["characters"],
                "label": strength_label(
                    similarities["characters"]
                ),
                "shared": shared["characters"]
            },
            "themes": {
                "score": similarities["themes"],
                "label": strength_label(
                    similarities["themes"]
                ),
                "shared": shared["themes"]
            },
            "story_meaning": {
                "score": similarities["embedding"],
                "label": strength_label(
                    similarities["embedding"]
                )
            }
        }
    }