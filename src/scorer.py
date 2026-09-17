def calculate_score(
    theme_similarity,
    character_similarity,
    emotional_experience_similarity,
    embedding_similarity,
    tone_similarity,
    theme_weight=0.15,
    character_weight=0.25,
    emotional_experience_weight=0.30,
    embedding_weight=0.20,
    tone_weight=0.10
):
    """
    Calculate the final AfterCredits recommendation score.

    The score is a weighted combination of:
    - themes
    - character traits
    - emotional experiences
    - semantic embeddings
    - tone
    """

    theme_contribution = theme_similarity * theme_weight
    character_contribution = character_similarity * character_weight
    emotional_contribution = (
        emotional_experience_similarity
        * emotional_experience_weight
    )
    embedding_contribution = embedding_similarity * embedding_weight
    tone_contribution = tone_similarity * tone_weight

    total_score = (
        theme_contribution
        + character_contribution
        + emotional_contribution
        + embedding_contribution
        + tone_contribution
    )

    return {
        "score": total_score,
        "contributions": {
            "themes": theme_contribution,
            "characters": character_contribution,
            "emotional_experience": emotional_contribution,
            "embedding": embedding_contribution,
            "tone": tone_contribution
        }
    }

def calculate_score_no_tone(
    theme_similarity,
    character_similarity,
    emotional_experience_similarity,
    embedding_similarity
):
    """
    Experimental scorer with tone removed.

    Tone is temporarily excluded to test whether
    it helps or hurts ranking performance.
    """

    theme_weight = 0.15
    character_weight = 0.25
    emotional_experience_weight = 0.30
    embedding_weight = 0.30

    theme_contribution = (
        theme_similarity * theme_weight
    )

    character_contribution = (
        character_similarity * character_weight
    )

    emotional_contribution = (
        emotional_experience_similarity
        * emotional_experience_weight
    )

    embedding_contribution = (
        embedding_similarity
        * embedding_weight
    )

    total_score = (
        theme_contribution
        + character_contribution
        + emotional_contribution
        + embedding_contribution
    )

    return {
        "score": total_score,
        "contributions": {
            "themes": theme_contribution,
            "characters": character_contribution,
            "emotional_experience": emotional_contribution,
            "embedding": embedding_contribution,
            "tone": 0.0
        }
    }

