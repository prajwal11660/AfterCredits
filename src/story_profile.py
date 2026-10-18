import math
from dataclasses import dataclass, field
from typing import Dict, List

from emotional_experience import extract_emotional_experiences
from model_runtime import (
    classify_emotions,
    classify_zero_shot,
    create_semantic_embedding,
)
from tone import extract_tone


PROFILE_VERSION = 2


# Themes AfterCredits currently cares about
THEMES = [
    "belonging",
    "friendship",
    "identity",
    "loneliness",
    "love",
    "family",
    "grief",
    "trauma",
    "growing up",
    "self discovery",
    "ambition",
    "redemption",
    "freedom",
    "betrayal",
    "survival",
]


# Character traits AfterCredits currently cares about
CHARACTER_TRAITS = [
    "socially awkward",
    "introverted",
    "self-doubt",
    "emotionally vulnerable",
    "introspective",
    "ambitious",
    "rebellious",
    "confident",
    "independent",
    "desperate for connection",
    "emotionally resilient",
    "manipulative",
]


def extract_character_traits(text):
    results = classify_zero_shot(text, CHARACTER_TRAITS)

    traits = {}

    for label, score in zip(results["labels"], results["scores"]):
        if score > 0.1:
            traits[label] = round(float(score), 3)

    return traits


def extract_emotions(text):
    results = classify_emotions(text)

    emotions = {}

    for emotion in results[0]:
        if emotion["score"] > 0.1:
            emotions[emotion["label"]] = round(
                float(emotion["score"]), 3
            )

    return emotions


def extract_themes(text):
    results = classify_zero_shot(text, THEMES)

    themes = {}

    for label, score in zip(results["labels"], results["scores"]):
        if score > 0.1:
            themes[label] = round(float(score), 3)

    return themes


def create_embedding(text):
    return create_semantic_embedding(text)


@dataclass
class StoryProfile:
    title: str
    emotions: Dict[str, float] = field(default_factory=dict)
    emotional_experiences: Dict[str, float] = field(default_factory=dict)
    themes: Dict[str, float] = field(default_factory=dict)
    character_traits: Dict[str, float] = field(default_factory=dict)

    # Tone contains two groups:
    # atmosphere and narrative_feel
    tone: Dict[str, Dict[str, float]] = field(default_factory=dict)

    embedding: List[float] = field(default_factory=list)


def create_story_profile(title, text):
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Story title must be a non-empty string.")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Story description must be a non-empty string.")

    title = title.strip()
    text = text.strip()
    themes = extract_themes(text)
    character_traits = extract_character_traits(text)
    emotions = extract_emotions(text)
    emotional_experiences = extract_emotional_experiences(text)
    embedding = create_embedding(text)
    tone = extract_tone(text)

    profile = StoryProfile(
        title=title,
        emotions=emotions,
        emotional_experiences=emotional_experiences,
        themes=themes,
        character_traits=character_traits,
        tone=tone,
        embedding=embedding
    )
    if not is_valid_story_profile(profile):
        from model_runtime import ModelUnavailableError

        raise ModelUnavailableError(
            "Story profiling returned an invalid profile."
        )
    return profile


def is_valid_story_profile(profile):
    if not isinstance(getattr(profile, "title", None), str):
        return False
    if not profile.title.strip():
        return False

    for name in (
        "emotions",
        "emotional_experiences",
        "themes",
        "character_traits",
    ):
        values = getattr(profile, name, None)
        if not isinstance(values, dict):
            return False
        if any(
            not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
            for score in values.values()
        ):
            return False

    tone = getattr(profile, "tone", None)
    if not isinstance(tone, dict):
        return False
    for group in tone.values():
        if not isinstance(group, dict) or any(
            not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
            for score in group.values()
        ):
            return False

    embedding = getattr(profile, "embedding", None)
    return (
        isinstance(embedding, (list, tuple))
        and bool(embedding)
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in embedding
        )
    )


if __name__ == "__main__":

    movie_text = """
    A socially awkward teenager enters high school feeling lonely
    and isolated. He forms meaningful friendships, experiences love,
    struggles with painful memories, and slowly discovers a sense
    of belonging and his own identity.
    """

    profile = create_story_profile(
        "The Perks of Being a Wallflower",
        movie_text
    )

    print("\nStory Profile:")
    print(f"Title: {profile.title}")

    print("\nThemes:")
    for theme, score in profile.themes.items():
        print(f"{theme:20} {score:.3f}")

    print("\nCharacter Traits:")
    for trait, score in profile.character_traits.items():
        print(f"{trait:30} {score:.3f}")

    print("\nEmotions:")
    for emotion, score in profile.emotions.items():
        print(f"{emotion:20} {score:.3f}")

    print("\nEmotional Experiences:")
    for experience, score in profile.emotional_experiences.items():
        print(f"{experience:25} {score:.3f}")

    print("\nAtmosphere:")
    for label, score in profile.tone.get("atmosphere", {}).items():
        print(f"{label:20} {score:.3f}")

    print("\nNarrative Feel:")
    for label, score in profile.tone.get("narrative_feel", {}).items():
        print(f"{label:20} {score:.3f}")

    print(f"\nEmbedding dimensions: {len(profile.embedding)}")