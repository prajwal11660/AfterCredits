from dataclasses import dataclass, field
from typing import Dict, List

from transformers import pipeline
from sentence_transformers import SentenceTransformer

from emotional_experience import extract_emotional_experiences
from tone import extract_tone


# Emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)

# Semantic embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Zero-shot classifier
theme_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


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
    results = theme_classifier(
        text,
        candidate_labels=CHARACTER_TRAITS,
        multi_label=True
    )

    traits = {}

    for label, score in zip(results["labels"], results["scores"]):
        if score > 0.1:
            traits[label] = round(float(score), 3)

    return traits


def extract_emotions(text):
    results = emotion_classifier(text)

    emotions = {}

    for emotion in results[0]:
        if emotion["score"] > 0.1:
            emotions[emotion["label"]] = round(
                float(emotion["score"]), 3
            )

    return emotions


def extract_themes(text):
    results = theme_classifier(
        text,
        candidate_labels=THEMES,
        multi_label=True
    )

    themes = {}

    for label, score in zip(results["labels"], results["scores"]):
        if score > 0.1:
            themes[label] = round(float(score), 3)

    return themes


def create_embedding(text):
    embedding = embedding_model.encode(text)
    return embedding.tolist()


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
    themes = extract_themes(text)
    character_traits = extract_character_traits(text)
    emotions = extract_emotions(text)
    emotional_experiences = extract_emotional_experiences(text)
    embedding = create_embedding(text)
    tone = extract_tone(text)

    return StoryProfile(
        title=title,
        emotions=emotions,
        emotional_experiences=emotional_experiences,
        themes=themes,
        character_traits=character_traits,
        tone=tone,
        embedding=embedding
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