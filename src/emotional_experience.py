from transformers import pipeline


emotional_experience_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


EMOTIONAL_EXPERIENCES = [
    "loneliness",
    "belonging",
    "connection",
    "isolation",
    "vulnerability",
    "hope",
    "nostalgia",
    "grief",
    "heartbreak",
    "fear",
    "comfort",
    "acceptance",
    "self doubt",
    "joy",
    "love",
    "loss",
    "confusion",
    "alienation",
    "self discovery",
    "emotional healing",
]


def extract_emotional_experiences(text):

    results = emotional_experience_classifier(
        text,
        candidate_labels=EMOTIONAL_EXPERIENCES,
        multi_label=True
    )

    experiences = {}

    for label, score in zip(
        results["labels"],
        results["scores"]
    ):
        if score > 0.1:
            experiences[label] = round(
                float(score), 3
            )

    return experiences