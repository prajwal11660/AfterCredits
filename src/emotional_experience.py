from model_runtime import classify_zero_shot


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

    results = classify_zero_shot(text, EMOTIONAL_EXPERIENCES)

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