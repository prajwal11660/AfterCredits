from transformers import pipeline

tone_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

ATMOSPHERE = [
    "warm",
    "dark",
    "hopeful",
    "melancholic",
    "somber",
    "unsettling"
]

NARRATIVE_FEEL = [
    "introspective",
    "tense",
    "bittersweet",
    "uplifting",
    "nostalgic",
    "gritty",
    "whimsical",
    "romantic"
]


def classify_tones(text, labels):
    results = tone_classifier(
        text,
        candidate_labels=labels,
        multi_label=True
    )

    tones = {}

    for label, score in zip(results["labels"], results["scores"]):
        if score > 0.1:
            tones[label] = round(float(score), 3)

    return tones


def extract_tone(text):
    return {
        "atmosphere": classify_tones(text, ATMOSPHERE),
        "narrative_feel": classify_tones(text, NARRATIVE_FEEL)
    }


if __name__ == "__main__":
    text = """
   Two emotionally damaged people slowly fall in love while
struggling with painful experiences from their past.
    """ 

    tone = extract_tone(text)

    print("\nAtmosphere:")
    for label, score in tone["atmosphere"].items():
        print(f"{label}: {score}")

    print("\nNarrative Feel:")
    for label, score in tone["narrative_feel"].items():
        print(f"{label}: {score}")