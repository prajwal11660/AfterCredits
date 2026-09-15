from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)

texts = [
    "He feels completely alone despite being surrounded by people.",
    "He finally finds people who accept him for who he is.",
    "He struggles with painful memories from his past."
]

for text in texts:
    print("\nTEXT:", text)

    results = classifier(text)

    for emotion in results[0]:
        if emotion["score"] > 0.1:
            print(f"{emotion['label']:15} {emotion['score']:.3f}")