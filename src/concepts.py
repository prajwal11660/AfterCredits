from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text = """
A lonely teenager discovers a group of friends where he finally
feels accepted and like he belongs. Through these friendships,
he slowly becomes more confident and begins to understand himself.
"""

concepts = [
    "loneliness",
    "belonging",
    "friendship",
    "romantic love",
    "success",
    "crime",
    "identity",
    "trauma",
]

results = classifier(
    text,
    candidate_labels=concepts,
    multi_label=True
)

for label, score in zip(results["labels"], results["scores"]):
    print(f"{label:20} {score:.3f}")