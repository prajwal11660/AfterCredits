from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text = """
A socially awkward teenager constantly doubts himself.
He feels isolated from the people around him, but deeply
wants to connect with others. He is sensitive, introspective,
emotionally vulnerable, and slowly becomes more confident
after forming meaningful friendships.
"""

traits = [
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
    "manipulative"
]

results = classifier(
    text,
    candidate_labels=traits,
    multi_label=True
)

for label, score in zip(results["labels"], results["scores"]):
    print(f"{label:30} {score:.3f}")