from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "He finally finds people who accept him for who he is.",
    "A lonely teenager discovers a group of friends where he finally feels like he belongs.",
    "He becomes successful and earns the respect of everyone around him.",
    "A detective solves a difficult murder case and catches the killer."
]

embeddings = model.encode(texts)

similarities = cosine_similarity(embeddings)

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        print(f"\n{i+1} vs {j+1}")
        print(f"{similarities[i][j]:.3f}")
        print(f"  {texts[i]}")
        print(f"  {texts[j]}")