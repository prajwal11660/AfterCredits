from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


movies = [
    {
        "title": "The Perks of Being a Wallflower",
        "description": """
        A socially awkward teenager enters high school feeling lonely
        and isolated. He forms meaningful friendships, experiences love,
        struggles with painful memories, and slowly discovers a sense
        of belonging and his own identity.
        """
    }
]


books = [
    {
        "title": "Book A",
        "description": """
        A young student arrives at college feeling completely alone.
        An unexpected friendship helps him overcome isolation,
        understand himself, and find a place where he belongs.
        """
    },
    {
        "title": "Book B",
        "description": """
        A detective investigates a series of murders in a futuristic
        city controlled by artificial intelligence.
        """
    },
    {
        "title": "Book C",
        "description": """
        Two emotionally damaged people slowly fall in love while
        struggling with painful experiences from their past.
        """
    }
]


movie_description = movies[0]["description"]

book_descriptions = [book["description"] for book in books]

all_text = [movie_description] + book_descriptions

embeddings = model.encode(all_text)

movie_embedding = embeddings[0:1]
book_embeddings = embeddings[1:]

similarities = cosine_similarity(
    movie_embedding,
    book_embeddings
).flatten()

for book, score in zip(books, similarities):
    print(f"{book['title']}: {score:.3f}")