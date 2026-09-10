from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movie = """
A lonely teenager struggles to fit in at school,
forms meaningful friendships, deals with emotional trauma,
and slowly discovers a sense of belonging.
"""


books = {
    "Book A": """
    A lonely teenager struggles to fit in at school,
    forms meaningful friendships, deals with emotional trauma, 
    and slowly discovers a sense of belonging.
    """,

    "Book B": """
    A detective investigates a series of murders in a
    futuristic city controlled by artificial intelligence.
    """,

    "Book C": """
    Two emotionally damaged people slowly fall in love
    while struggling with painful experiences from their past.
    """
}
vectorizer = TfidfVectorizer()

all_text = [movie] + list(books.values())

vectors = vectorizer.fit_transform(all_text)
print(vectorizer.get_feature_names_out())

similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

for book, score in zip(books.keys(), similarities):
    print(f"{book}: {score:.3f}")