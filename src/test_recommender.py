from recommender import recommend_books
from explanation import explain_recommendation


books = [
    {
        "title": "Book A",
        "description": """
        A lonely young person struggling with trauma and identity
        slowly finds friendship, belonging, and emotional healing.
        """
    },
    {
        "title": "Book B",
        "description": """
        A brilliant but troubled young person struggles with
        self-doubt, isolation, and difficulty trusting others.
        """
    },
    {
        "title": "Book C",
        "description": """
        A young person leaves their ordinary life behind and
        discovers adventure, freedom, and a new sense of identity.
        """
    }
]


movie_description = """
A sensitive teenager entering high school struggles with trauma,
loneliness, identity, and feeling like an outsider. Through
friendship and connection with a group of friends, he gradually
finds belonging and begins to heal emotionally.
"""


result = recommend_books(
    "The Perks of Being a Wallflower",
    movie_description,
    books,
    top_k=3
)

movie_profile = result["movie_profile"]
recommendations = result["recommendations"]


print("\nAFTERCREDITS RECOMMENDATIONS")
print("=" * 40)


for index, book in enumerate(recommendations, start=1):

    explanation = explain_recommendation(
        book,
        movie_profile,
        book["profile"]
    )

    print(
        f"\n{index}. {explanation['title']} "
        f"→ {explanation['score']:.3f}"
    )

    print(
        "   Emotional experience:",
        explanation["signals"]["emotional_experience"]["label"]
    )

    print(
        "   Shared emotional experiences:",
        ", ".join(
            explanation["signals"]["emotional_experience"]["shared"]
        ) or "None"
    )

    print(
        "   Characters:",
        explanation["signals"]["characters"]["label"]
    )

    print(
        "   Shared character traits:",
        ", ".join(
            explanation["signals"]["characters"]["shared"]
        ) or "None"
    )

    print(
        "   Themes:",
        explanation["signals"]["themes"]["label"]
    )

    print(
        "   Shared themes:",
        ", ".join(
            explanation["signals"]["themes"]["shared"]
        ) or "None"
    )

    print(
        "   Story meaning:",
        explanation["signals"]["story_meaning"]["label"]
    )