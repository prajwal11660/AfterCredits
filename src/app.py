import sys
from pathlib import Path

import streamlit as st


SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from book_profiles import (
    load_profiles_or_local_books,
)
from recommender import recommend_books
from explanation import explain_recommendation

st.set_page_config(
    page_title="AfterCredits",
    page_icon="🎬",
    layout="wide"
)


st.title("🎬 AfterCredits")
st.write("The story doesn't end when the credits roll.")

st.write(
    "Find books that match the emotional experience of a movie — "
    "not just its genre or plot."
)


movie_title = st.text_input(
    "Movie title",
    placeholder="e.g. Into the Wild"
)


movie_description = st.text_area(
    "What is the movie about?",
    placeholder=(
        "Describe what the movie feels like. "
        "It can be just one sentence."
    ),
    height=150
)


top_k = st.slider(
    "Number of recommendations",
    3,
    10,
    5
)


if st.button("Find My Books", type="primary"):

    if not movie_title.strip():
        st.warning("Please enter a movie title.")

    elif not movie_description.strip():
        st.warning("Please describe the movie.")

    else:

        try:

            with st.spinner("Understanding the movie..."):
                profile_error = None

                @st.cache_resource
                def load_api_books():
                    return load_profiles_or_local_books()

                books, profile_error = load_api_books()

                result = recommend_books(
                    movie_title,
                    movie_description,
                    books,
                    top_k=top_k,
                    model_unavailable_reason=profile_error,
                )

                movie_profile = result["movie_profile"]
                recommendations = result["recommendations"]

            if result["method"] == "lexical_tfidf":
                st.warning(
                    "Using fast local TF-IDF text matching instead of "
                    "neural story profiles. "
                    f"Reason: {result['fallback_reason']}"
                )

            if not recommendations:
                st.warning(
                    "The local catalog has no books with a strong enough "
                    "match. Try adding more detail or a different movie."
                )
            else:
                st.success(
                    f"Found {len(recommendations)} recommendations."
                )
                st.divider()

            for index, book in enumerate(recommendations, start=1):
                st.subheader(
                    f"{index}. {book['title']}"
                )

                st.caption(
                    f"by {book['author']}  •  "
                    f"Match score: {book['score']:.3f}"
                )

                st.write(book["description"])

                with st.expander("Why this book?"):
                    if book["method"] == "lexical_tfidf":
                        matched_terms = book["matched_terms"]
                        if matched_terms:
                            st.write(
                                "Matching words: "
                                + ", ".join(matched_terms)
                            )
                        else:
                            st.write(
                                "No direct keywords matched. This result "
                                "may be a weak fit for your description."
                            )
                        continue

                    explanation = explain_recommendation(
                        book,
                        movie_profile,
                        book["profile"]
                    )

                    signals = explanation["signals"]

                    st.write(
                        f"**Emotional experience:** "
                        f"{signals['emotional_experience']['label']}"
                    )

                    emotions = signals[
                        "emotional_experience"
                    ]["shared"]

                    if emotions:
                        st.write(
                            "Shared experiences: "
                            + ", ".join(emotions)
                        )

                    st.write(
                        f"**Characters:** "
                        f"{signals['characters']['label']}"
                    )

                    characters = signals[
                        "characters"
                    ]["shared"]

                    if characters:
                        st.write(
                            "Shared traits: "
                            + ", ".join(characters)
                        )

                    st.write(
                        f"**Themes:** "
                        f"{signals['themes']['label']}"
                    )

                    themes = signals[
                        "themes"
                    ]["shared"]

                    if themes:
                        st.write(
                            "Shared themes: "
                            + ", ".join(themes)
                        )

                    st.write(
                        f"**Story meaning:** "
                        f"{signals['story_meaning']['label']}"
                    )

        except Exception as error:

            st.error("Something went wrong.")

            st.exception(error)