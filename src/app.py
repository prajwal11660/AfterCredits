import sys
from pathlib import Path

import streamlit as st


SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from book_profiles import (
    load_profiles_or_local_books,
)
from open_library import discover_books, merge_book_candidates
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
                @st.cache_resource
                def load_local_books():
                    return load_profiles_or_local_books()

                @st.cache_data(ttl=21600, show_spinner=False)
                def discover_movie_books(title, description):
                    return discover_books(title, description)

                local_books, _profile_error = load_local_books()
                discovered_books, query_expansion, search_errors = (
                    discover_movie_books(movie_title, movie_description)
                )
                books = merge_book_candidates(
                    local_books,
                    discovered_books,
                )
                fallback_reason = (
                    "Movie-specific Open Library candidates are ranked "
                    "with local TF-IDF matching."
                )

                result = recommend_books(
                    movie_title,
                    movie_description,
                    books,
                    top_k=top_k,
                    model_unavailable_reason=fallback_reason,
                    query_expansion=query_expansion,
                )

                movie_profile = result["movie_profile"]
                recommendations = result["recommendations"]

            if result["method"] == "lexical_tfidf":
                st.warning(
                    "Suggestions are retrieved from the local catalog and "
                    "Open Library, then ranked against the plot using "
                    "fast TF-IDF matching."
                )
            if discovered_books:
                st.caption(
                    f"Compared {len(books)} books, including "
                    f"{len(discovered_books)} Open Library candidates."
                )
            elif search_errors:
                st.warning(
                    "Open Library could not be reached. Showing matches from "
                    "the local catalog only."
                )
            else:
                st.caption(
                    "Open Library found no additional candidates; "
                    "recommendations use the local catalog."
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
                    f"Match score: {book['score']:.3f}  •  "
                    f"Source: {book.get('source', 'Local catalog')}"
                )

                st.write(book["description"])
                if book.get("source_url"):
                    st.markdown(
                        f"[View on Open Library]({book['source_url']})"
                    )

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