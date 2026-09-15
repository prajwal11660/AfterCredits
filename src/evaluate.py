from evaluation_data import EVALUATION_DATA

from story_profile import create_story_profile
from similarity import dictionary_similarity, tone_similarity
from embedding_similarity import cosine_similarity
from scorer import calculate_score


def calculate_book_score(movie_profile, book_profile):

    theme_similarity = dictionary_similarity(
        movie_profile.themes,
        book_profile.themes
    )

    character_similarity = dictionary_similarity(
        movie_profile.character_traits,
        book_profile.character_traits
    )

    emotional_experience_similarity = dictionary_similarity(
        movie_profile.emotional_experiences,
        book_profile.emotional_experiences
    )

    embedding_similarity = cosine_similarity(
        movie_profile.embedding,
        book_profile.embedding
    )

    tone_similarity_score = tone_similarity(
        movie_profile.tone,
        book_profile.tone
    )

    return calculate_score(
        theme_similarity,
        character_similarity,
        emotional_experience_similarity,
        embedding_similarity,
        tone_similarity_score
    )


def evaluate_movie(movie_data):

    movie_profile = create_story_profile(
        movie_data["movie"],
        movie_data["description"]
    )

    results = []

    for book in movie_data["books"]:

        book_profile = create_story_profile(
            book["title"],
            book["description"]
        )

        result = calculate_book_score(
            movie_profile,
            book_profile
        )

        results.append({
            "title": book["title"],
            "score": result["score"],
            "expected": book["expected"]
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


def calculate_pairwise_accuracy(results):

    correct = 0
    total = 0

    for i in range(len(results)):

        for j in range(i + 1, len(results)):

            predicted_order = (
                results[i]["score"] > results[j]["score"]
            )

            expected_order = (
                results[i]["expected"] < results[j]["expected"]
            )

            if predicted_order == expected_order:
                correct += 1

            total += 1

    return correct, total


if __name__ == "__main__":

    print("\n==============================")
    print("AFTERCREDITS EVALUATION")
    print("==============================")

    total_correct = 0
    total_comparisons = 0

    for movie_data in EVALUATION_DATA:

        print(f"\nMovie: {movie_data['movie']}")

        results = evaluate_movie(movie_data)

        print("\nPredicted Ranking:")

        for rank, result in enumerate(results, start=1):

            print(
                f"{rank}. {result['title']} "
                f"→ {result['score']:.3f} "
                f"(expected: {result['expected']})"
            )

        correct, comparisons = calculate_pairwise_accuracy(results)

        total_correct += correct
        total_comparisons += comparisons

        movie_accuracy = correct / comparisons

        print(
            f"\nMovie Pairwise Accuracy: "
            f"{movie_accuracy:.3f}"
        )

    overall_accuracy = (
        total_correct / total_comparisons
        if total_comparisons > 0
        else 0.0
    )

    print("\n==============================")
    print("OVERALL RESULTS")
    print("==============================")

    print(f"Movies evaluated: {len(EVALUATION_DATA)}")
    print(f"Correct comparisons: {total_correct}")
    print(f"Total comparisons: {total_comparisons}")
    print(f"Overall Pairwise Accuracy: {overall_accuracy:.3f}")