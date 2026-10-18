import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))

import book_profiles
import model_runtime
import recommender
from book_data import load_books
from book_profiles import ProfileCacheError
from book_validation import validate_books
from model_runtime import ModelUnavailableError
from story_profile import StoryProfile


def profile(title, embedding=None):
    return StoryProfile(
        title=title,
        themes={"friendship": 0.8},
        emotional_experiences={"belonging": 0.7},
        character_traits={"introspective": 0.6},
        embedding=embedding or [1.0, 0.5],
    )


class BookValidationTests(unittest.TestCase):
    def test_normalizes_valid_records_and_reports_invalid_records(self):
        books, issues = validate_books([
            {
                "title": "  Good Book  ",
                "author": None,
                "description": "  A useful description. ",
            },
            {"title": "Missing description"},
            "not a record",
        ])

        self.assertEqual(len(books), 1)
        self.assertEqual(books[0]["title"], "Good Book")
        self.assertEqual(books[0]["author"], "Unknown")
        self.assertEqual(books[0]["description"], "A useful description.")
        self.assertEqual(len(issues), 2)

    def test_rejects_non_list_input(self):
        with self.assertRaises(TypeError):
            validate_books({})


class RecommendationTests(unittest.TestCase):
    books = [
        {
            "title": "Friendship Story",
            "author": "A. Author",
            "description": (
                "A lonely teenager finds friendship, belonging, "
                "and emotional connection."
            ),
        },
        {
            "title": "Detective Story",
            "description": "A detective investigates a futuristic crime.",
        },
    ]

    def test_model_failure_uses_deterministic_lexical_ranking(self):
        with patch.object(
            recommender,
            "create_story_profile",
            side_effect=ModelUnavailableError("models are offline"),
        ):
            result = recommender.recommend_books(
                "A movie",
                "A lonely teenager discovers friendship and belonging.",
                self.books,
                top_k=2,
            )

        self.assertEqual(result["method"], "lexical_tfidf")
        self.assertEqual(result["fallback_reason"], "models are offline")
        self.assertEqual(
            result["recommendations"][0]["title"],
            "Friendship Story",
        )
        self.assertEqual(
            result["recommendations"][0]["method"],
            "lexical_tfidf",
        )
        self.assertEqual(
            result["recommendations"][0]["author"],
            "A. Author",
        )

    def test_model_path_uses_valid_cached_book_profiles(self):
        cached_books = [
            {**self.books[0], "profile": profile("Friendship Story")},
            {**self.books[1], "profile": profile("Detective Story")},
        ]
        with patch.object(
            recommender,
            "create_story_profile",
            return_value=profile("Movie"),
        ) as create_profile:
            result = recommender.recommend_books(
                "A movie",
                "A movie description.",
                cached_books,
            )

        self.assertEqual(result["method"], "model")
        self.assertEqual(create_profile.call_count, 1)
        self.assertEqual(len(result["recommendations"]), 2)

    def test_preexisting_model_failure_skips_model_retry(self):
        with patch.object(
            recommender,
            "create_story_profile",
            side_effect=AssertionError("model path should not be retried"),
        ):
            result = recommender.recommend_books(
                "A movie",
                "A lonely teenager finds friendship.",
                self.books,
                model_unavailable_reason="profile models are offline",
            )

        self.assertEqual(result["method"], "lexical_tfidf")
        self.assertEqual(
            result["fallback_reason"],
            "profile models are offline",
        )

    def test_game_of_thrones_diagnostic_returns_relevant_local_books(self):
        result = recommender.recommend_books(
            "Game of Thrones",
            "Family drama with ancient war between clans and fighting for throne.",
            load_books(),
            top_k=5,
            model_unavailable_reason="diagnostic fallback",
        )

        titles = [item["title"] for item in result["recommendations"]]
        self.assertEqual(titles[0], "A Game of Thrones")
        self.assertIn("Shogun", titles)
        self.assertIn("Dune", titles)
        self.assertIn("The Lord of the Rings", titles)
        self.assertNotIn(
            "The Curious Incident of the Dog in the Night-Time",
            titles,
        )

    def test_lexical_fallback_omits_zero_or_weak_matches(self):
        result = recommender.recommend_books(
            "Game of Thrones",
            "Ancient war between clans fighting for the throne.",
            [{
                "title": "Unrelated book",
                "description": "A quiet story about a garden.",
            }],
            model_unavailable_reason="diagnostic fallback",
        )
        self.assertEqual(result["recommendations"], [])

    def test_invalid_books_are_skipped_and_empty_input_fails_clearly(self):
        with patch.object(
            recommender,
            "create_story_profile",
            side_effect=ModelUnavailableError("models are offline"),
        ):
            result = recommender.recommend_books(
                "A movie",
                "A lonely teenager finds friendship, belonging, and emotional connection.",
                [self.books[0], {"title": "No description"}],
            )

        self.assertEqual(len(result["recommendations"]), 1)
        with self.assertRaisesRegex(ValueError, "valid book"):
            recommender.recommend_books(
                "A movie",
                "A movie description.",
                [{"title": "No description"}],
            )

    def test_empty_story_and_invalid_top_k_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Movie description"):
            recommender.recommend_books("Movie", "  ", self.books)
        with self.assertRaisesRegex(ValueError, "top_k"):
            recommender.recommend_books(
                "Movie",
                "Description",
                self.books,
                top_k=0,
            )


class ProfileArtifactTests(unittest.TestCase):
    books = [{
        "title": "A Book",
        "description": "A valid description.",
    }]

    def test_versioned_artifact_round_trips_and_checks_source(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "profiles.pkl"
            with patch.object(
                book_profiles,
                "create_story_profile",
                side_effect=lambda title, text: profile(title),
            ):
                book_profiles._build_profiles(self.books, artifact_path)

            loaded = book_profiles._load_profiles(
                artifact_path,
                source_books=self.books,
            )
            self.assertEqual(loaded[0]["title"], "A Book")

            changed_source = [{
                "title": "A Book",
                "description": "A changed description.",
            }]
            with self.assertRaisesRegex(ProfileCacheError, "does not match"):
                book_profiles._load_profiles(
                    artifact_path,
                    source_books=changed_source,
                )

    def test_legacy_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "legacy.json"
            artifact_path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ProfileCacheError, "legacy"):
                book_profiles._load_profiles(artifact_path)

    def test_inference_errors_are_reported_as_model_unavailable(self):
        with patch.object(
            model_runtime,
            "_zero_shot_classifier",
            side_effect=RuntimeError("backend unavailable"),
        ):
            with self.assertRaises(ModelUnavailableError):
                model_runtime.classify_zero_shot("story", ["friendship"])

    def test_missing_profile_cache_uses_local_books_without_model_build(self):
        expected_books = [{
            "title": "Local book",
            "author": "A. Author",
            "description": "A local description.",
        }]
        with (
            patch.object(
                book_profiles,
                "load_api_book_profiles",
                side_effect=ProfileCacheError("cache missing"),
            ),
            patch.object(
                book_profiles,
                "load_books",
                return_value=expected_books,
            ),
            patch.object(book_profiles, "build_api_book_profiles") as build,
        ):
            books, reason = book_profiles.load_profiles_or_local_books()

        self.assertEqual(books, expected_books)
        self.assertIn("TF-IDF", reason)
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
