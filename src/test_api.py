import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parent))

import api


class RecommendationApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.catalog = [
            {
                "title": "A Game of Thrones",
                "author": "George R. R. Martin",
                "description": (
                    "Rival noble families struggle for the throne through "
                    "war, alliances, and betrayal."
                ),
            },
            {
                "title": "Shogun",
                "author": "James Clavell",
                "description": (
                    "Rival clans contend in civil war and court intrigue."
                ),
            },
        ]

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_recommendations_endpoint_returns_validated_local_results(self):
        with patch.object(
            api,
            "get_book_catalog",
            return_value=(self.catalog, "models unavailable"),
        ), patch.object(
            api,
            "discover_books",
            return_value=([], "", []),
        ):
            response = self.client.post(
                "/recommendations",
                json={
                    "movie_title": "Game of Thrones",
                    "movie_description": (
                        "Family drama with ancient war between clans "
                        "fighting for the throne."
                    ),
                    "top_k": 2,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["method"], "lexical_tfidf")
        self.assertNotIn("C:\\", payload["fallback_reason"])
        self.assertEqual(len(payload["recommendations"]), 2)
        self.assertEqual(
            payload["recommendations"][0]["title"],
            "A Game of Thrones",
        )
        self.assertIn(
            "throne",
            payload["recommendations"][0]["matched_terms"],
        )
        self.assertEqual(
            payload["recommendations"][0]["method"],
            "lexical_tfidf",
        )

    def test_invalid_requests_are_rejected(self):
        response = self.client.post(
            "/recommendations",
            json={
                "movie_title": "   ",
                "movie_description": "A story",
                "top_k": 2,
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_discovered_open_library_books_are_returned_with_source(self):
        discovered = [{
            "title": "Rita Hayworth and Shawshank Redemption",
            "author": "Stephen King",
            "description": "A prison inmate tells of a friendship and escape.",
            "search_terms": ["prisoners", "escape"],
            "source": "Open Library",
            "source_url": "https://openlibrary.org/works/OL123W",
        }]
        with patch.object(
            api,
            "get_book_catalog",
            return_value=(self.catalog, "profiles unavailable"),
        ), patch.object(
            api,
            "discover_books",
            return_value=(discovered, "prison inmate escape", []),
        ):
            response = self.client.post(
                "/recommendations",
                json={
                    "movie_title": "Shawshank Redemption",
                    "movie_description": (
                        "A man is jailed and escapes prison."
                    ),
                    "top_k": 3,
                },
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(
            result["recommendations"][0]["title"],
            "Rita Hayworth and Shawshank Redemption",
        )
        self.assertEqual(
            result["recommendations"][0]["source"],
            "Open Library",
        )
        self.assertEqual(
            result["recommendations"][0]["source_url"],
            "https://openlibrary.org/works/OL123W",
        )

        response = self.client.post(
            "/recommendations",
            json={
                "movie_title": "A Movie",
                "movie_description": "A story",
                "top_k": 21,
            },
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
