import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import requests


sys.path.insert(0, str(Path(__file__).resolve().parent))

import open_library
from open_library import discover_books, merge_book_candidates
from lexical_similarity import title_similarity


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class OpenLibraryDiscoveryTests(unittest.TestCase):
    def test_discovers_and_enriches_thematic_candidates(self):
        work = {
            "key": "/works/OL123W",
            "title": "Rita Hayworth and Shawshank Redemption",
            "authors": [{"name": "Stephen King"}],
            "subject": ["prisoners", "prisons", "escapes", "friendship"],
        }
        request_urls = []

        def fake_get(url, params=None, headers=None, timeout=None):
            request_urls.append(url)
            if url.endswith("/search.json"):
                return FakeResponse({"docs": []})
            if "/subjects/" in url:
                return FakeResponse({"works": [work]})
            if url.endswith("/works/OL123W.json"):
                return FakeResponse({
                    "description": (
                        "A longtime prison inmate recounts his friendship "
                        "with a fellow prisoner and a daring escape."
                    )
                })
            raise AssertionError(f"Unexpected Open Library URL: {url}")

        with patch.object(open_library.requests, "get", side_effect=fake_get):
            books, expansion, errors = discover_books(
                "shashawank redemption",
                "A man is jailed and escapes prison, finding hope and friendship.",
                limit=10,
            )

        self.assertEqual(errors, [])
        self.assertIn("prison", expansion)
        self.assertEqual(len(books), 1)
        self.assertEqual(
            books[0]["title"],
            "Rita Hayworth and Shawshank Redemption",
        )
        self.assertEqual(books[0]["author"], "Stephen King")
        self.assertIn("daring escape", books[0]["description"])
        self.assertEqual(books[0]["source"], "Open Library")
        self.assertEqual(
            books[0]["source_url"],
            "https://openlibrary.org/works/OL123W",
        )
        self.assertTrue(any("/subjects/prisoners.json" in url for url in request_urls))

    def test_catalog_merge_keeps_local_record_and_adds_remote_evidence(self):
        local = [{
            "title": "A Book",
            "author": "A. Writer",
            "description": "Local description.",
        }]
        discovered = [{
            "title": "A Book",
            "author": "Unknown",
            "description": "Remote description.",
            "search_terms": ["prisoners"],
            "source": "Open Library",
            "source_url": "https://openlibrary.org/works/OL1W",
        }, {
            "title": "Another Book",
            "author": "B. Writer",
            "description": "Remote-only description.",
            "search_terms": ["escape"],
            "source": "Open Library",
            "source_url": "https://openlibrary.org/works/OL2W",
        }]

        merged = merge_book_candidates(local, discovered)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["description"], "Local description.")
        self.assertEqual(merged[0]["source"], "Local catalog and Open Library")
        self.assertEqual(merged[0]["search_terms"], ["prisoners"])
        self.assertEqual(merged[1]["source"], "Open Library")

    def test_network_failure_is_reported_and_local_fallback_can_continue(self):
        with patch.object(
            open_library.requests,
            "get",
            side_effect=requests.Timeout("service timed out"),
        ):
            books, _, errors = discover_books(
                "Unknown movie",
                "A story about friendship.",
            )

        self.assertEqual(books, [])
        self.assertTrue(errors)

    def test_misspelled_movie_title_can_match_the_source_novel(self):
        work = {
            "key": "/works/OL123W",
            "title": "Rita Hayworth and Shawshank Redemption",
            "authors": [{"name": "Stephen King"}],
            "subject": ["fiction", "prisoners", "prisons", "escapes"],
        }

        def fake_get(url, params=None, headers=None, timeout=None):
            if url.endswith("/search.json"):
                return FakeResponse({"docs": []})
            if "/subjects/" in url:
                return FakeResponse({"works": [work]})
            if url.endswith("/works/OL123W.json"):
                return FakeResponse({
                    "description": (
                        "A wrongly imprisoned man befriends another inmate "
                        "and plans an escape."
                    )
                })
            raise AssertionError(f"Unexpected Open Library URL: {url}")

        with patch.object(open_library.requests, "get", side_effect=fake_get):
            books, _, errors = discover_books(
                "shashawank redemption",
                "A man is jailed and escapes prison.",
                limit=8,
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(books), 1)
        self.assertTrue(
            title_similarity(
                "shashawank redemption",
                books[0]["title"],
            ) > 0.7
        )
        self.assertEqual(
            books[0]["title"],
            "Rita Hayworth and Shawshank Redemption",
        )

    def test_nonfiction_catalog_results_are_excluded(self):
        self.assertTrue(open_library._is_likely_nonfiction({
            "subject": ["Prisoners", "Personal memoirs", "Biography"],
        }))
        self.assertFalse(open_library._is_likely_nonfiction({
            "subject": ["Prisoners", "Autobiographical fiction"],
        }))


if __name__ == "__main__":
    unittest.main()
