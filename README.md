# AfterCredits

AfterCredits finds real books related to a movie from the movie title and a
plot description. It retrieves candidates from Open Library, adds the
project's local catalog, and ranks the candidates.

## Features

- Discover books from Open Library using the movie title, plot keywords,
  and topic searches selected by simple text rules.
- Combine online candidates with the curated local catalog.
- Rank candidates using plot similarity, catalog-subject overlap, and
  typo-tolerant title matching.
- Show the source, source link, and matching evidence for each recommendation.
- Fall back to the local catalog if Open Library is unavailable.

## Requirements

- Python 3.10 or newer
- Windows PowerShell commands are shown below; equivalent Python and pip
  commands work on other platforms.

## Setup

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use
`.venv\Scripts\python.exe` instead of activating `.venv`.

## Run the app

```powershell
python -m streamlit run src\app.py
```

For each movie, the app runs title, plot-keyword, and topic-subject searches
against Open Library. The topic searches are selected by rules in the code;
there is no generative model interpreting the plot. Retrieved records are
merged with `data/books.json` and ranked. Streamlit caches discovery results
for six hours.

Movie titles, plot descriptions, and derived search phrases are sent to Open
Library. If the service is unavailable, the app continues with local books.

## How recommendations work

AfterCredits does not ask a generative model to invent book titles. It
retrieves real book records from Open Library and ranks them. This lets a
movie or niche book be found even when it is not in the bundled catalog or
known to a language model.

The current ranker combines four signals: plot-description TF-IDF similarity
(65%), subject overlap (20%), title text similarity (5%), and typo-tolerant
title matching (10%). Rule-based topic expansion helps connect related terms
(for example, “jail” and “prison”). Very weak candidates are filtered out.
These are ranking signals, not probabilities or human-verified relevance
scores. Results still depend on Open Library's catalog coverage and metadata,
so check the source and matching evidence.

This is a retrieval and ranking project—not a generative AI recommender. It
does not make up book titles. The optional neural profiling code is not
required for the live retrieval flow.

## FastAPI

Run the REST API from the repository root:

```powershell
python -m uvicorn api:app --app-dir src --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

`GET /health` reports service health. `POST /recommendations` accepts JSON:

```json
{
  "movie_title": "Game of Thrones",
  "movie_description": "Family drama with an ancient war between rival clans.",
  "top_k": 5
}
```

The response includes the ranking method, match scores, matched terms, and a
source link for each result. Movie titles, descriptions, and derived topic
searches are sent to Open Library. If that service is unavailable, the API
can still rank the local catalog.

## Book catalog

The curated catalog is stored in `data/books.json` as a JSON array. Each record must
contain non-empty `title` and `description` strings. `author` is optional
and defaults to `"Unknown"`. Invalid records are logged and skipped.

Add or edit records in `data/books.json` to customize the local catalog.
Open Library supplies additional candidates at runtime. Generated profile
cache files are ignored by Git.

## Tests

Run the offline test suites with:

```powershell
python -m unittest discover -s src -p test_runtime_hardening.py -v
python -m unittest discover -s src -p test_open_library.py -v
python -m unittest discover -s src -p test_api.py -v
```

These tests mock Open Library calls and do not require internet access.
