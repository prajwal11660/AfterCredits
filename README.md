# AfterCredits

AfterCredits recommends books that share a movie's themes, characters, and
emotional experience. It includes a Streamlit app and a local book catalog.

## Features

- Rank books against a movie title and description.
- Explain recommendations with shared story signals.
- Fall back to local TF-IDF matching if the neural models are unavailable.
- Validate book records and profile caches before using them.

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

The app uses the local catalog in `data/books.json` and TF-IDF matching when
there is no compatible profile cache. It does not download books or build
profiles during a recommendation request. Results show the matching method
and, for the fallback, the query terms that matched each book.

## Optional neural profiles

Neural profiles use Hugging Face Transformers and Sentence Transformers.
Install these packages only if you want to generate profiles:

```powershell
pip install transformers sentence-transformers torch
```

Model files may download the first time they are used, and profile
generation can take several minutes. To build a cache for the local catalog:

```powershell
python src\build_book_profiles.py
```

Caches are versioned JSON artifacts. Missing, stale, or invalid caches are
not used by the app; it returns to the local TF-IDF path instead.

## Book catalog

The catalog is stored in `data/books.json` as a JSON array. Each record must
contain non-empty `title` and `description` strings. `author` is optional
and defaults to `"Unknown"`. Invalid records are logged and skipped.

Add or edit records in `data/books.json` to customize the local catalog.
Generated profile cache files are ignored by Git and can be rebuilt as
needed.

## Tests

Run the network-free runtime and recommendation tests with:

```powershell
python -m unittest discover -s src -p test_runtime_hardening.py -v
```
