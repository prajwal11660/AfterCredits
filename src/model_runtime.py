from functools import lru_cache


class ModelUnavailableError(RuntimeError):
    """Raised when a required ML model cannot load or run."""


@lru_cache(maxsize=1)
def _zero_shot_classifier():
    try:
        from transformers import pipeline

        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    except Exception as error:
        raise ModelUnavailableError(
            "The zero-shot classification model could not be loaded."
        ) from error


@lru_cache(maxsize=1)
def _emotion_classifier():
    try:
        from transformers import pipeline

        return pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
        )
    except Exception as error:
        raise ModelUnavailableError(
            "The emotion classification model could not be loaded."
        ) from error


@lru_cache(maxsize=1)
def _embedding_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as error:
        raise ModelUnavailableError(
            "The semantic embedding model could not be loaded."
        ) from error


def classify_zero_shot(text, candidate_labels):
    try:
        return _zero_shot_classifier()(
            text,
            candidate_labels=candidate_labels,
            multi_label=True,
        )
    except ModelUnavailableError:
        raise
    except Exception as error:
        raise ModelUnavailableError(
            "The zero-shot classification model failed during inference."
        ) from error


def classify_emotions(text):
    try:
        return _emotion_classifier()(text, truncation=True)
    except ModelUnavailableError:
        raise
    except Exception as error:
        raise ModelUnavailableError(
            "The emotion classification model failed during inference."
        ) from error


def create_semantic_embedding(text):
    try:
        return _embedding_model().encode(text).tolist()
    except ModelUnavailableError:
        raise
    except Exception as error:
        raise ModelUnavailableError(
            "The semantic embedding model failed during inference."
        ) from error
