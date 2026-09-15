import numpy as np


def cosine_similarity(embedding_a, embedding_b):
    """
    Calculate cosine similarity between two embeddings.
    """

    vector_a = np.array(embedding_a)
    vector_b = np.array(embedding_b)

    dot_product = np.dot(vector_a, vector_b)

    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)