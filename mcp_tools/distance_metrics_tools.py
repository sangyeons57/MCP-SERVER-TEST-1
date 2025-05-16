import numpy as np
from typing import List

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py

def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculates the Euclidean distance between two vectors.
    sqrt(sum((x_i - y_i)^2 for i in range(n)))

    Args:
        vec1: The first vector (list of floats).
        vec2: The second vector (list of floats).
              Must have the same dimension as vec1.
    Returns:
        The Euclidean distance.
    Raises:
        ValueError: If vectors are empty or have different dimensions.
    """
    if not vec1 or not vec2:
        raise ValueError("Input vectors cannot be empty.")
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimensions.")
    
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    return float(np.linalg.norm(v1 - v2)) # np.linalg.norm computes Euclidean (L2) norm by default

def manhattan_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculates the Manhattan (L1) distance between two vectors.
    sum(|x_i - y_i| for i in range(n))

    Args:
        vec1: The first vector (list of floats).
        vec2: The second vector (list of floats).
              Must have the same dimension as vec1.
    Returns:
        The Manhattan distance.
    Raises:
        ValueError: If vectors are empty or have different dimensions.
    """
    if not vec1 or not vec2:
        raise ValueError("Input vectors cannot be empty.")
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimensions.")

    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    return float(np.sum(np.abs(v1 - v2)))

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculates the cosine similarity between two non-zero vectors.
    (A . B) / (||A|| * ||B||)

    Args:
        vec1: The first vector (list of floats).
        vec2: The second vector (list of floats).
              Must have the same dimension as vec1.
    Returns:
        The cosine similarity (between -1 and 1).
    Raises:
        ValueError: If vectors are empty, have different dimensions, or if either vector is all zeros.
    """
    if not vec1 or not vec2:
        raise ValueError("Input vectors cannot be empty.")
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimensions.")

    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Cosine similarity is not defined for zero vectors.")

    dot_product = np.dot(v1, v2)
    return float(dot_product / (norm_v1 * norm_v2))

def get_distance_metrics_tools() -> list:
    """Returns a list of all distance metrics tool functions."""
    return [
        euclidean_distance,
        manhattan_distance,
        cosine_similarity
    ] 