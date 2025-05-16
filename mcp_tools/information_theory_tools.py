import math
import numpy as np # For log2 and sum if needed, and to handle arrays
from typing import List

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py

def entropy(probabilities: List[float]) -> float:
    """
    Calculates the Shannon entropy of a discrete probability distribution.
    H(X) = - sum(p(x) * log2(p(x)) for x in X)

    Args:
        probabilities: A list of probabilities forming a distribution.
                       All probabilities must be non-negative and sum to 1 (approximately).
    Returns:
        The Shannon entropy in bits.
    Raises:
        ValueError: If probabilities are invalid (e.g., negative, don't sum to 1, or empty).
    """
    if not probabilities:
        raise ValueError("Probabilities list cannot be empty.")
    if not np.all(np.array(probabilities) >= 0):
        raise ValueError("All probabilities must be non-negative.")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        raise ValueError(f"Probabilities must sum to 1.0. Current sum: {sum(probabilities)}")

    # Filter out zero probabilities to avoid log2(0) issues.
    # p * log2(p) is 0 if p is 0.
    probs = [p for p in probabilities if p > 0]
    if not probs: # If all probabilities were zero (should be caught by sum check)
        return 0.0 
    
    return -np.sum(p * np.log2(p) for p in probs)

def kl_divergence(p_probs: List[float], q_probs: List[float]) -> float:
    """
    Calculates the Kullback-Leibler (KL) divergence between two discrete probability distributions P and Q.
    D_KL(P || Q) = sum(p(x) * log2(p(x) / q(x)) for x in X)
    It measures how much P diverges from Q. Not symmetric.

    Args:
        p_probs: A list of probabilities for distribution P.
                 All probabilities must be non-negative and sum to 1.
        q_probs: A list of probabilities for distribution Q.
                 Must have the same length as p_probs.
                 All probabilities must be non-negative and sum to 1.
                 Elements in q_probs should be > 0 where p_probs is > 0.
    Returns:
        The KL divergence in bits.
    Raises:
        ValueError: If probability lists are invalid (lengths, sums, negative values, or q(x)=0 where p(x)>0).
    """
    if not p_probs or not q_probs:
        raise ValueError("Probability lists P and Q cannot be empty.")
    if len(p_probs) != len(q_probs):
        raise ValueError("Probability distributions P and Q must have the same number of elements.")
    
    p = np.array(p_probs, dtype=float)
    q = np.array(q_probs, dtype=float)

    if not np.all(p >= 0) or not np.all(q >= 0):
        raise ValueError("All probabilities in P and Q must be non-negative.")
    if not math.isclose(np.sum(p), 1.0, abs_tol=1e-9):
        raise ValueError(f"Probabilities in P must sum to 1.0. Current sum: {np.sum(p)}")
    if not math.isclose(np.sum(q), 1.0, abs_tol=1e-9):
        raise ValueError(f"Probabilities in Q must sum to 1.0. Current sum: {np.sum(q)}")

    # Calculate KL divergence, taking care of p(x) > 0 and q(x) == 0 case
    # If p(x) > 0 and q(x) == 0, divergence is infinite.
    # If p(x) == 0, the term is 0.
    divergence = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            if q[i] == 0:
                # According to strict definition, if P(i) > 0 and Q(i) = 0, D_KL is infinite.
                # Some implementations might return a large number or handle it differently.
                # Raising an error is clear for a tool.
                raise ValueError(f"KL divergence is infinite: p({i}) > 0 and q({i}) = 0.")
            divergence += p[i] * (np.log2(p[i]) - np.log2(q[i]))
        # If p[i] == 0, the term p[i] * log(p[i]/q[i]) is 0, so no need to add.

    return divergence

def get_information_theory_tools() -> list:
    """Returns a list of all information theory tool functions."""
    return [
        entropy,
        kl_divergence
    ] 