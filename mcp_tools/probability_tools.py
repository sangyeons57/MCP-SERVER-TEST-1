import math
from scipy import stats
from typing import Union

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py
# when these functions are registered.

# --- Normal Distribution ---

def normal_pdf(x: float, mean: float, std_dev: float) -> float:
    """
    Calculates the Probability Density Function (PDF) for a normal distribution
    at a given point x.
    Args:
        x: The point at which to evaluate the PDF.
        mean: The mean (mu) of the normal distribution.
        std_dev: The standard deviation (sigma) of the normal distribution. Must be > 0.
    Returns:
        The PDF value at x.
    Raises:
        ValueError: If std_dev is not positive.
    """
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive.")
    return stats.norm.pdf(x, loc=mean, scale=std_dev)

def normal_cdf(x: float, mean: float, std_dev: float) -> float:
    """
    Calculates the Cumulative Distribution Function (CDF) for a normal distribution
    at a given point x.
    Args:
        x: The point at which to evaluate the CDF.
        mean: The mean (mu) of the normal distribution.
        std_dev: The standard deviation (sigma) of the normal distribution. Must be > 0.
    Returns:
        The CDF value at x (P(X <= x)).
    Raises:
        ValueError: If std_dev is not positive.
    """
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive.")
    return stats.norm.cdf(x, loc=mean, scale=std_dev)

def normal_ppf(q: float, mean: float, std_dev: float) -> float:
    """
    Calculates the Percent Point Function (PPF) or inverse CDF for a normal distribution
    for a given probability q.
    Args:
        q: The probability (0 < q < 1).
        mean: The mean (mu) of the normal distribution.
        std_dev: The standard deviation (sigma) of the normal distribution. Must be > 0.
    Returns:
        The value x such that P(X <= x) = q.
    Raises:
        ValueError: If std_dev is not positive or q is not in (0, 1).
    """
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive.")
    if not (0 < q < 1):
        raise ValueError("Probability q must be between 0 and 1 (exclusive).")
    return stats.norm.ppf(q, loc=mean, scale=std_dev)

# --- Binomial Distribution ---

def binomial_pmf(k: int, n: int, p: float) -> float:
    """
    Calculates the Probability Mass Function (PMF) for a binomial distribution.
    Args:
        k: The number of successes (integer, 0 <= k <= n).
        n: The number of trials (integer, n >= 0).
        p: The probability of success on a single trial (0 <= p <= 1).
    Returns:
        The PMF value P(X = k).
    Raises:
        ValueError: If parameters are out of valid range.
    """
    if not (isinstance(k, int) and isinstance(n, int) and k >= 0 and n >= 0 and k <= n):
        raise ValueError("k and n must be non-negative integers with k <= n.")
    if not (0 <= p <= 1):
        raise ValueError("Probability p must be between 0 and 1 (inclusive).")
    return stats.binom.pmf(k, n, p)

def binomial_cdf(k: int, n: int, p: float) -> float:
    """
    Calculates the Cumulative Distribution Function (CDF) for a binomial distribution.
    Args:
        k: The number of successes (integer, 0 <= k <= n).
        n: The number of trials (integer, n >= 0).
        p: The probability of success on a single trial (0 <= p <= 1).
    Returns:
        The CDF value P(X <= k).
    Raises:
        ValueError: If parameters are out of valid range.
    """
    if not (isinstance(k, int) and isinstance(n, int) and k >= 0 and n >= 0 and k <= n): # k can be < 0 for cdf, but k<=n still holds
         if k < 0: k = -1 # scipy handles k < 0 for cdf by returning 0
         elif not (isinstance(k, int) and isinstance(n, int) and n >= 0 and k <= n):
            raise ValueError("k and n must be integers with n >= 0 and k <= n.")
    if not (0 <= p <= 1):
        raise ValueError("Probability p must be between 0 and 1 (inclusive).")
    return stats.binom.cdf(k, n, p)

def binomial_ppf(q: float, n: int, p: float) -> int:
    """
    Calculates the Percent Point Function (PPF) or inverse CDF for a binomial distribution.
    Args:
        q: The probability (0 < q < 1).
        n: The number of trials (integer, n >= 0).
        p: The probability of success on a single trial (0 <= p <= 1).
    Returns:
        The smallest integer k such that P(X <= k) >= q.
    Raises:
        ValueError: If parameters are out of valid range.
    """
    if not (isinstance(n, int) and n >= 0):
        raise ValueError("Number of trials n must be a non-negative integer.")
    if not (0 <= p <= 1):
        raise ValueError("Probability p must be between 0 and 1 (inclusive).")
    if not (0 < q < 1):
        raise ValueError("Probability q must be between 0 and 1 (exclusive).")
    # PPF for discrete distributions returns float, so we convert to int
    return int(stats.binom.ppf(q, n, p))

# --- Poisson Distribution ---

def poisson_pmf(k: int, mu: float) -> float:
    """
    Calculates the Probability Mass Function (PMF) for a Poisson distribution.
    Args:
        k: The number of events (integer, k >= 0).
        mu: The average rate of events (lambda, mu > 0).
    Returns:
        The PMF value P(X = k).
    Raises:
        ValueError: If parameters are out of valid range.
    """
    if not (isinstance(k, int) and k >= 0):
        raise ValueError("Number of events k must be a non-negative integer.")
    if mu <= 0:
        raise ValueError("Average rate mu (lambda) must be positive.")
    return stats.poisson.pmf(k, mu)

def poisson_cdf(k: int, mu: float) -> float:
    """
    Calculates the Cumulative Distribution Function (CDF) for a Poisson distribution.
    Args:
        k: The number of events (integer, k >= 0).
        mu: The average rate of events (lambda, mu > 0).
    Returns:
        The CDF value P(X <= k).
    Raises:
        ValueError: If parameters are out of valid range.
    """
    if not (isinstance(k, int) and k >= 0): # k can be < 0 for cdf
        if k < 0 : k = -1 # scipy handles k < 0 for cdf by returning 0
        elif not (isinstance(k,int)):
             raise ValueError("Number of events k must be an integer.")
    if mu <= 0:
        raise ValueError("Average rate mu (lambda) must be positive.")
    return stats.poisson.cdf(k, mu)

def poisson_ppf(q: float, mu: float) -> int:
    """
    Calculates the Percent Point Function (PPF) or inverse CDF for a Poisson distribution.
    Args:
        q: The probability (0 < q < 1).
        mu: The average rate of events (lambda, mu > 0).
    Returns:
        The smallest integer k such that P(X <= k) >= q.
    Raises:
        ValueError: If parameters are out of valid range.
    """
    if mu <= 0:
        raise ValueError("Average rate mu (lambda) must be positive.")
    if not (0 < q < 1):
        raise ValueError("Probability q must be between 0 and 1 (exclusive).")
    # PPF for discrete distributions returns float, so we convert to int
    return int(stats.poisson.ppf(q, mu))

def get_probability_tools() -> list:
    """Returns a list of all probability tool functions."""
    return [
        normal_pdf, normal_cdf, normal_ppf,
        binomial_pmf, binomial_cdf, binomial_ppf,
        poisson_pmf, poisson_cdf, poisson_ppf
    ] 