import math
import statistics # Import the statistics module
from typing import List, Union
from scipy import stats # For confidence interval
import numpy as np # For sqrt, and potentially if other stats functions need it

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py
# when these functions are registered.

def average(numbers: List[float]) -> float:
    """Calculates the average of a list of numbers.
    Args:
        numbers: A list of numbers to calculate the average from.
    Returns:
        The average of the numbers.
    Raises:
        ValueError: If the list of numbers is empty.
    """
    if not numbers:
        raise ValueError("Cannot calculate the average of an empty list.")
    return sum(numbers) / len(numbers)

def median(numbers: List[float]) -> float:
    """Calculates the median (middle value) of a list of numbers.
    Args:
        numbers: A list of numbers.
    Returns:
        The median of the numbers.
    Raises:
        ValueError: If the list of numbers is empty.
    """
    if not numbers:
        raise ValueError("Cannot calculate the median of an empty list.")
    return statistics.median(numbers)

def mode(numbers: List[float]) -> Union[float, List[float]]:
    """Calculates the mode (most common value) of a list of numbers.
    Args:
        numbers: A list of numbers.
    Returns:
        The mode of the numbers.
    Raises:
        statistics.StatisticsError: If the list is empty or if there is no unique mode (e.g., all values are unique, or multiple values have the same highest frequency).
    """
    if not numbers:
        # statistics.mode raises StatisticsError for empty list, we can preempt with ValueError for consistency or let it propagate.
        # For now, let's be consistent with other functions.
        raise ValueError("Cannot calculate the mode of an empty list.")
    try:
        return statistics.mode(numbers)
    except statistics.StatisticsError: # Handle multiple modes
        return statistics.multimode(numbers)

def variance(numbers: List[float], population: bool = False) -> float:
    """Calculates the variance of a list of numbers.
    Args:
        numbers: A list of numbers.
        population: If True, calculates the population variance (divides by N).
                    If False (default), calculates the sample variance (divides by N-1).
    Returns:
        The variance of the numbers.
    Raises:
        ValueError: If the list of numbers is empty (or has 1 element for sample variance).
    """
    if not numbers:
        raise ValueError("Cannot calculate variance of an empty list.")
    if not population and len(numbers) < 2:
        raise ValueError("Sample variance requires at least two data points.")
    
    if population:
        return statistics.pvariance(numbers)
    else:
        return statistics.variance(numbers)

def standard_deviation(numbers: List[float], population: bool = False) -> float:
    """Calculates the standard deviation of a list of numbers.
    Args:
        numbers: A list of numbers.
        population: If True, calculates the population standard deviation.
                    If False (default), calculates the sample standard deviation.
    Returns:
        The standard deviation of the numbers.
    Raises:
        ValueError: If the list of numbers is empty (or has 1 element for sample stdev).
    """
    if not numbers:
        raise ValueError("Cannot calculate standard deviation of an empty list.")
    if not population and len(numbers) < 2:
        raise ValueError("Sample standard deviation requires at least two data points.")

    if population:
        return statistics.pstdev(numbers)
    else:
        return statistics.stdev(numbers)

def confidence_interval_mean(
    sample_data: List[float], 
    confidence_level: float = 0.95,
    use_t_distribution: bool = True 
) -> tuple[float, float]:
    """
    Calculates the confidence interval for the population mean based on sample data.

    Args:
        sample_data: A list of numerical data representing the sample.
        confidence_level: The desired confidence level (e.g., 0.95 for 95%).
                          Must be between 0 and 1 (exclusive).
        use_t_distribution: If True (default), uses the t-distribution (appropriate for
                              unknown population std dev or small samples). If False,
                              uses the normal distribution (z-score), assuming known
                              population std dev or large sample.

    Returns:
        A tuple (lower_bound, upper_bound) for the confidence interval.
    Raises:
        ValueError: If sample_data is too small (less than 2 items for t-dist),
                    or if confidence_level is not in (0,1).
    """
    n = len(sample_data)
    if n < 2 and use_t_distribution:
        raise ValueError("Sample size must be at least 2 to use t-distribution for confidence interval.")
    if n == 0:
         raise ValueError("Sample data cannot be empty.") # For norm dist or general case
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1 (exclusive).")

    sample_mean = average(sample_data)
    
    if use_t_distribution:
        # For t-distribution, standard error uses sample standard deviation.
        # ddof=1 for sample standard deviation in scipy.stats.sem
        # or calculate manually: sample_std_dev = standard_deviation(sample_data, population=False)
        # std_error = sample_std_dev / np.sqrt(n)
        std_error = stats.sem(sample_data) # Standard Error of the Mean, ddof=1 by default
        if std_error == 0: # Handle case where all sample data points are identical
            return (sample_mean, sample_mean)
        degrees_freedom = n - 1
        interval = stats.t.interval(confidence_level, degrees_freedom, loc=sample_mean, scale=std_error)
    else:
        # For normal distribution, we'd ideally use population standard deviation if known.
        # If not known and n is large, sample std dev is an approximation.
        # Let's assume for this branch, if population std dev isn't directly provided,
        # we proceed with sample std dev as an estimate for std_error calculation.
        # This is a simplification; in practice, t-distribution is safer if pop_std is unknown.
        if n < 2 : # Should be larger for Z, e.g. 30, but for tool usability let's keep it low
            print("Warning: Using Z-distribution for confidence interval with small sample size.")
        sample_std_dev = standard_deviation(sample_data, population=False)
        if sample_std_dev == 0: # All data points are the same
            return (sample_mean, sample_mean)
        std_error = sample_std_dev / np.sqrt(n)
        interval = stats.norm.interval(confidence_level, loc=sample_mean, scale=std_error)
        
    return interval # (lower_bound, upper_bound)

def get_stats_tools() -> list:
    """Returns a list of all statistical tool functions."""
    return [
        average,
        median,
        mode,
        variance,
        standard_deviation,
        confidence_interval_mean # Added new tool
    ] 