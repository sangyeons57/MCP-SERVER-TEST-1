import math
import statistics # Import the statistics module
from typing import List, Union, Tuple, Dict
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

def pearson_correlation(data1: List[float], data2: List[float]) -> Dict[str, float]:
    """
    Calculates the Pearson correlation coefficient and the p-value for testing non-correlation.
    Args:
        data1: First list of numbers.
        data2: Second list of numbers. Must have the same length as data1.
    Returns:
        A dictionary with 'correlation_coefficient' and 'p_value'.
    Raises:
        ValueError: If lists are not of the same length or have less than 2 elements.
    """
    if len(data1) != len(data2):
        raise ValueError("Input lists must have the same length for correlation.")
    if len(data1) < 2:
        raise ValueError("At least two data points are required for correlation.")
    
    corr_coeff, p_value = stats.pearsonr(data1, data2)
    if np.isnan(corr_coeff): # Handle cases like constant input where correlation is undefined
        raise ValueError("Pearson correlation coefficient is undefined (NaN), possibly due to constant input data.")
    return {"correlation_coefficient": float(corr_coeff), "p_value": float(p_value)}

def spearman_correlation(data1: List[float], data2: List[float]) -> Dict[str, float]:
    """
    Calculates the Spearman rank-order correlation coefficient and the p-value.
    Args:
        data1: First list of numbers.
        data2: Second list of numbers. Must have the same length as data1.
    Returns:
        A dictionary with 'correlation_coefficient' and 'p_value'.
    Raises:
        ValueError: If lists are not of the same length or have less than 2 elements.
    """
    if len(data1) != len(data2):
        raise ValueError("Input lists must have the same length for Spearman correlation.")
    if len(data1) < 2:
        raise ValueError("At least two data points are required for Spearman correlation.")
        
    corr_coeff, p_value = stats.spearmanr(data1, data2)
    if np.isnan(corr_coeff):
        raise ValueError("Spearman correlation coefficient is undefined (NaN), possibly due to constant input data or ties.")
    return {"correlation_coefficient": float(corr_coeff), "p_value": float(p_value)}

def covariance(data1: List[float], data2: List[float], population: bool = False) -> float:
    """
    Calculates the covariance between two lists of numbers.
    Args:
        data1: First list of numbers.
        data2: Second list of numbers. Must have the same length as data1.
        population: If True, calculates population covariance (N).
                    If False (default), calculates sample covariance (N-1).
    Returns:
        The covariance value.
    Raises:
        ValueError: If lists are not of the same length or have less than 2 elements for sample covariance.
    """
    if len(data1) != len(data2):
        raise ValueError("Input lists must have the same length for covariance.")
    n = len(data1)
    if n == 0:
        raise ValueError("Cannot calculate covariance of empty lists.")
    if not population and n < 2:
        raise ValueError("Sample covariance requires at least two data points.")

    # numpy.cov returns a covariance matrix. We need the covariance between data1 and data2.
    # For two variables, cov_matrix[0, 1] (or cov_matrix[1,0]) is the covariance.
    # ddof=0 for population covariance (N), ddof=1 for sample covariance (N-1).
    cov_matrix = np.cov(data1, data2, ddof=0 if population else 1)
    return float(cov_matrix[0, 1])

def simple_linear_regression(x_values: List[float], y_values: List[float]) -> Dict[str, float]:
    """
    Performs a simple linear regression (y = slope * x + intercept).
    Args:
        x_values: List of independent variable values.
        y_values: List of dependent variable values. Must have the same length as x_values.
    Returns:
        A dictionary containing 'slope', 'intercept', 'r_value' (Pearson correlation),
        'p_value' (for the slope), and 'stderr' (standard error of the estimate).
    Raises:
        ValueError: If lists are not of the same length or have less than 2 elements.
    """
    if len(x_values) != len(y_values):
        raise ValueError("Input lists must have the same length for linear regression.")
    if len(x_values) < 2:
        raise ValueError("At least two data points are required for linear regression.")
        
    slope, intercept, r_value, p_value, stderr = stats.linregress(x_values, y_values)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "p_value": float(p_value),
        "stderr_of_estimate": float(stderr)
    }

def t_test_1samp(sample_data: List[float], popmean: float, alternative: str = 'two-sided') -> Dict[str, float]:
    """
    Performs a one-sample t-test to check if the sample mean is different from a known population mean.
    Args:
        sample_data: List of sample observations.
        popmean: The expected mean of the population.
        alternative: Defines the alternative hypothesis.
                     'two-sided': mean of underlying distribution is not equal to popmean (default)
                     'less': mean of underlying distribution is less than popmean
                     'greater': mean of underlying distribution is greater than popmean
    Returns:
        A dictionary with 'statistic' (t-statistic) and 'p_value'.
    Raises:
        ValueError: If sample_data has less than 2 elements or alternative is invalid.
    """
    if len(sample_data) < 2: # scipy.stats.ttest_1samp typically requires at least 2.
        raise ValueError("One-sample t-test requires at least two data points in the sample.")
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("Invalid alternative hypothesis. Choose from 'two-sided', 'less', 'greater'.")

    statistic, p_value = stats.ttest_1samp(sample_data, popmean, alternative=alternative)
    return {"statistic": float(statistic), "p_value": float(p_value)}

def t_test_ind(sample1_data: List[float], sample2_data: List[float], equal_var: bool = True, alternative: str = 'two-sided') -> Dict[str, float]:
    """
    Performs an independent two-sample t-test to check if the means of two independent samples are different.
    Args:
        sample1_data: List of observations for the first sample.
        sample2_data: List of observations for the second sample.
        equal_var: If True (default), perform a standard independent 2 sample test
                   that assumes equal population variances. If False, perform Welch's t-test,
                   which does not assume equal population variance.
        alternative: Defines the alternative hypothesis.
                     'two-sided': means of the distributions are unequal.
                     'less': mean of the distribution underlying sample1_data is less than the mean of the distribution underlying sample2_data.
                     'greater': mean of the distribution underlying sample1_data is greater than the mean of the distribution underlying sample2_data.
    Returns:
        A dictionary with 'statistic' (t-statistic) and 'p_value'.
    Raises:
        ValueError: If samples have less than 2 elements or alternative is invalid.
    """
    if len(sample1_data) < 2 or len(sample2_data) < 2:
         raise ValueError("Independent two-sample t-test requires at least two data points in each sample.")
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("Invalid alternative hypothesis. Choose from 'two-sided', 'less', 'greater'.")

    statistic, p_value = stats.ttest_ind(sample1_data, sample2_data, equal_var=equal_var, alternative=alternative)
    return {"statistic": float(statistic), "p_value": float(p_value)}

def chi_squared_test(observed_freq: List[int], expected_freq: List[int] = None) -> Dict[str, float]:
    """
    Performs a Chi-squared goodness of fit test.
    Tests the null hypothesis that the categorical data has the given frequencies.
    Args:
        observed_freq: List of observed frequencies in each category.
        expected_freq: List of expected frequencies in each category.
                       If None, the test assumes uniform distribution (all categories equally likely).
                       Must have the same length as observed_freq if provided.
    Returns:
        A dictionary with 'statistic' (Chi-squared statistic) and 'p_value'.
    Raises:
        ValueError: If lengths of observed and expected frequencies don't match (if expected_freq is provided),
                    or if any frequency is negative, or sum of frequencies is zero.
    """
    if any(f < 0 for f in observed_freq):
        raise ValueError("Observed frequencies cannot be negative.")
    if expected_freq is not None and any(f <= 0 for f in expected_freq): # Expected frequencies should be > 0
        raise ValueError("Expected frequencies must be positive.")
    if expected_freq is not None and len(observed_freq) != len(expected_freq):
        raise ValueError("Observed and expected frequencies must have the same length.")
    if sum(observed_freq) == 0:
        raise ValueError("Total observed frequency cannot be zero.")
    if expected_freq is not None and sum(expected_freq) == 0:
         raise ValueError("Total expected frequency cannot be zero.")
    # Note: scipy.stats.chisquare normalizes expected frequencies if they don't sum to observed.
    # For simplicity, we might want to enforce sum(observed) == sum(expected) if expected is provided.
    # However, scipy handles it, so we can allow it.
    
    statistic, p_value = stats.chisquare(f_obs=observed_freq, f_exp=expected_freq)
    return {"statistic": float(statistic), "p_value": float(p_value)}

def get_stats_tools() -> list:
    """Returns a list of all statistical tool functions."""
    return [
        average,
        median,
        mode,
        variance,
        standard_deviation,
        confidence_interval_mean,
        pearson_correlation,
        spearman_correlation,
        covariance,
        simple_linear_regression,
        t_test_1samp,
        t_test_ind,
        chi_squared_test
    ] 