# mcp_tools/number_theory_tools.py
"""
Number theory tools for the MCP Calculator.
"""
import math
from typing import List, Dict, Any

def is_prime(n: int) -> bool:
    """
    Checks if a number n is prime.
    Args:
        n: An integer.
    Returns:
        True if n is prime, False otherwise.
        Numbers less than 2 are not prime.
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer.")
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Check from 5 onwards with a step of 6 (optimizing for numbers not divisible by 2 or 3)
    # Primes are of the form 6k +/- 1
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def prime_factorization(n: int) -> List[int]:
    """
    Returns the prime factorization of an integer n as a list of prime factors.
    Args:
        n: A positive integer.
    Returns:
        A list of prime factors of n, sorted in ascending order.
        Returns an empty list for n=1. Returns [n] if n is prime.
    Raises:
        ValueError: If n is not positive.
        TypeError: If n is not an integer.
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer.")
    if n <= 0:
        raise ValueError("Input must be a positive integer for prime factorization.")
    if n == 1:
        return []

    factors = []
    
    # Handle factor 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2
        
    # Handle factor 3
    while n % 3 == 0:
        factors.append(3)
        n //= 3
        
    # Check for factors from 5 onwards (6k +/- 1)
    i = 5
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        while n % (i + 2) == 0:
            factors.append(i + 2)
            n //= (i + 2)
        i += 6
        
    # If n is still greater than 1, it must be prime
    if n > 1:
        factors.append(n)
        
    return sorted(factors) # Ensure sorted order, though the process should naturally produce it.

def get_number_theory_tools():
    """Returns a list of number theory tool functions."""
    return [is_prime, prime_factorization] 