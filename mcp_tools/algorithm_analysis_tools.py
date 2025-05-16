import math
import numpy # For potential use in expression_str via eval
from typing import List, Dict, Any, Callable

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py

def big_o_evaluator(expression_str: str, variable_values: Dict[str, float]) -> float:
    """
    Evaluates a mathematical expression string with given variable values.
    Useful for estimating computational cost for a given input size based on a Big-O like expression.

    Args:
        expression_str: The mathematical expression string (e.g., "2*n**2 + 10*n + 5", "c * k * log(k)").
                        Uses variables defined in variable_values. 
                        Can use 'math.' or 'numpy.' prefixed functions.
        variable_values: A dictionary where keys are variable names (str) and 
                         values are their numerical values (float or int).
                         E.g., {"n": 1000.0, "c": 2.0, "k": 50.0}

    Returns:
        The numerical result of the evaluated expression.
    Raises:
        SyntaxError: If expression_str is invalid.
        NameError: If expression_str uses undefined variables or functions 
                   (not in variable_values, math, numpy).
        Exception: For other evaluation errors.
    """
    eval_globals = {"math": math, "numpy": numpy, "__builtins__": {}}
    # The locals for eval will be the provided variable_values
    try:
        # Ensure all variable values are numeric before eval
        for var_name, value in variable_values.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Value for variable '{var_name}' must be numeric (int or float).")
        
        result = eval(expression_str, eval_globals, variable_values)
        if not isinstance(result, (int, float)):
            raise TypeError(f"Expression '{expression_str}' did not evaluate to a number.")
        return float(result)
    except (SyntaxError, NameError, TypeError, ValueError) as e:
        raise type(e)(f"Error evaluating expression '{expression_str}' with variables {variable_values}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error evaluating '{expression_str}': {e}")

def master_theorem_solver(a: float, b: float, k: float, i: float = 0) -> str:
    """
    Solves recurrence relations of the form T(n) = a*T(n/b) + f(n)
    where f(n) is Theta(n^k * (log n)^i), using the Master Theorem.
    Assumes a >= 1, b > 1, k >= 0, i >= 0.

    Args:
        a: The number of subproblems.
        b: The factor by which the subproblem size is reduced.
        k: The exponent of n in f(n).
        i: The exponent of log n in f(n) (defaults to 0 if f(n) is just Theta(n^k)).

    Returns:
        A string describing the time complexity T(n) in Big-Theta notation.
    Raises:
        ValueError: If input parameters are outside their valid ranges for the Master Theorem.
    """
    if a < 1:
        raise ValueError("Parameter 'a' (number of subproblems) must be >= 1.")
    if b <= 1:
        raise ValueError("Parameter 'b' (subproblem size reduction factor) must be > 1.")
    if k < 0:
        raise ValueError("Parameter 'k' (exponent of n in f(n)) must be >= 0.")
    if i < 0:
        raise ValueError("Parameter 'i' (exponent of log n in f(n)) must be >= 0.")

    # Calculate log_b(a)
    log_b_a = math.log(a, b)

    # Case 1: log_b(a) > k
    if math.isclose(log_b_a, k) and log_b_a > k or log_b_a > k:
        return f"T(n) = Theta(n^{log_b_a:.2f})"

    # Case 2: log_b(a) == k
    if math.isclose(log_b_a, k):
        if i > -1:
            return f"T(n) = Theta(n^{k:.2f} * (log n)^{i+1:.0f})"
        elif math.isclose(i, -1):
             return f"T(n) = Theta(n^{k:.2f} * log log n)" 
        else: # i < -1
             return f"T(n) = Theta(n^{k:.2f})"

    # Case 3: log_b(a) < k
    if log_b_a < k:
        if i >= 0: # Regularity condition for f(n) = Omega(n^(log_b_a + eps)) is assumed to hold.
            if math.isclose(i,0):
                return f"T(n) = Theta(n^{k:.2f})"
            else:
                return f"T(n) = Theta(n^{k:.2f} * (log n)^{i:.0f})"
        else: # i < 0, f(n) is Theta(n^k / (log n)^|i|)
            # This case requires more careful handling of regularity condition, often simplified.
            # For simplicity, if f(n) grows polynomially faster, result is Theta(f(n)).
            if math.isclose(i,0):
                return f"T(n) = Theta(n^{k:.2f})"
            else:
                return f"T(n) = Theta(n^{k:.2f} * (log n)^{i:.0f})" # Note: i is negative here in formula, kept as is for input.
    
    return "Master theorem case not clearly met with provided parameters or requires regularity condition check not implemented."

def get_algorithm_analysis_tools() -> List[Callable]:
    """Returns a list of all algorithm analysis tool functions."""
    return [
        big_o_evaluator,
        master_theorem_solver
    ] 