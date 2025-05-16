import math
import numpy # For consistent eval context with other tools like optimization
from typing import List # typing.List is for older Pythons, can be list for newer. Keep for compatibility.

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py

def numerical_derivative(func_str: str, x_val: float, h: float = 1e-5) -> float:
    """
    Calculates the numerical derivative of a given function string at a point x_val
    using the central difference formula: (f(x+h) - f(x-h)) / (2*h).
    The function string should be a valid Python expression using 'x' as the variable.
    Example: func_str = "x**2", "math.sin(x)", "numpy.exp(x)".

    Args:
        func_str: A string representing the function f(x) (e.g., "x**2 + 2*x").
                  It's crucial that this string is safe to evaluate.
        x_val: The point at which to calculate the derivative.
        h: The step size for the finite difference method.
    Returns:
        The approximate derivative of f(x) at x_val.
    Raises:
        SyntaxError: If func_str is not a valid Python expression.
        NameError: If func_str uses undefined variables (other than 'x', 'math', 'numpy').
        TypeError: If func_str does not evaluate to a numerical type.
        Exception: For other evaluation errors.
    """
    eval_globals = {"math": math, "numpy": numpy, "__builtins__": {}}
    try:
        f_x_plus_h = eval(func_str, eval_globals, {"x": x_val + h})
        f_x_minus_h = eval(func_str, eval_globals, {"x": x_val - h})

        if not isinstance(f_x_plus_h, (int, float)) or \
           not isinstance(f_x_minus_h, (int, float)):
            raise TypeError("The function string did not evaluate to a number for f(x+h) or f(x-h).")

        return (f_x_plus_h - f_x_minus_h) / (2 * h)
    except (SyntaxError, NameError, TypeError) as e:
        # Re-raise specific, somewhat expected errors for clarity
        raise type(e)(f"Error in function string '{func_str}': {e}")
    except Exception as e:
        # Catch any other errors during eval and raise a general one
        raise Exception(f"Unexpected error evaluating function string '{func_str}': {e}")

def numerical_integral(func_str: str, a: float, b: float, n: int = 1000) -> float:
    """
    Calculates the numerical integral (area under the curve) of a given function string
    from a to b using the trapezoidal rule.
    The function string should be a valid Python expression using 'x' as the variable.

    Args:
        func_str: A string representing the function f(x) (e.g., "x**2").
                  Can use 'math.' or 'numpy.' prefixes for functions (e.g., "math.sin(x)", "numpy.exp(x)").
        a: The lower limit of integration.
        b: The upper limit of integration.
        n: The number of trapezoids (subintervals) to use for approximation.
           More trapezoids generally lead to a more accurate result.
    Returns:
        The approximate definite integral of the function from a to b.
    Raises:
        ValueError: If n is not positive.
        SyntaxError, NameError, TypeError: If func_str has issues or evaluates incorrectly.
        Exception: For other evaluation errors.
    """
    if n <= 0:
        raise ValueError("Number of subintervals (n) must be positive.")
    if a == b:
        return 0.0

    h_step = (b - a) / n
    integral_sum = 0.0
    eval_globals = {"math": math, "numpy": numpy, "__builtins__": {}} # Consistent eval context

    try:
        # Evaluate f(a) and f(b) once
        f_a = eval(func_str, eval_globals, {"x": a})
        f_b = eval(func_str, eval_globals, {"x": b})

        if not isinstance(f_a, (int, float)) or not isinstance(f_b, (int, float)):
            raise TypeError("The function string did not evaluate to a number at the integration bounds.")

        integral_sum += (f_a + f_b) / 2.0

        for i in range(1, n):
            x_i = a + i * h_step
            f_x_i = eval(func_str, eval_globals, {"x": x_i})
            if not isinstance(f_x_i, (int, float)):
                raise TypeError(f"The function string did not evaluate to a number at x={x_i}.")
            integral_sum += f_x_i
            
        return integral_sum * h_step
    except (SyntaxError, NameError, TypeError) as e:
        raise type(e)(f"Error in function string '{func_str}': {e}")
    except Exception as e:
        raise Exception(f"Unexpected error evaluating function string '{func_str}' during integration: {e}")

def get_calculus_tools() -> List[callable]: # Using List[Callable] for better type hinting
    """Returns a list of all calculus tool functions."""
    return [
        numerical_derivative,
        numerical_integral
    ]