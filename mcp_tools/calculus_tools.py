import math
import numpy as np # numpy로 변경하고 일관성 유지
from typing import List, Callable, Dict, Any # Dict, Any, Callable 추가

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
    eval_globals = {"math": math, "numpy": np, "__builtins__": {}}
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
    eval_globals = {"math": math, "numpy": np, "__builtins__": {}} # Consistent eval context

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

def numerical_gradient(func_str: str, point_vars: List[float], h: float = 1e-5) -> List[float]:
    """
    Calculates the numerical gradient of a multivariable function string at a given point.
    The function string should use variables like x[0], x[1], ...
    e.g., func_str = "x[0]**2 + math.sin(x[1])"
    point_vars = [1.0, 0.5] (values for x[0], x[1] respectively)

    Args:
        func_str: A string representing the multivariable function f(x[0], x[1], ...).
        point_vars: A list of float values for each variable at which to calculate the gradient.
        h: The step size for the finite difference method.
    Returns:
        A list of floats representing the gradient vector.
    Raises:
        SyntaxError, NameError, TypeError, IndexError: If func_str has issues or point_vars is incorrect.
        Exception: For other evaluation errors.
    """
    eval_globals = {"math": math, "numpy": np, "__builtins__": {}}
    gradient = []

    for i in range(len(point_vars)):
        point_plus_h = list(point_vars) # Create a mutable copy
        point_minus_h = list(point_vars)

        point_plus_h[i] += h
        point_minus_h[i] -= h
        
        try:
            f_plus_h = eval(func_str, eval_globals, {"x": point_plus_h})
            f_minus_h = eval(func_str, eval_globals, {"x": point_minus_h})

            if not isinstance(f_plus_h, (int, float)) or \
               not isinstance(f_minus_h, (int, float)):
                raise TypeError(f"Function string did not evaluate to a number for variable x[{i}].")

            partial_derivative = (f_plus_h - f_minus_h) / (2 * h)
            gradient.append(partial_derivative)
        except IndexError:
            raise IndexError(f"Function string may be trying to access x[{i}] but point_vars has only {len(point_vars)} elements.")
        except (SyntaxError, NameError, TypeError) as e:
            raise type(e)(f"Error evaluating function string '{func_str}' for variable x[{i}]: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error evaluating function string '{func_str}' for x[{i}]: {e}")
            
    return gradient

def numerical_jacobian(func_vector_str: List[str], point_vars: List[float], h: float = 1e-5) -> List[List[float]]:
    """
    Calculates the numerical Jacobian matrix of a vector of multivariable functions.
    Each function string in func_vector_str should use variables like x[0], x[1], ...
    e.g., func_vector_str = ["x[0]**2 + x[1]", "x[0]*math.sin(x[1])"]
    point_vars = [1.0, 0.5]

    Args:
        func_vector_str: A list of strings, each representing a component function
                         f_i(x[0], x[1], ...).
        point_vars: A list of float values for each variable.
        h: The step size for the finite difference.
    Returns:
        The Jacobian matrix as a list of lists of floats.
        Rows correspond to functions, columns correspond to variables.
    Raises:
        Similar errors as numerical_gradient for each component function.
    """
    jacobian_matrix = []
    for func_str_i in func_vector_str:
        # Calculate the gradient for each function f_i, which forms a row of the Jacobian
        gradient_row = numerical_gradient(func_str_i, point_vars, h)
        jacobian_matrix.append(gradient_row)
    return jacobian_matrix

def numerical_hessian(func_str: str, point_vars: List[float], h: float = 1e-7) -> List[List[float]]:
    """
    Calculates the numerical Hessian matrix of a multivariable scalar function.
    The function string should use variables like x[0], x[1], ...
    Uses central differences for second derivatives.
    H_ij = (f(x + h_i*e_i + h_j*e_j) - f(x + h_i*e_i - h_j*e_j) - 
            f(x - h_i*e_i + h_j*e_j) + f(x - h_i*e_i - h_j*e_j)) / (4 * h_i * h_j) for i != j
    H_ii = (f(x + h_i*e_i) - 2*f(x) + f(x - h_i*e_i)) / (h_i^2)

    Args:
        func_str: A string representing the scalar multivariable function f(x[0], x[1], ...).
        point_vars: A list of float values for each variable.
        h: A small step size. Using a smaller h for Hessian due to second derivatives.
    Returns:
        The Hessian matrix as a list of lists of floats.
    Raises:
        Similar errors as numerical_gradient.
    """
    n_vars = len(point_vars)
    hessian_matrix = [[0.0] * n_vars for _ in range(n_vars)]
    eval_globals = {"math": math, "numpy": np, "__builtins__": {}}

    try:
        f_x = eval(func_str, eval_globals, {"x": point_vars})
        if not isinstance(f_x, (int, float)):
            raise TypeError("Function string did not evaluate to a number at the central point.")

        for i in range(n_vars):
            # Diagonal elements (second partial derivative wrt x_i)
            point_plus_h_i = list(point_vars)
            point_minus_h_i = list(point_vars)
            point_plus_h_i[i] += h
            point_minus_h_i[i] -= h

            f_plus_h_i = eval(func_str, eval_globals, {"x": point_plus_h_i})
            f_minus_h_i = eval(func_str, eval_globals, {"x": point_minus_h_i})
            
            if not isinstance(f_plus_h_i, (int, float)) or not isinstance(f_minus_h_i, (int, float)):
                 raise TypeError(f"Function string did not evaluate to a number for H[{i}][{i}] calculation.")

            hessian_matrix[i][i] = (f_plus_h_i - 2 * f_x + f_minus_h_i) / (h ** 2)

            # Off-diagonal elements (mixed partial derivatives)
            for j in range(i + 1, n_vars):
                point_pp = list(point_vars) # x + h_i*e_i + h_j*e_j
                point_pm = list(point_vars) # x + h_i*e_i - h_j*e_j
                point_mp = list(point_vars) # x - h_i*e_i + h_j*e_j
                point_mm = list(point_vars) # x - h_i*e_i - h_j*e_j

                point_pp[i] += h; point_pp[j] += h
                point_pm[i] += h; point_pm[j] -= h
                point_mp[i] -= h; point_mp[j] += h
                point_mm[i] -= h; point_mm[j] -= h

                f_pp = eval(func_str, eval_globals, {"x": point_pp})
                f_pm = eval(func_str, eval_globals, {"x": point_pm})
                f_mp = eval(func_str, eval_globals, {"x": point_mp})
                f_mm = eval(func_str, eval_globals, {"x": point_mm})

                if not all(isinstance(val, (int, float)) for val in [f_pp, f_pm, f_mp, f_mm]):
                    raise TypeError(f"Function string did not evaluate to a number for H[{i}][{j}] calculation.")
                
                mixed_partial = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
                hessian_matrix[i][j] = mixed_partial
                hessian_matrix[j][i] = mixed_partial # Hessian is symmetric

        return hessian_matrix
    except IndexError:
        raise IndexError(f"Function string may be trying to access x[i] but point_vars has only {len(point_vars)} elements.")
    except (SyntaxError, NameError, TypeError) as e:
        raise type(e)(f"Error evaluating function string '{func_str}': {e}")
    except Exception as e:
        raise Exception(f"Unexpected error evaluating function string '{func_str}': {e}")

def get_calculus_tools() -> List[Callable]:
    """Returns a list of all calculus tool functions."""
    return [
        numerical_derivative,
        numerical_integral,
        numerical_gradient,
        numerical_jacobian,
        numerical_hessian
    ]