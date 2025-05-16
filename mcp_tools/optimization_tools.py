import math # Import math for use in eval() if func_str needs it
from scipy import optimize
import numpy # Often used with optimize, good to have in eval context
from typing import List, Optional, Callable, Dict # For type hinting

# Attempt to import numerical_derivative for use in gradient_descent_1d
# This creates a dependency between modules, which is acceptable for this structure.
# If calculus_tools.py is not found or numerical_derivative is not in it, an ImportError will occur at load time.
try:
    from calculus_tools import numerical_derivative
except ImportError:
    # Fallback if numerical_derivative cannot be imported.
    # Gradient descent will require explicit derivative_func_str in this case.
    numerical_derivative = None 

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py

def minimize_1d_function(func_str: str, bounds: tuple[float, float], method: str = 'bounded') -> dict:
    """
    Minimizes a 1D function provided as a string, within given bounds.
    Uses scipy.optimize.minimize_scalar.

    Args:
        func_str: A string representing the function f(x) (e.g., "(x - 2)**2").
                  The variable must be 'x'. Functions from 'math' and 'numpy' can be used
                  (e.g., "math.sin(x) + numpy.cos(x*2)").
                  It's crucial that this string is safe to evaluate.
        bounds: A tuple (min, max) specifying the interval to search for the minimum.
        method: The minimization method to use (e.g., 'bounded', 'brent', 'golden').
                'bounded' is good if you have strict bounds.

    Returns:
        A dictionary containing the results from minimize_scalar, typically including:
        'x': The x value at which the minimum is found.
        'fun': The value of the function at the minimum.
        'success': Boolean indicating if the optimization was successful.
        'message': A message describing the exit condition.
        'nfev': Number of function evaluations.
    Raises:
        ValueError: If bounds are invalid or method is not recognized by scipy.
        Exception: If there's an error evaluating func_str.
    """
    if not (isinstance(bounds, tuple) and len(bounds) == 2 and bounds[0] < bounds[1]):
        raise ValueError("Bounds must be a tuple of two floats (min, max) where min < max.")

    eval_globals = {"math": math, "numpy": numpy, "__builtins__": {}}
    try:
        func = lambda x_val: eval(func_str, eval_globals, {"x": x_val})
        
        test_x = bounds[0]
        if test_x == 0 and bounds[1] > 0 : test_x = 0
        elif bounds[0] < 0 < bounds[1]: test_x = 0
        else: test_x = (bounds[0] + bounds[1]) / 2.0
        func(test_x)
    except Exception as e:
        raise Exception(f"Error evaluating function string '{func_str}' with test value x={test_x}: {e}")

    try:
        result = optimize.minimize_scalar(func, bounds=bounds, method=method)
        # Convert numpy types to standard Python types for JSON serialization if needed
        # For example, result.x can be numpy.float64
        return {
            'x': float(result.x),
            'fun': float(result.fun),
            'success': bool(result.success),
            'message': str(result.message),
            'nfev': int(result.nfev) # Number of function evaluations
            # 'status': result.status # another potentially useful field
            # 'nit': result.nit # number of iterations
        }
    except Exception as e:
        # Catch errors from the optimization process itself
        raise Exception(f"Error during optimization with method '{method}': {e}")

def gradient_descent_1d(
    func_str: str, 
    initial_x: float, 
    learning_rate: float, 
    iterations: int, 
    derivative_func_str: Optional[str] = None,
    h_derivative: float = 1e-5 # Only used if derivative_func_str is None
) -> Dict[str, any]:
    """
    Performs 1D gradient descent to find a local minimum of a function.

    Args:
        func_str: String representation of the function f(x) (e.g., "x**2 - 4*x + 4").
                  Uses 'x' as the variable. 'math' and 'numpy' can be used.
        initial_x: The starting point for x.
        learning_rate: The step size for each iteration.
        iterations: The number of iterations to perform.
        derivative_func_str: Optional string representation of the derivative f'(x).
                             If None, numerical_derivative (if available) will be used.
        h_derivative: Step size for numerical_derivative if used.

    Returns:
        A dictionary containing:
        'x_min': The x value found after N iterations (potential local minimum).
        'f_min': The function value at x_min.
        'iterations_run': Number of iterations performed.
        'history_x': List of x values at each iteration.
        'history_f': List of function values at each iteration.
    Raises:
        ValueError: If iterations or learning_rate are not positive.
        Exception: If func_str or derivative_func_str (if provided) cannot be evaluated,
                   or if numerical_derivative is needed but not available/fails.
    """
    if iterations <= 0:
        raise ValueError("Number of iterations must be positive.")
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")

    eval_globals = {"math": math, "numpy": numpy, "__builtins__": {}}
    
    try:
        # Define the function f(x) from func_str
        f = lambda val: eval(func_str, eval_globals, {"x": val})
        # Test evaluation
        f(initial_x)
    except Exception as e:
        raise Exception(f"Error evaluating main function string '{func_str}': {e}")

    # Define the derivative function df/dx
    df_dx: Callable[[float], float]
    if derivative_func_str:
        try:
            df_dx = lambda val: eval(derivative_func_str, eval_globals, {"x": val})
            # Test evaluation
            df_dx(initial_x)
        except Exception as e:
            raise Exception(f"Error evaluating derivative function string '{derivative_func_str}': {e}")
    elif numerical_derivative is not None:
        print("Using numerical_derivative for gradient descent.") # For logging/debugging
        # numerical_derivative itself handles eval of func_str
        df_dx = lambda val: numerical_derivative(func_str, val, h_derivative)
        # Test evaluation (numerical_derivative has its own internal tests, but an initial call is good)
        try:
            df_dx(initial_x)
        except Exception as e:
            raise Exception(f"Error during initial call to numerical_derivative: {e}")
    else:
        raise RuntimeError(
            "Derivative function string not provided and numerical_derivative tool is not available."
        )

    x_current = initial_x
    history_x = [x_current]
    history_f = [f(x_current)]

    for i in range(iterations):
        grad = df_dx(x_current)
        x_next = x_current - learning_rate * grad
        x_current = x_next
        history_x.append(x_current)
        history_f.append(f(x_current))

    return {
        "x_min": x_current,
        "f_min": f(x_current),
        "iterations_run": iterations,
        "history_x": history_x,
        "history_f": history_f
    }

def get_optimization_tools() -> list:
    """Returns a list of all optimization tool functions."""
    return [
        minimize_1d_function,
        gradient_descent_1d # Added new tool
    ] 