import numpy as np

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py
# when these functions are registered.

# --- Linear Algebra Operations ---
def matrix_add(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> list[list[float]]:
    """Adds two matrices.
    Example: @CalculatorMCP add matrix [[1,2],[3,4]] to matrix [[5,6],[7,8]]

    Args:
        matrix_a: The first matrix (list of lists of floats).
        matrix_b: The second matrix (list of lists of floats).

    Returns:
        The resulting matrix from the addition.

    Raises:
        ValueError: If matrices have incompatible shapes.
    """
    try:
        np_a = np.array(matrix_a, dtype=float)
        np_b = np.array(matrix_b, dtype=float)
        result_matrix = np_a + np_b
        return result_matrix.tolist()
    except ValueError as e:
        raise ValueError(f"Matrix addition error: {e}. Ensure matrices have compatible shapes.")

def matrix_subtract(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> list[list[float]]:
    """Subtracts the second matrix from the first.
    Example: @CalculatorMCP subtract matrix [[1,2],[3,4]] from matrix [[5,6],[7,8]] 
    (Note: example shows subtracting b from a, current prompt text is a from b)

    Args:
        matrix_a: The matrix to subtract from (list of lists of floats).
        matrix_b: The matrix to subtract (list of lists of floats).

    Returns:
        The resulting matrix from the subtraction.

    Raises:
        ValueError: If matrices have incompatible shapes.
    """
    try:
        np_a = np.array(matrix_a, dtype=float)
        np_b = np.array(matrix_b, dtype=float)
        result_matrix = np_a - np_b
        return result_matrix.tolist()
    except ValueError as e:
        raise ValueError(f"Matrix subtraction error: {e}. Ensure matrices have compatible shapes.")

def matrix_scalar_multiply(matrix: list[list[float]], scalar: float) -> list[list[float]]:
    """Multiplies a matrix by a scalar.
    Example: @CalculatorMCP multiply matrix [[1,2],[3,4]] by scalar 2

    Args:
        matrix: The matrix to multiply (list of lists of floats).
        scalar: The scalar value.

    Returns:
        The resulting matrix.
    """
    np_matrix = np.array(matrix, dtype=float)
    result_matrix = np_matrix * scalar
    return result_matrix.tolist()

def matrix_multiply(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> list[list[float]]:
    """Multiplies two matrices (matrix product).
    Example: @CalculatorMCP multiply matrix [[1,2],[3,4]] by matrix [[5,6],[7,8]]

    Args:
        matrix_a: The first matrix (list of lists of floats).
        matrix_b: The second matrix (list of lists of floats).

    Returns:
        The resulting matrix product.

    Raises:
        ValueError: If matrices have incompatible shapes for multiplication.
    """
    try:
        np_a = np.array(matrix_a, dtype=float)
        np_b = np.array(matrix_b, dtype=float)
        result_matrix = np.matmul(np_a, np_b)
        return result_matrix.tolist()
    except ValueError as e:
        raise ValueError(f"Matrix multiplication error: {e}. Ensure columns in first matrix match rows in second.")

def transpose(matrix: list[list[float]]) -> list[list[float]]:
    """Transposes a matrix.
    Example: @CalculatorMCP transpose matrix [[1,2,3],[4,5,6]]

    Args:
        matrix: The matrix to transpose (list of lists of floats).

    Returns:
        The transposed matrix.
    """
    np_matrix = np.array(matrix, dtype=float)
    result_matrix = np_matrix.T # .T is the shorthand for transpose in NumPy
    return result_matrix.tolist()

def get_linalg_tools() -> list:
    """Returns a list of all linear algebra tool functions."""
    return [
        matrix_add, matrix_subtract, matrix_scalar_multiply, 
        matrix_multiply, transpose
    ] 