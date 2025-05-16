import numpy as np
from typing import List

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py
# when these functions are registered.

# --- Linear Algebra Operations ---
def matrix_add(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
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
    a = np.array(matrix_a)
    b = np.array(matrix_b)
    if a.shape != b.shape:
        raise ValueError("Matrices must have the same dimensions for addition.")
    return (a + b).tolist()

def matrix_subtract(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
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
    a = np.array(matrix_a)
    b = np.array(matrix_b)
    if a.shape != b.shape:
        raise ValueError("Matrices must have the same dimensions for subtraction.")
    return (a - b).tolist()

def matrix_scalar_multiply(matrix: List[List[float]], scalar: float) -> List[List[float]]:
    """Multiplies a matrix by a scalar.
    Example: @CalculatorMCP multiply matrix [[1,2],[3,4]] by scalar 2

    Args:
        matrix: The matrix to multiply (list of lists of floats).
        scalar: The scalar value.

    Returns:
        The resulting matrix.
    """
    a = np.array(matrix)
    return (a * scalar).tolist()

def matrix_multiply(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
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
    a = np.array(matrix_a)
    b = np.array(matrix_b)
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Number of columns in the first matrix ({a.shape[1]}) "
            f"must equal the number of rows in the second matrix ({b.shape[0]}) for multiplication."
        )
    return np.dot(a, b).tolist()

def transpose(matrix: List[List[float]]) -> List[List[float]]:
    """Transposes a matrix.
    Example: @CalculatorMCP transpose matrix [[1,2,3],[4,5,6]]

    Args:
        matrix: The matrix to transpose (list of lists of floats).

    Returns:
        The transposed matrix.
    """
    return np.array(matrix).T.tolist()

def solve_linear_system(matrix_a: List[List[float]], vector_b: List[float]) -> List[float]:
    """
    Solves a system of linear equations of the form Ax = b for x.
    Args:
        matrix_a: A square matrix A of coefficients (list of lists of floats).
        vector_b: A vector b of dependent variables (list of floats).
                  Must have the same number of elements as rows/columns in matrix_A.
    Returns:
        The solution vector x (list of floats).
    Raises:
        ValueError: If matrix_A is not square, or if dimensions are incompatible,
                    or if the system is singular (no unique solution).
    """
    A = np.array(matrix_a, dtype=float)
    b_vec = np.array(vector_b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square (n x n).")
    if A.shape[0] != b_vec.shape[0]:
        raise ValueError("Number of rows in matrix A must match the number of elements in vector b.")

    try:
        solution = np.linalg.solve(A, b_vec)
        return solution.tolist()
    except np.linalg.LinAlgError as e:
        # This error is raised for singular matrices (no unique solution) among other things.
        raise ValueError(f"Could not solve linear system: {e}. The matrix might be singular.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while solving linear system: {e}")

def eigen_decomposition(matrix_a: List[List[float]]) -> dict:
    """
    Computes the eigenvalues and eigenvectors of a square matrix A.
    Args:
        matrix_a: A square matrix (list of lists of floats).
    Returns:
        A dictionary with two keys:
        'eigenvalues': A list of eigenvalues (can be complex).
        'eigenvectors': A list of lists representing the eigenvectors (columns).
                       Each eigenvector corresponds to an eigenvalue at the same index.
    Raises:
        ValueError: If matrix_a is not square or other numpy.linalg.LinAlgError occurs.
    """
    A = np.array(matrix_a, dtype=float) # Use float for broader compatibility, eig can handle it

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square (n x n) for eigenvalue decomposition.")

    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # Eigenvalues can be complex, eigenvectors can also be complex.
        # np.linalg.eig returns eigenvectors as columns of a matrix.
        return {
            "eigenvalues": [complex(v) for v in eigenvalues], # Ensure complex type, convert if real
            "eigenvectors": eigenvectors.T.tolist() # Transpose to get row-vectors, then list of lists
        }
    except np.linalg.LinAlgError as e:
        # This can occur for various reasons, e.g., matrix not converging for the algorithm.
        raise ValueError(f"Eigenvalue decomposition failed: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during eigenvalue decomposition: {e}")

def get_linalg_tools() -> list:
    """Returns a list of all linear algebra tool functions."""
    return [
        matrix_add, matrix_subtract, matrix_scalar_multiply, matrix_multiply, transpose,
        solve_linear_system,
        eigen_decomposition
    ] 