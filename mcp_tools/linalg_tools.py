import numpy as np
from typing import List, Dict, Tuple, Any

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

def determinant(matrix: List[List[float]]) -> float:
    """
    Calculates the determinant of a square matrix.
    Args:
        matrix: A square matrix (list of lists of floats).
    Returns:
        The determinant of the matrix.
    Raises:
        ValueError: If the matrix is not square.
    """
    A = np.array(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square to calculate its determinant.")
    return np.linalg.det(A)

def inverse_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """
    Calculates the inverse of a square matrix.
    Args:
        matrix: A square, invertible matrix (list of lists of floats).
    Returns:
        The inverse of the matrix.
    Raises:
        ValueError: If the matrix is not square or is singular (not invertible).
    """
    A = np.array(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square to calculate its inverse.")
    try:
        inv_A = np.linalg.inv(A)
        return inv_A.tolist()
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular and cannot be inverted.")

def eigen_decomposition(matrix_a: List[List[float]]) -> Dict[str, List[Any]]:
    """
    Computes the eigenvalues and eigenvectors of a square matrix A.
    Eigenvectors are returned as columns.
    Args:
        matrix_a: A square matrix (list of lists of floats).
    Returns:
        A dictionary with two keys:
        'eigenvalues': A list of eigenvalues (can be complex, returned as strings for JSON compatibility if complex).
        'eigenvectors_as_columns': A list of lists, where each inner list is an eigenvector corresponding
                                   to an eigenvalue at the same index in the 'eigenvalues' list.
                                   Eigenvectors are represented as columns.
    Raises:
        ValueError: If matrix_a is not square or other numpy.linalg.LinAlgError occurs.
    """
    A = np.array(matrix_a, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square (n x n) for eigenvalue decomposition.")

    try:
        eigenvalues, eigenvectors_matrix = np.linalg.eig(A)
        
        # Convert eigenvalues to string if complex for broader JSON compatibility, else float
        processed_eigenvalues = []
        for v in eigenvalues:
            if isinstance(v, complex):
                processed_eigenvalues.append(str(v)) # MCP/JSON might handle complex better as strings
            else:
                processed_eigenvalues.append(float(v))

        # np.linalg.eig returns eigenvectors as columns of a matrix.
        # To return as a list of column vectors:
        eigenvectors_as_columns_list = [col.tolist() for col in eigenvectors_matrix.T]

        return {
            "eigenvalues": processed_eigenvalues,
            "eigenvectors_as_columns": eigenvectors_as_columns_list
        }
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Eigenvalue decomposition failed: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during eigenvalue decomposition: {e}")

def lu_decomposition(matrix: List[List[float]]) -> Dict[str, List[List[float]]]:
    """
    Performs LU decomposition of a square matrix A such that A = PLU.
    Note: scipy.linalg.lu returns P as a permutation matrix, L as lower triangular, U as upper triangular.
          For simplicity, we'll return P, L, U directly.
    Args:
        matrix: A square matrix (list of lists of floats).
    Returns:
        A dictionary with keys 'P' (Permutation matrix), 'L' (Lower triangular matrix),
        and 'U' (Upper triangular matrix).
    Raises:
        ValueError: If the matrix is not square or decomposition fails.
    """
    from scipy.linalg import lu # Import scipy.linalg only here
    A = np.array(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square for LU decomposition.")
    try:
        P, L, U = lu(A)
        return {
            "P": P.tolist(),
            "L": L.tolist(),
            "U": U.tolist()
        }
    except Exception as e: # Catching general exception as scipy.linalg.lu specific errors are less common
        raise ValueError(f"LU decomposition failed: {e}")

def qr_decomposition(matrix: List[List[float]]) -> Dict[str, List[List[float]]]:
    """
    Performs QR decomposition of a matrix A such that A = QR.
    Q is an orthogonal matrix, R is an upper triangular matrix.
    Args:
        matrix: A matrix (list of lists of floats).
    Returns:
        A dictionary with keys 'Q' (Orthogonal matrix) and 'R' (Upper triangular matrix).
    Raises:
        ValueError: If decomposition fails.
    """
    A = np.array(matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError("Input must be a 2D matrix for QR decomposition.")
    try:
        Q, R = np.linalg.qr(A)
        return {
            "Q": Q.tolist(),
            "R": R.tolist()
        }
    except np.linalg.LinAlgError as e:
        raise ValueError(f"QR decomposition failed: {e}")

def svd_decomposition(matrix: List[List[float]]) -> Dict[str, List[Any]]:
    """
    Performs Singular Value Decomposition (SVD) of a matrix A such that A = U S Vh.
    U: Unitary matrix having left singular vectors as columns.
    S: The singular values, sorted in non-increasing order. This is a 1-D array.
    Vh: Unitary matrix having right singular vectors as rows. (V.H, hermitian transpose of V)
    Args:
        matrix: A matrix (list of lists of floats).
    Returns:
        A dictionary with keys 'U' (Unitary matrix), 'S' (Singular values as a list),
        and 'Vh' (Unitary matrix, V hermitian).
    Raises:
        ValueError: If decomposition fails.
    """
    A = np.array(matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError("Input must be a 2D matrix for SVD.")
    try:
        U, s, Vh = np.linalg.svd(A)
        return {
            "U": U.tolist(),
            "S": s.tolist(), # s is 1-D array
            "Vh": Vh.tolist()
        }
    except np.linalg.LinAlgError as e:
        raise ValueError(f"SVD failed: {e}")

def get_linalg_tools() -> list:
    """Returns a list of all linear algebra tool functions."""
    return [
        matrix_add, matrix_subtract, matrix_scalar_multiply, matrix_multiply, transpose,
        solve_linear_system,
        determinant,
        inverse_matrix,
        eigen_decomposition,
        lu_decomposition,
        qr_decomposition,
        svd_decomposition
    ] 