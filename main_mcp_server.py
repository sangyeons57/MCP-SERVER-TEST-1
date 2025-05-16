from mcp.server.fastmcp import FastMCP

# Import tool-providing functions from other modules
from mcp_tools.arithmetic_tools import get_arithmetic_tools
from mcp_tools.linalg_tools import get_linalg_tools
from mcp_tools.stats_tools import get_stats_tools
from mcp_tools.calculus_tools import get_calculus_tools
from mcp_tools.probability_tools import get_probability_tools
from mcp_tools.optimization_tools import get_optimization_tools
from mcp_tools.information_theory_tools import get_information_theory_tools
from mcp_tools.distance_metrics_tools import get_distance_metrics_tools
from mcp_tools.algorithm_analysis_tools import get_algorithm_analysis_tools
from mcp_tools.ml_metrics_tools import get_ml_metrics_tools
from mcp_tools.graph_tools import get_graph_tools
from mcp_tools.number_theory_tools import get_number_theory_tools
from mcp_tools.conversion_tools import get_conversion_tools
from mcp_tools.crypto_basics_tools import get_crypto_basics_tools

# Create the main MCP server instance
mcp = FastMCP("CalculatorMCP")

# Register tools from various modules
for tool_getter in [
    get_arithmetic_tools,
    get_linalg_tools,
    get_stats_tools,
    get_calculus_tools,
    get_probability_tools,
    get_optimization_tools,
    get_information_theory_tools,
    get_distance_metrics_tools,
    get_algorithm_analysis_tools,
    get_ml_metrics_tools,
    get_graph_tools,
    get_number_theory_tools,
    get_conversion_tools,
    get_crypto_basics_tools
]:
    for tool_func in tool_getter():
        mcp.tool()(tool_func)

@mcp.resource("calculator://info")
def calculator_info():
    """Provides information about the available calculator tools."""
    info_string = """
Available Calculator Tools:
===========================

Arithmetic & Advanced Math (from arithmetic_tools.py):
------------------------------------------------------
- add(a, b): Adds two numbers.
- subtract(a, b): Subtracts b from a.
- multiply(a, b): Multiplies two numbers.
- divide(a, b): Divides a by b.
- modulo(a, b): Calculates a % b.
- power(base, exponent): Calculates base raised to the exponent.
- sqrt(number): Calculates the square root.
- absolute(number): Calculates the absolute value.
- log(number, base): Calculates logarithm of a number with a given base.
- ln(number): Calculates the natural logarithm (base e).
- log10(number): Calculates the common logarithm (base 10).
- sin_degrees(angle): Sine of an angle in degrees.
- cos_degrees(angle): Cosine of an angle in degrees.
- tan_degrees(angle): Tangent of an angle in degrees.
- arcsin_degrees(value): Arc sine in degrees.
- arccos_degrees(value): Arc cosine in degrees.
- arctan_degrees(value): Arc tangent in degrees.
- factorial(n): Calculates n! (for non-negative integers).
- permutations(n, k): Calculates P(n, k).
- combinations(n, k): Calculates C(n, k).
- gcd(a, b): Greatest Common Divisor of two integers.
- lcm(a, b): Least Common Multiple of two integers.
- bitwise_and(a, b): Bitwise AND of two integers.
- bitwise_or(a, b): Bitwise OR of two integers.
- bitwise_xor(a, b): Bitwise XOR of two integers.
- bitwise_not(a): Bitwise NOT of an integer.
- left_shift(a, bits): Bitwise left shift.
- right_shift(a, bits): Bitwise right shift.

Linear Algebra (from linalg_tools.py, requires numpy):
----------------------------------------------------
- matrix_add(matrix_a, matrix_b): Adds two matrices.
- matrix_subtract(matrix_a, matrix_b): Subtracts matrix_b from matrix_a.
- matrix_scalar_multiply(matrix, scalar): Multiplies a matrix by a scalar.
- matrix_multiply(matrix_a, matrix_b): Multiplies two matrices.
- transpose(matrix): Transposes a matrix.
- solve_linear_system(matrix_a, vector_b): Solves Ax = b for x.
- eigen_decomposition(matrix_a): Computes eigenvalues and eigenvectors of matrix A.

Statistics (from stats_tools.py, uses statistics module, scipy, numpy):
---------------------------------------------------------
- average(numbers): Calculates the average of a list of numbers.
- median(numbers): Calculates the median of a list of numbers.
- mode(numbers): Calculates the mode (or modes) of a list of numbers.
- variance(numbers, population=False): Calculates the variance (sample by default).
- standard_deviation(numbers, population=False): Calculates the standard deviation (sample by default).
- confidence_interval_mean(sample_data, confidence_level=0.95, use_t_distribution=True):
  Calculates the confidence interval for the population mean.

Calculus (from calculus_tools.py):
----------------------------------
- numerical_derivative(func_str, x, h=1e-5): Numerical derivative of f(x) at x.
  (func_str is a Python expression, e.g., 'x**2' or 'math.sin(x)')
- numerical_integral(func_str, a, b, n=1000): Numerical integral of f(x) from a to b.
  (func_str is a Python expression, e.g., 'x**2' or 'math.sin(x)')

Probability & Distributions (from probability_tools.py, requires scipy):
----------------------------------------------------------------------
- normal_pdf(x, mean, std_dev): PDF of normal distribution.
- normal_cdf(x, mean, std_dev): CDF of normal distribution.
- normal_ppf(q, mean, std_dev): Inverse CDF (PPF) of normal distribution.
- binomial_pmf(k, n, p): PMF of binomial distribution.
- binomial_cdf(k, n, p): CDF of binomial distribution.
- binomial_ppf(q, n, p): Inverse CDF (PPF) of binomial distribution.
- poisson_pmf(k, mu): PMF of Poisson distribution.
- poisson_cdf(k, mu): CDF of Poisson distribution.
- poisson_ppf(q, mu): Inverse CDF (PPF) of Poisson distribution.

Optimization (from optimization_tools.py, requires scipy):
---------------------------------------------------------
- minimize_1d_function(func_str, bounds, method='bounded'): Minimizes a 1D function string f(x).
  (e.g., func_str="(x-3)**2", bounds=(0, 5))
- gradient_descent_1d(func_str, initial_x, learning_rate, iterations, derivative_func_str=None, h_derivative=1e-5):
  Performs 1D gradient descent. Uses numerical derivative if derivative_func_str is not given.

Information Theory (from information_theory_tools.py, requires numpy):
---------------------------------------------------------------------
- entropy(probabilities): Calculates Shannon entropy of a probability distribution.
- kl_divergence(p_probs, q_probs): Calculates KL divergence between two distributions P and Q.

Distance Metrics (from distance_metrics_tools.py, requires numpy):
------------------------------------------------------------------
- euclidean_distance(vec1, vec2): Euclidean distance between two vectors.
- manhattan_distance(vec1, vec2): Manhattan distance between two vectors.
- cosine_similarity(vec1, vec2): Cosine similarity between two vectors.

Algorithm Analysis (from algorithm_analysis_tools.py, uses math, numpy):
-----------------------------------------------------------------------
- big_o_evaluator(expression_str, variable_values): Evaluates a Big-O like expression string with variable values.
  (e.g., expression_str="2*n**2", variable_values={"n":10})
- master_theorem_solver(a, b, k, i=0): Solves T(n) = a*T(n/b) + Theta(n^k * (log n)^i).

Machine Learning Metrics (from ml_metrics_tools.py, uses numpy, stats_tools.average):
------------------------------------------------------------------------------------
- mean_squared_error(y_true, y_pred): Calculates Mean Squared Error.
- r_squared(y_true, y_pred): Calculates R-squared (coefficient of determination).
- cross_entropy_loss(y_true, y_pred, epsilon=1e-12): Calculates binary cross-entropy loss.
- confusion_matrix_metrics(tp, tn, fp, fn): Calculates metrics like accuracy, precision, recall, F1-score from confusion matrix components.

Graph Theory (from graph_tools.py, uses numpy):
------------------------------------------------
- node_degree(adjacency_matrix, node_index): Calculates degree of a node in an unweighted, undirected graph.
- graph_density(adjacency_matrix, is_directed=False): Calculates density of an unweighted graph.

Number Theory (from number_theory_tools.py, uses math):
------------------------------------------------------
- is_prime(n): Checks if an integer n is prime.
- prime_factorization(n): Returns the prime factorization of a positive integer n.

Conversion Tools (from conversion_tools.py, uses datetime):
-----------------------------------------------------------
- convert_temperature(value, from_unit, to_unit): Converts temperature (C, F, K).
- convert_base(number_str, from_base, to_base): Converts number string between bases 2-36.
- calculate_time_difference(datetime_str1, datetime_str2, format_str='%Y-%m-%d %H:%M:%S', output_unit='seconds'): Calculates time difference.
- convert_units(value, from_unit, to_unit, category): Converts units for 'length', 'weight', 'data_decimal', 'data_binary'.

Basic Cryptography (from crypto_basics_tools.py, uses hashlib):
--------------------------------------------------------------
- caesar_cipher(text, shift, mode='encrypt'): Encrypts/decrypts using Caesar cipher.
- vigenere_cipher(text, key, mode='encrypt'): Encrypts/decrypts using Vigenere cipher.
- calculate_md5(text): Calculates MD5 hash of a string.
- calculate_sha256(text): Calculates SHA256 hash of a string.

To use a tool, you can ask the assistant: "@CalculatorMCP <tool_name> <arguments>"
For example: "@CalculatorMCP add a=5 b=3" or "@CalculatorMCP normal_pdf x=0 mean=0 std_dev=1"
    """
    return info_string

if __name__ == "__main__":
    print("Starting CalculatorMCP server...")
    tool_getters = {
        "arithmetic": get_arithmetic_tools,
        "linear algebra": get_linalg_tools,
        "statistical": get_stats_tools,
        "calculus": get_calculus_tools,
        "probability": get_probability_tools,
        "optimization": get_optimization_tools,
        "information theory": get_information_theory_tools,
        "distance metrics": get_distance_metrics_tools,
        "algorithm analysis": get_algorithm_analysis_tools,
        "ml metrics": get_ml_metrics_tools,
        "graph theory": get_graph_tools,
        "number theory": get_number_theory_tools,
        "conversion": get_conversion_tools,
        "cryptography": get_crypto_basics_tools
    }
    for name, getter in tool_getters.items():
        try:
            tools = getter()
            print(f"Registered {name} tools: {[f.__name__ for f in tools]}")
        except Exception as e:
            print(f"Error registering {name} tools: {e}")

    print("MCP server 'CalculatorMCP' is attempting to run on port 8001 (forwarded from container's 8000).")
    print("Access the Swagger UI for testing at http://localhost:8001/docs")
    print("The MCP SSE endpoint should be http://localhost:8001/sse")
    print("Calculator info available at calculator://info")
    mcp.run(transport="sse") 