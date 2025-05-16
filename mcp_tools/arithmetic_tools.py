import math

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py
# when these functions are registered.

# --- Basic Arithmetic Operations ---
def add(a: float, b: float) -> float:
    """Adds two numbers.
    Args:
        a: The first number.
        b: The second number.
    Returns:
        The sum of the two numbers.
    """
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtracts the second number from the first.
    Args:
        a: The first number.
        b: The number to subtract.
    Returns:
        The difference between the two numbers.
    """
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiplies two numbers.
    Args:
        a: The first number.
        b: The second number.
    Returns:
        The product of the two numbers.
    """
    return a * b

def divide(a: float, b: float) -> float:
    """Divides the first number by the second.
    Args:
        a: The dividend.
        b: The divisor (cannot be zero).
    Returns:
        The quotient.
    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

def modulo(a: float, b: float) -> float:
    """Calculates the remainder of the division of a by b.
    Args:
        a: The dividend.
        b: The divisor (cannot be zero).
    Returns:
        The remainder of a divided by b.
    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot calculate modulo with zero divisor.")
    return a % b

def power(a: float, b: float) -> float:
    """Calculates the first number raised to the power of the second number.
    Args:
        a: The base.
        b: The exponent.
    Returns:
        a raised to the power of b.
    """
    return a ** b

def sqrt(number: float) -> float:
    """Calculates the square root of a number.
    Args:
        number: The number to calculate the square root of (must be non-negative).
    Returns:
        The square root of the number.
    Raises:
        ValueError: If the number is negative.
    """
    if number < 0:
        raise ValueError("Cannot calculate the square root of a negative number.")
    return math.sqrt(number)

def absolute(number: float) -> float:
    """Calculates the absolute value of a number.
    Args:
        number: The number to calculate the absolute value of.
    Returns:
        The absolute value of the number.
    """
    return math.fabs(number)

# --- Advanced Mathematical Operations ---
def log(number: float, base: float) -> float:
    """Calculates the logarithm of a number with a specified base.
    Args:
        number: The number to calculate the logarithm of (must be positive).
        base: The base of the logarithm (must be positive and not equal to 1).
    Returns:
        The logarithm of the number with the given base.
    Raises:
        ValueError: If number or base is not positive, or if base is 1.
    """
    if number <= 0:
        raise ValueError("Logarithm is defined only for positive numbers.")
    if base <= 0 or base == 1:
        raise ValueError("Logarithm base must be positive and not equal to 1.")
    return math.log(number, base)

def ln(number: float) -> float:
    """Calculates the natural logarithm (base e) of a number.
    Args:
        number: The number to calculate the natural logarithm of (must be positive).
    Returns:
        The natural logarithm of the number.
    Raises:
        ValueError: If the number is not positive.
    """
    if number <= 0:
        raise ValueError("Natural logarithm is defined only for positive numbers.")
    return math.log(number)

def log10(number: float) -> float:
    """Calculates the common logarithm (base 10) of a number.
    Args:
        number: The number to calculate the common logarithm of (must be positive).
    Returns:
        The common logarithm of the number.
    Raises:
        ValueError: If the number is not positive.
    """
    if number <= 0:
        raise ValueError("Common logarithm is defined only for positive numbers.")
    return math.log10(number)

def sin_degrees(angle_degrees: float) -> float:
    """Calculates the sine of an angle given in degrees.
    Args:
        angle_degrees: The angle in degrees.
    Returns:
        The sine of the angle.
    """
    return math.sin(math.radians(angle_degrees))

def cos_degrees(angle_degrees: float) -> float:
    """Calculates the cosine of an angle given in degrees.
    Args:
        angle_degrees: The angle in degrees.
    Returns:
        The cosine of the angle.
    """
    return math.cos(math.radians(angle_degrees))

def tan_degrees(angle_degrees: float) -> float:
    """Calculates the tangent of an angle given in degrees.
    Args:
        angle_degrees: The angle in degrees.
    Returns:
        The tangent of the angle.
    """
    if (angle_degrees % 180) == 90:
        # For angles like 90, 270, tan is undefined. 
        # math.tan might return a very large float or raise an error depending on the system.
        # Consider raising a specific error or returning float('inf') or float('-inf').
        # For now, we let math.tan handle it but this could be a point of refinement.
        pass 
    return math.tan(math.radians(angle_degrees))

def factorial(n: int) -> int:
    """Calculates the factorial of a non-negative integer.
    Args:
        n: The integer to calculate the factorial of (must be non-negative).
    Returns:
        The factorial of n.
    Raises:
        ValueError: If n is negative or not an integer.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("Factorial is defined only for non-negative integers.")
    return math.factorial(n)

# --- NEW Advanced Mathematical Operations ---
def arcsin_degrees(value: float) -> float:
    """Calculates the arc sine of a value, returning the angle in degrees.
    Args:
        value: The value to calculate arc sine of (-1 <= value <= 1).
    Returns:
        The angle in degrees.
    Raises:
        ValueError: If value is outside the range [-1, 1].
    """
    if not (-1 <= value <= 1):
        raise ValueError("Arc sine is defined only for values between -1 and 1.")
    return math.degrees(math.asin(value))

def arccos_degrees(value: float) -> float:
    """Calculates the arc cosine of a value, returning the angle in degrees.
    Args:
        value: The value to calculate arc cosine of (-1 <= value <= 1).
    Returns:
        The angle in degrees.
    Raises:
        ValueError: If value is outside the range [-1, 1].
    """
    if not (-1 <= value <= 1):
        raise ValueError("Arc cosine is defined only for values between -1 and 1.")
    return math.degrees(math.acos(value))

def arctan_degrees(value: float) -> float:
    """Calculates the arc tangent of a value, returning the angle in degrees.
    Args:
        value: The value to calculate arc tangent of.
    Returns:
        The angle in degrees.
    """
    return math.degrees(math.atan(value))

def arctan2_degrees(y: float, x: float) -> float:
    """Calculates the arc tangent of y/x using two arguments, returning the angle in degrees.
    This function can determine the correct quadrant for the angle.
    Args:
        y: The y-coordinate.
        x: The x-coordinate.
    Returns:
        The angle in degrees (-180 to 180).
    """
    return math.degrees(math.atan2(y, x))

def permutation(n: int, k: int) -> int:
    """Calculates the number of permutations P(n, k).
    Args:
        n: The total number of items.
        k: The number of items to arrange.
    Returns:
        The number of permutations.
    Raises:
        ValueError: If n or k are negative, or if k > n.
    """
    if not isinstance(n, int) or not isinstance(k, int) or n < 0 or k < 0:
        raise ValueError("Permutation arguments must be non-negative integers.")
    if k > n:
        raise ValueError("k cannot be greater than n for permutations.")
    return math.perm(n, k)

def combination(n: int, k: int) -> int:
    """Calculates the number of combinations C(n, k).
    Args:
        n: The total number of items.
        k: The number of items to choose.
    Returns:
        The number of combinations.
    Raises:
        ValueError: If n or k are negative, or if k > n.
    """
    if not isinstance(n, int) or not isinstance(k, int) or n < 0 or k < 0:
        raise ValueError("Combination arguments must be non-negative integers.")
    if k > n:
        raise ValueError("k cannot be greater than n for combinations.")
    return math.comb(n, k)

# --- Integer Operations ---
def gcd(a: int, b: int) -> int:
    """Calculates the Greatest Common Divisor (GCD) of two integers.
    Args:
        a: The first integer.
        b: The second integer.
    Returns:
        The GCD of a and b.
    """
    return math.gcd(a, b)

def lcm(a: int, b: int) -> int:
    """Calculates the Least Common Multiple (LCM) of two integers.
    Args:
        a: The first integer.
        b: The second integer.
    Returns:
        The LCM of a and b.
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b) # LCM using GCD

# --- Bitwise Operations ---
def bitwise_and(a: int, b: int) -> int:
    """Performs a bitwise AND operation on two integers.
    Args:
        a: The first integer.
        b: The second integer.
    Returns:
        The result of a AND b.
    """
    return a & b

def bitwise_or(a: int, b: int) -> int:
    """Performs a bitwise OR operation on two integers.
    Args:
        a: The first integer.
        b: The second integer.
    Returns:
        The result of a OR b.
    """
    return a | b

def bitwise_xor(a: int, b: int) -> int:
    """Performs a bitwise XOR operation on two integers.
    Args:
        a: The first integer.
        b: The second integer.
    Returns:
        The result of a XOR b.
    """
    return a ^ b

def bitwise_not(a: int) -> int:
    """Performs a bitwise NOT operation on an integer.
    Note: Python's ~ operator behaves as -(x+1) for two's complement.
    Args:
        a: The integer.
    Returns:
        The result of NOT a.
    """
    return ~a

def left_shift(a: int, bits: int) -> int:
    """Performs a bitwise left shift operation.
    Args:
        a: The integer to shift.
        bits: The number of bits to shift left.
    Returns:
        The result of a << bits.
    Raises:
        ValueError: If bits is negative.
    """
    if bits < 0:
        raise ValueError("Shift bit count cannot be negative.")
    return a << bits

def right_shift(a: int, bits: int) -> int:
    """Performs a bitwise right shift operation.
    Args:
        a: The integer to shift.
        bits: The number of bits to shift right.
    Returns:
        The result of a >> bits.
    Raises:
        ValueError: If bits is negative.
    """
    if bits < 0:
        raise ValueError("Shift bit count cannot be negative.")
    return a >> bits

# --- Statistical Calculations (currently just average) ---
def get_arithmetic_tools() -> list:
    """Returns a list of all arithmetic and basic advanced math tool functions."""
    # Add new functions to this list
    return [
        add, subtract, multiply, divide, modulo, power, sqrt, absolute,
        log, ln, log10, sin_degrees, cos_degrees, tan_degrees, factorial,
        arcsin_degrees, arccos_degrees, arctan_degrees, arctan2_degrees,
        permutation, combination,
        gcd, lcm,
        bitwise_and, bitwise_or, bitwise_xor, bitwise_not, left_shift, right_shift
    ] 