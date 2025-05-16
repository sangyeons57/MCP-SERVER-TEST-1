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

# --- Statistical Calculations (currently just average) ---
def average(numbers: list[float]) -> float:
    """Calculates the average of a list of numbers.
    Args:
        numbers: A list of numbers to calculate the average from.
    Returns:
        The average of the numbers.
    Raises:
        ValueError: If the list of numbers is empty.
    """
    if not numbers:
        raise ValueError("Cannot calculate the average of an empty list.")
    return sum(numbers) / len(numbers)

def get_arithmetic_tools() -> list:
    """Returns a list of all arithmetic and basic advanced math tool functions."""
    return [
        add, subtract, multiply, divide, modulo, power, sqrt, absolute,
        log, ln, log10, sin_degrees, cos_degrees, tan_degrees, factorial,
        average
    ] 