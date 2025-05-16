# mcp_tools/conversion_tools.py
"""
Conversion and formatting tools for the MCP Calculator.
"""
import datetime
from typing import List, Dict, Any, Union

def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """
    Converts temperature between Celsius, Fahrenheit, and Kelvin.
    Args:
        value: The temperature value to convert.
        from_unit: Original unit ('C', 'F', or 'K'). Case-insensitive.
        to_unit: Target unit ('C', 'F', or 'K'). Case-insensitive.
    Returns:
        The converted temperature value.
    Raises:
        ValueError: If units are not recognized or are the same.
    """
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    valid_units = ['C', 'F', 'K']

    if from_unit not in valid_units or to_unit not in valid_units:
        raise ValueError(f"Invalid temperature unit. Choose from {valid_units}.")
    if from_unit == to_unit:
        return value

    # Convert to Celsius first as an intermediate step
    celsius_val = 0.0
    if from_unit == 'F':
        celsius_val = (value - 32) * 5/9
    elif from_unit == 'K':
        celsius_val = value - 273.15
    else: # from_unit == 'C'
        celsius_val = value

    # Convert from Celsius to target unit
    if to_unit == 'F':
        return (celsius_val * 9/5) + 32
    elif to_unit == 'K':
        return celsius_val + 273.15
    else: # to_unit == 'C'
        return celsius_val

def convert_base(number_str: str, from_base: int, to_base: int) -> str:
    """
    Converts a number string from one base to another (supports bases 2-36).
    Args:
        number_str: The number as a string in from_base. For bases > 10, uses A-Z.
        from_base: The original base of number_str (2-36).
        to_base: The target base (2-36).
    Returns:
        The number converted to to_base as a string.
    Raises:
        ValueError: If bases are out of range (2-36) or number_str is invalid for from_base.
    """
    if not (2 <= from_base <= 36 and 2 <= to_base <= 36):
        raise ValueError("Bases must be between 2 and 36.")
    
    try:
        # Convert to base 10 (decimal) first
        decimal_value = int(number_str, from_base)
    except ValueError:
        raise ValueError(f"Invalid number string '{number_str}' for base {from_base}.")

    if decimal_value == 0:
        return "0"

    # Convert from decimal to to_base
    digits = "0123456789abcdefghijklmnopqrstuvwxyz" # For bases up to 36
    result = ""
    temp_val = abs(decimal_value) # Handle negative numbers later if needed, but int() handles it for input

    while temp_val > 0:
        result = digits[temp_val % to_base] + result
        temp_val //= to_base
    
    if decimal_value < 0: # Not typically handled by int(str, base) for negative numbers, but good to consider.
                          # Python's int(str,base) handles negative sign if it's at the start
                          # for base 10, but not for others directly.
                          # Let's assume standard positive number conversion for now.
        # Python's int(value, base) would typically raise error if number_str has '-' for non-base 10
        # For simplicity, this function assumes positive numbers as typically used in base conversion tools
        # or that the sign is handled by the caller for non-base 10 inputs.
        # The `int(number_str, from_base)` already correctly parses signed numbers if from_base is 10
        # or if number_str starts with '-' and from_base is 0 (auto-detect).
        # For other bases, negative sign is usually not part of the `int(str, base)` standard.
        # This implementation primarily focuses on the magnitude conversion.
        pass # Standard library `int` handles signed input for base 10.
             # For other bases, it expects digits only.

    return result if result else "0"


def calculate_time_difference(datetime_str1: str, datetime_str2: str, 
                              format_str: str = "%Y-%m-%d %H:%M:%S", 
                              output_unit: str = "seconds") -> float:
    """
    Calculates the difference between two datetime strings.
    Args:
        datetime_str1: First datetime string.
        datetime_str2: Second datetime string.
        format_str: The strptime format for parsing the datetime strings.
                    Default is "%Y-%m-%d %H:%M:%S".
        output_unit: The unit for the output difference ('seconds', 'minutes', 'hours', 'days').
                     Default is 'seconds'.
    Returns:
        The time difference in the specified output_unit. Positive if datetime_str2 is later.
    Raises:
        ValueError: If datetime strings don't match format or output_unit is invalid.
    """
    try:
        dt1 = datetime.datetime.strptime(datetime_str1, format_str)
        dt2 = datetime.datetime.strptime(datetime_str2, format_str)
    except ValueError as e:
        raise ValueError(f"Error parsing datetime strings. Ensure they match format '{format_str}': {e}")

    difference = dt2 - dt1
    
    output_unit = output_unit.lower()
    if output_unit == "seconds":
        return difference.total_seconds()
    elif output_unit == "minutes":
        return difference.total_seconds() / 60
    elif output_unit == "hours":
        return difference.total_seconds() / 3600
    elif output_unit == "days":
        return difference.total_seconds() / (3600 * 24)
    else:
        raise ValueError("Invalid output_unit. Choose from 'seconds', 'minutes', 'hours', 'days'.")

# Basic unit conversion examples (can be expanded significantly)
_length_factors = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, 
                   "mi": 1609.34, "yd": 0.9144, "ft": 0.3048, "in": 0.0254}
_weight_factors = {"kg": 1.0, "g": 0.001, "mg": 0.000001, 
                   "lb": 0.453592, "oz": 0.0283495}
# Data size (powers of 2 for KiB, MiB etc, or powers of 10 for KB, MB - be specific)
# For simplicity, using common interpretations (1KB = 1000 B, 1KiB = 1024 B)
_data_size_factors_decimal = {"B":1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12, "PB": 10**15}
_data_size_factors_binary =  {"B":1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40, "PiB": 2**50}


def convert_units(value: float, from_unit: str, to_unit: str, category: str) -> float:
    """
    Converts a value between different units within a specified category.
    Supported categories: 'length', 'weight', 'data_decimal', 'data_binary'.
    
    Length units: m, km, cm, mm, mi, yd, ft, in
    Weight units: kg, g, mg, lb, oz
    Data_decimal units: B, KB, MB, GB, TB, PB (10^x)
    Data_binary units: B, KiB, MiB, GiB, TiB, PiB (2^x)

    Args:
        value: The numerical value to convert.
        from_unit: The original unit of the value.
        to_unit: The target unit to convert to.
        category: The category of conversion ('length', 'weight', 'data_decimal', 'data_binary').
    Returns:
        The converted value.
    Raises:
        ValueError: If category or units are not supported/recognized.
    """
    category = category.lower()
    factors = {}
    if category == "length":
        factors = _length_factors
    elif category == "weight":
        factors = _weight_factors
    elif category == "data_decimal":
        factors = _data_size_factors_decimal
    elif category == "data_binary":
        factors = _data_size_factors_binary
    else:
        raise ValueError(f"Unsupported category: {category}. Choose from 'length', 'weight', 'data_decimal', 'data_binary'.")

    if from_unit not in factors or to_unit not in factors:
        raise ValueError(f"Unsupported unit in category '{category}'. Supported units: {list(factors.keys())}")

    # Convert to base unit (e.g., meters for length, kilograms for weight, Bytes for data)
    value_in_base_unit = value * factors[from_unit]
    
    # Convert from base unit to target unit
    converted_value = value_in_base_unit / factors[to_unit]
    return converted_value

def get_conversion_tools():
    """Returns a list of conversion tool functions."""
    return [
        convert_temperature, 
        convert_base,
        calculate_time_difference,
        convert_units
    ] 