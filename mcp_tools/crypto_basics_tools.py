# mcp_tools/crypto_basics_tools.py
"""
Basic cryptography tools for the MCP Calculator (educational purposes).
"""
import hashlib
from typing import List, Dict, Any

def caesar_cipher(text: str, shift: int, mode: str = 'encrypt') -> str:
    """
    Encrypts or decrypts text using Caesar cipher.
    Handles uppercase and lowercase English letters. Non-alphabetic characters are unchanged.
    Args:
        text: The input string.
        shift: The integer shift value.
        mode: 'encrypt' or 'decrypt'. Default is 'encrypt'.
    Returns:
        The processed string.
    Raises:
        ValueError: If mode is invalid.
    """
    if mode.lower() not in ['encrypt', 'decrypt']:
        raise ValueError("Mode must be 'encrypt' or 'decrypt'.")

    if mode.lower() == 'decrypt':
        shift = -shift
    
    result = []
    for char in text:
        if 'a' <= char <= 'z':
            start = ord('a')
            result.append(chr((ord(char) - start + shift) % 26 + start))
        elif 'A' <= char <= 'Z':
            start = ord('A')
            result.append(chr((ord(char) - start + shift) % 26 + start))
        else:
            result.append(char)
    return "".join(result)

def vigenere_cipher(text: str, key: str, mode: str = 'encrypt') -> str:
    """
    Encrypts or decrypts text using Vigenere cipher.
    Handles uppercase and lowercase English letters. Non-alphabetic characters are unchanged.
    The key should only contain alphabetic characters.
    Args:
        text: The input string.
        key: The keyword string (alphabetic only).
        mode: 'encrypt' or 'decrypt'. Default is 'encrypt'.
    Returns:
        The processed string.
    Raises:
        ValueError: If mode is invalid or key is empty/non-alphabetic.
    """
    if mode.lower() not in ['encrypt', 'decrypt']:
        raise ValueError("Mode must be 'encrypt' or 'decrypt'.")
    if not key or not key.isalpha():
        raise ValueError("Key must be non-empty and contain only alphabetic characters.")

    key_upper = key.upper()
    key_len = len(key_upper)
    key_idx = 0
    result = []

    for char_text in text:
        if 'a' <= char_text <= 'z':
            start_text = ord('a')
            shift = ord(key_upper[key_idx % key_len]) - ord('A')
            if mode.lower() == 'decrypt':
                shift = -shift
            processed_char = chr((ord(char_text) - start_text + shift) % 26 + start_text)
            result.append(processed_char)
            key_idx += 1
        elif 'A' <= char_text <= 'Z':
            start_text = ord('A')
            shift = ord(key_upper[key_idx % key_len]) - ord('A')
            if mode.lower() == 'decrypt':
                shift = -shift
            processed_char = chr((ord(char_text) - start_text + shift) % 26 + start_text)
            result.append(processed_char)
            key_idx += 1
        else:
            result.append(char_text)
            
    return "".join(result)

def calculate_md5(text: str) -> str:
    """Calculates the MD5 hash of a string (UTF-8 encoded)."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def calculate_sha256(text: str) -> str:
    """Calculates the SHA256 hash of a string (UTF-8 encoded)."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_crypto_basics_tools():
    """Returns a list of basic cryptography tool functions."""
    return [
        caesar_cipher,
        vigenere_cipher,
        calculate_md5,
        calculate_sha256
    ] 