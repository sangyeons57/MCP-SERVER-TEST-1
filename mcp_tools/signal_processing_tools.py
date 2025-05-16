import numpy as np
from typing import List, Tuple

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py

def fourier_transform(signal: List[float], sampling_rate: float = 1.0) -> Dict[str, List[any]]:
    """
    Computes the Discrete Fourier Transform (DFT) of a real-valued signal.
    Returns the frequencies and the complex-valued FFT results (amplitude and phase).

    Args:
        signal: A list of floats representing the input signal samples.
        sampling_rate: The number of samples per unit of time (e.g., Hz). Defaults to 1.0.
                       Used to calculate meaningful frequencies.
    Returns:
        A dictionary containing:
        'frequencies': A list of frequencies corresponding to the FFT output (up to Nyquist frequency).
        'fft_complex': A list of strings, where each string is the complex FFT output 
                       (e.g., "real+imagj") for the corresponding frequency.
        'fft_magnitude': A list of floats representing the magnitude (abs) of each FFT component.
        'fft_phase_degrees': A list of floats representing the phase (angle in degrees) of each FFT component.
    Raises:
        ValueError: If signal is empty.
    """
    if not signal:
        raise ValueError("Input signal cannot be empty.")

    n = len(signal)
    if n == 0:
        return {"frequencies": [], "fft_complex": [], "fft_magnitude": [], "fft_phase_degrees": []}

    # Compute FFT
    # numpy.fft.fft returns complex numbers. It computes the DFT.
    fft_result = np.fft.fft(signal)
    
    # Compute frequencies
    # numpy.fft.fftfreq generates the corresponding frequencies for each FFT bin.
    # We are interested in the positive frequencies (up to Nyquist limit for real signals)
    frequencies = np.fft.fftfreq(n, d=1.0/sampling_rate)

    # For real input, the FFT is symmetric. We only need to look at the first half (0 to N/2).
    # However, for a general tool, returning the full set might be expected by some, 
    # or the positive frequency half. Let's return the positive frequency half for typical analysis.
    # But np.fft.rfft and np.fft.rfftfreq are designed for real inputs and handle this more directly.

    # Using rfft for real inputs is more efficient and gives the positive frequency spectrum directly.
    fft_real_result = np.fft.rfft(signal) # Result for positive frequencies
    frequencies_real = np.fft.rfftfreq(n, d=1.0/sampling_rate) # Frequencies for rfft output

    fft_complex_str = [str(complex(c)) for c in fft_real_result]
    fft_magnitude = [float(abs(c)) for c in fft_real_result]
    fft_phase_degrees = [float(np.angle(c, deg=True)) for c in fft_real_result]

    return {
        "frequencies": frequencies_real.tolist(),
        "fft_complex": fft_complex_str,
        "fft_magnitude": fft_magnitude,
        "fft_phase_degrees": fft_phase_degrees
    }

def get_signal_processing_tools() -> list:
    """Returns a list of all signal processing tool functions."""
    return [
        fourier_transform
    ] 