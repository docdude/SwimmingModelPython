import numpy as np

def savgol_coeffs(window_size, polyorder, deriv=0):
    """
    Compute the Savitzky-Golay filter coefficients.
    
    Parameters:
    window_size (int): The length of the filter window (must be odd).
    polyorder (int): The order of the polynomial used to fit the samples.
    deriv (int, optional): The order of the derivative to compute. Defaults to 0.
    
    Returns:
    numpy.ndarray: The filter coefficients.
    """
    # Validate input parameters
    if window_size < polyorder + 1:
        raise ValueError("window_size must be at least polyorder + 1")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if deriv > polyorder:
        raise ValueError("deriv must be less than or equal to polyorder")

    # Compute the Savitzky-Golay filter coefficients
    half_window = window_size // 2
    x = np.arange(-half_window, half_window + 1)
    
    # Compute the Vandermonde matrix
    A = np.vander(x, N=polyorder + 1, increasing=True)
    
    # Compute the pseudo-inverse using NumPy's linear algebra functions
    A_pinv = np.linalg.pinv(A)
    
    # Compute the coefficients for the specified derivative
    coeffs = A_pinv[deriv]
    
    return coeffs

# Example usage for different window sizes and polynomial orders
window_sizes = [5, 7, 9]
polyorders = [2, 3, 4]

for window_size in window_sizes:
    for polyorder in polyorders:
        print(f"Window Size: {window_size}, Polynomial Order: {polyorder}")
        coeffs = savgol_coeffs(window_size, polyorder)
        print("Coefficients: [", ", ".join(f"{value:.6f}" for value in coeffs), "]")
        print()
