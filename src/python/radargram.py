import numpy as np
from scipy import signal
from scipy import stats
from skimage.filters import sobel


class Radargram:
    """
    A class for computing various attributes of radar data.

    Attributes can be pixel-based or window-based, with configurable window sizes.
    """

    def __init__(self, data=None):
        """
        Initialize the Radargram object.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data array
        """
        self.data = data

    # ---- Pixel-based attribute calculations ----

    def instantaneous_amplitude(self, data=None):
        """
        Calculate the instantaneous amplitude (envelope) using Hilbert transform.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data

        Returns:
        --------
        ndarray
            2D array of instantaneous amplitude (envelope)
        """
        data = self._get_data(data)
        envelope = np.zeros_like(data, dtype=float)

        # Process each trace (column) separately
        for i in range(data.shape[1]):
            trace = data[:, i]
            analytic_signal = signal.hilbert(trace)
            envelope[:, i] = np.abs(analytic_signal)

        return envelope

    def instantaneous_phase(self, data=None):
        """
        Calculate the instantaneous phase using Hilbert transform.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data

        Returns:
        --------
        tuple
            Tuple of (real part, imaginary part) of the phase
        """
        data = self._get_data(data)
        phase = np.zeros_like(data, dtype=float)

        # Process each trace (column) separately
        for i in range(data.shape[1]):
            trace = data[:, i]
            analytic_signal = signal.hilbert(trace)
            phase[:, i] = np.angle(analytic_signal)

        # Return real and imaginary parts
        return np.cos(phase), np.sin(phase)

    def instantaneous_frequency(self, data=None, dt=1.0):
        """
        Calculate the instantaneous frequency using the derivative of instantaneous phase.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        dt : float, optional
            Time sampling interval, defaults to 1.0

        Returns:
        --------
        ndarray
            2D array of instantaneous frequency
        """
        data = self._get_data(data)
        freq = np.zeros_like(data, dtype=float)

        # Process each trace (column) separately
        for i in range(data.shape[1]):
            trace = data[:, i]
            analytic_signal = signal.hilbert(trace)
            phase = np.angle(analytic_signal)
            unwrapped_phase = np.unwrap(phase)
            # Calculate the derivative and convert to frequency
            freq[1:, i] = np.diff(unwrapped_phase) / (2.0 * np.pi * dt)
            # First sample is set to second sample to avoid edge effects
            freq[0, i] = freq[1, i]

        return freq

    def absolute_gradient(self, data=None):
        """
        Calculate the absolute gradient using Sobel filter.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data

        Returns:
        --------
        ndarray
            2D array of absolute gradient
        """
        data = self._get_data(data)
        # Use sobel filter for gradient calculation
        sobel_h = sobel(data, axis=0)
        sobel_v = sobel(data, axis=1)
        return np.sqrt(sobel_h**2 + sobel_v**2)

    # ---- Window-based attribute calculations ----

    def average_energy(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the average energy within a window (optimized version).

        Sum of squared amplitudes divided by the number of samples in the window.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        x_dis : int, optional
            Distance to window edge in x-direction, defaults to 1
        y_dis : int, optional
            Distance to window edge in y-direction, defaults to 1
        mode : str, optional
            Padding mode for scipy.ndimage.filters, defaults to 'mirror'

        Returns:
        --------
        ndarray
            2D array of average energy
        """
        from scipy.ndimage import uniform_filter

        data = self._get_data(data)
        squared_data = data**2

        # Use uniform filter (moving average) which is much faster
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)
        return uniform_filter(squared_data, size=window_size, mode=mode)

    def rms_amplitude(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the RMS amplitude within a window.

        Square root of the sum of squared amplitudes divided by the number of samples in the window.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        x_dis : int, optional
            Distance to window edge in x-direction, defaults to 1
        y_dis : int, optional
            Distance to window edge in y-direction, defaults to 1
        mode : str, optional
            Padding mode for numpy.pad, defaults to 'mirror'

        Returns:
        --------
        ndarray
            2D array of RMS amplitude
        """
        avg_energy = self.average_energy(data, x_dis, y_dis, mode)
        return np.sqrt(avg_energy)

    def coherence(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the coherence within a window (optimized version).

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        x_dis : int, optional
            Distance to window edge in x-direction, defaults to 1
        y_dis : int, optional
            Distance to window edge in y-direction, defaults to 1
        mode : str, optional
            Padding mode for scipy.ndimage.filters, defaults to 'mirror'

        Returns:
        --------
        ndarray
            2D array of coherence
        """
        from scipy.ndimage import uniform_filter, generic_filter

        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)

        # Calculate mean and std using fast filters
        local_mean = uniform_filter(data, size=window_size, mode=mode)
        local_mean_sq = uniform_filter(data**2, size=window_size, mode=mode)
        local_std = np.sqrt(local_mean_sq - local_mean**2)

        # Calculate coherence
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # Ignore division by zero warnings
            coherence = 1 - (local_std / np.abs(local_mean))

        # Fix NaN and inf values
        coherence = np.nan_to_num(coherence, nan=0, posinf=1, neginf=0)
        return coherence

    def entropy(self, data=None, x_dis=1, y_dis=1, mode="mirror", bins=16):
        """
        Calculate the entropy within a window (optimized version).

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        x_dis : int, optional
            Distance to window edge in x-direction, defaults to 1
        y_dis : int, optional
            Distance to window edge in y-direction, defaults to 1
        mode : str, optional
            Padding mode for scipy.ndimage.filters, defaults to 'mirror'
        bins : int, optional
            Number of bins for histogram, reduced to 16 for performance, defaults to 16

        Returns:
        --------
        ndarray
            2D array of entropy
        """
        from scipy.ndimage import generic_filter

        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)

        # Define a function to calculate entropy of a window
        def calc_entropy(window):
            hist, _ = np.histogram(window, bins=bins, density=True)
            nonzero = hist > 0
            if np.any(nonzero):
                return -np.sum(hist[nonzero] * np.log2(hist[nonzero]))
            return 0

        # Apply the filter
        return generic_filter(data, calc_entropy, size=window_size, mode=mode)

    def mean(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the mean within a window (optimized version).

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        x_dis : int, optional
            Distance to window edge in x-direction, defaults to 1
        y_dis : int, optional
            Distance to window edge in y-direction, defaults to 1
        mode : str, optional
            Padding mode for scipy.ndimage.filters, defaults to 'mirror'

        Returns:
        --------
        ndarray
            2D array of mean values
        """
        from scipy.ndimage import uniform_filter

        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)
        return uniform_filter(data, size=window_size, mode=mode)

    def median(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the median within a window (optimized version).

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        x_dis : int, optional
            Distance to window edge in x-direction, defaults to 1
        y_dis : int, optional
            Distance to window edge in y-direction, defaults to 1
        mode : str, optional
            Padding mode for scipy.ndimage.filters, defaults to 'mirror'

        Returns:
        --------
        ndarray
            2D array of median values
        """
        from scipy.ndimage import median_filter

        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)
        return median_filter(data, size=window_size, mode=mode)

    def std(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the standard deviation within a window (optimized version).

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        x_dis : int, optional
            Distance to window edge in x-direction, defaults to 1
        y_dis : int, optional
            Distance to window edge in y-direction, defaults to 1
        mode : str, optional
            Padding mode for scipy.ndimage.filters, defaults to 'mirror'

        Returns:
        --------
        ndarray
            2D array of standard deviation values
        """
        from scipy.ndimage import uniform_filter

        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)

        # Calculate mean and mean of squares
        mean = uniform_filter(data, size=window_size, mode=mode)
        mean_sq = uniform_filter(data**2, size=window_size, mode=mode)

        # Calculate std dev
        variance = mean_sq - mean**2
        # Fix potential small negative values due to floating-point errors
        variance = np.maximum(variance, 0)
        return np.sqrt(variance)

    def skewness(self, data=None, x_dis=1, y_dis=1, mode="symmetric"):
        """
        Calculate the skewness within a window.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        x_dis : int, optional
            Distance to window edge in x-direction, defaults to 1
        y_dis : int, optional
            Distance to window edge in y-direction, defaults to 1
        mode : str, optional
            Padding mode for numpy.pad, defaults to 'symmetric'

        Returns:
        --------
        ndarray
            2D array of skewness values
        """
        data = self._get_data(data)
        return self._apply_window_operation(data, stats.skew, x_dis, y_dis, mode)

    def kurtosis(self, data=None, x_dis=1, y_dis=1, mode="symmetric"):
        """
        Calculate the kurtosis within a window.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        x_dis : int, optional
            Distance to window edge in x-direction, defaults to 1
        y_dis : int, optional
            Distance to window edge in y-direction, defaults to 1
        mode : str, optional
            Padding mode for numpy.pad, defaults to 'symmetric'

        Returns:
        --------
        ndarray
            2D array of kurtosis values
        """
        data = self._get_data(data)
        return self._apply_window_operation(data, stats.kurtosis, x_dis, y_dis, mode)

    def max(self, data=None, x_dis=1, y_dis=1, mode="symmetric"):
        """
        Calculate the maximum value within a window.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        x_dis : int, optional
            Distance to window edge in x-direction, defaults to 1
        y_dis : int, optional
            Distance to window edge in y-direction, defaults to 1
        mode : str, optional
            Padding mode for numpy.pad, defaults to 'symmetric'

        Returns:
        --------
        ndarray
            2D array of maximum values
        """
        data = self._get_data(data)
        return self._apply_window_operation(data, np.max, x_dis, y_dis, mode)

    def min(self, data=None, x_dis=1, y_dis=1, mode="symmetric"):
        """
        Calculate the minimum value within a window.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        x_dis : int, optional
            Distance to window edge in x-direction, defaults to 1
        y_dis : int, optional
            Distance to window edge in y-direction, defaults to 1
        mode : str, optional
            Padding mode for numpy.pad, defaults to 'symmetric'

        Returns:
        --------
        ndarray
            2D array of minimum values
        """
        data = self._get_data(data)
        return self._apply_window_operation(data, np.min, x_dis, y_dis, mode)

    def range(self, data=None, x_dis=1, y_dis=1, mode="symmetric"):
        """
        Calculate the range (max - min) within a window.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        x_dis : int, optional
            Distance to window edge in x-direction, defaults to 1
        y_dis : int, optional
            Distance to window edge in y-direction, defaults to 1
        mode : str, optional
            Padding mode for numpy.pad, defaults to 'symmetric'

        Returns:
        --------
        ndarray
            2D array of range values
        """
        max_vals = self.max(data, x_dis, y_dis, mode)
        min_vals = self.min(data, x_dis, y_dis, mode)
        return max_vals - min_vals

    # ---- Helper methods ----

    def _get_data(self, data):
        """Helper method to get the data array to use."""
        if data is None:
            if self.data is None:
                raise ValueError("No data provided and no data stored in object")
            return self.data
        return data

    def _apply_window_operation_fast(
        self, data, operation, x_dis, y_dis, mode, normalize=False
    ):
        """
        Apply a window operation to the data using fast methods.

        Parameters:
        -----------
        data : ndarray
            Input data
        operation : callable
            Function to apply to each window
        x_dis : int
            Distance to window edge in x-direction
        y_dis : int
            Distance to window edge in y-direction
        mode : str
            Padding mode for numpy.pad
        normalize : bool, optional
            Whether to normalize by window size, defaults to False

        Returns:
        --------
        ndarray
            Result of applying the operation to each window
        """
        from scipy.ndimage import generic_filter

        # Create a window for the operation
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)

        # Apply the filter
        result = generic_filter(
            data, lambda x: operation(x), size=window_size, mode=mode
        )

        # Normalize if required
        if normalize:
            result /= np.prod(window_size)

        return result

    def _apply_window_operation(
        self, data, operation, x_dis, y_dis, mode, normalize=False
    ):
        """
        Apply a window operation to the data.

        Parameters:
        -----------
        data : ndarray
            Input data
        operation : callable
            Function to apply to each window
        x_dis : int
            Distance to window edge in x-direction
        y_dis : int
            Distance to window edge in y-direction
        mode : str
            Padding mode for numpy.pad
        normalize : bool, optional
            Whether to normalize by window size, defaults to False

        Returns:
        --------
        ndarray
            Result of applying the operation to each window
        """
        padded_data = np.pad(data, ((y_dis, y_dis), (x_dis, x_dis)), mode=mode)
        result = np.zeros_like(data, dtype=float)

        window_size = (2 * y_dis + 1) * (2 * x_dis + 1)

        # For each pixel, apply the operation to its neighborhood
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # Extract window
                window = padded_data[
                    i : i + 2 * y_dis + 1, j : j + 2 * x_dis + 1
                ].flatten()
                # Apply operation
                try:
                    if normalize:
                        result[i, j] = operation(window) / window_size
                    else:
                        result[i, j] = operation(window)
                except:
                    # Handle cases where the operation fails (e.g., all NaN)
                    result[i, j] = np.nan

        return result


# Example usage:
if __name__ == "__main__":
    # Generate sample data
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    # Create radargram object
    rg = Radargram(Z)

    # Calculate attributes
    envelope = rg.instantaneous_amplitude()
    phase_real, phase_imag = rg.instantaneous_phase()
    freq = rg.instantaneous_frequency()
    avg_energy = rg.average_energy(x_dis=2, y_dis=2)
    rms = rg.rms_amplitude(x_dis=2, y_dis=2)

    # Print shapes
    print(f"Data shape: {Z.shape}")
    print(f"Envelope shape: {envelope.shape}")
    print(f"Phase real shape: {phase_real.shape}")
    print(f"Frequency shape: {freq.shape}")
    print(f"Average energy shape: {avg_energy.shape}")
    print(f"RMS amplitude shape: {rms.shape}")
