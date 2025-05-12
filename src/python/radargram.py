import numpy as np
from scipy.ndimage import (
    generic_filter,
    uniform_filter,
    maximum_filter,
    minimum_filter,
    median_filter,
)
from scipy import signal
from skimage.filters import sobel
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class Radargram:
    """
    A class for computing various attributes of radar data.

    Attributes can be pixel-based or window-based, with configurable window sizes.
    Supports two modes:
    1. On-demand mode: Radargram(data) - attributes are calculated when requested
    2. Pre-computed mode: Radargram(data, x_dis, y_dis) - attributes are calculated during initialization
    """

    def __init__(self, data=None, x_dis=None, y_dis=None):
        """
        Initialize the Radargram object.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data array
        x_dis : int, optional
            Distance to window edge in x-direction for pre-computing attributes
        y_dis : int, optional
            Distance to window edge in y-direction for pre-computing attributes
        mode : str, optional
            Padding mode for window operations, defaults to 'mirror'
        dt : float, optional
            Time sampling interval for instantaneous frequency calculation
        """
        self.data = data
        self._precalc = x_dis is not None and y_dis is not None
        self._x_dis = 1 if x_dis is None else x_dis
        self._y_dis = 1 if y_dis is None else y_dis

        # Storage for pre-calculated attributes
        self.instantaneous_amplitude = None
        self.instantaneous_phase_real = None
        self.instantaneous_phase_imag = None
        self.instantaneous_frequency = None
        self.absolute_gradient = None
        self.average_energy = None
        self.rms_amplitude = None
        self.coherence = None
        self.entropy = None
        self.mean = None
        self.median = None
        self.std = None
        self.skewness = None
        self.kurtosis = None
        self.max = None
        self.min = None
        self.range = None

        # Pre-calculate attributes if in pre-computed mode
        if self._precalc and data is not None:
            print(
                f"Pre-calculating attributes with window size {2 * self._x_dis + 1}x{2 * self._y_dis + 1}..."
            )
            print("Calculating Instantaneous Amplitude ...")
            self.instantaneous_amplitude = self.calc_instantaneous_amplitude()
            print("Calculating Instantaneous Phase ...")
            self.instantaneous_phase_real, self.instantaneous_phase_imag = (
                self.calc_instantaneous_phase()
            )
            print("Calculating Instantaneous Frequency ...")
            self.instantaneous_frequency = self.calc_instantaneous_frequency()
            print("Calculating Sweetness ...")
            self.sweetness = self.calc_sweetness()
            print("Calculating Absolute Gradient ...")
            self.absolute_gradient = self.calc_absolute_gradient()
            print("Calculating Average Energy ...")
            self.average_energy = self.calc_average_energy(x_dis=x_dis, y_dis=y_dis)
            print("Calculating RMS Amplitude ...")
            self.rms_amplitude = self.calc_rms_amplitude(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Coherence ...")
            self.coherence = self.calc_coherence(x_dis=x_dis, y_dis=y_dis)
            print("Skip Entropy ...")
            # self.entropy = self.calc_entropy(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Mean ...")
            self.mean = self.calc_mean(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Mean of Squared Values ...")
            self.mean_sq = self.calc_mean_sq(x_dis=x_dis, y_dis=y_dis)
            print("Skip Median ...")
            # self.median = self.calc_median(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Standard Deviation ...")
            self.std = self.calc_std(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Skewness ...")
            self.skewness = self.calc_skewness(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Kurtosis ...")
            self.kurtosis = self.calc_kurtosis(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Maximum ...")
            self.max = self.calc_max(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Minimum ...")
            self.min = self.calc_min(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Range ...")
            self.range = self.calc_range(x_dis=x_dis, y_dis=y_dis)
            print("Pre-calculation complete.")

    # ---- Pixel-based attribute calculations ----

    def calc_instantaneous_amplitude(self, data=None):
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

    def calc_instantaneous_phase(self, data=None):
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

    def calc_instantaneous_frequency(self, data=None, dt=0.1173):
        """
        Calculate the instantaneous frequency using the derivative of instantaneous phase.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        dt : float, optional
            Time sampling interval, defaults to 0.1173

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

    def calc_sweetness(self, data=None):
        """
        Calculate the sweetness attribute, defined as instantaneous amplitude
        divided by instantaneous frequency.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data

        Returns:
        --------
        ndarray
            2D array of sweetness values
        """
        if (
            self._precalc
            and self.instantaneous_amplitude is not None
            and self.instantaneous_frequency is not None
        ):
            inst_amp = self.instantaneous_amplitude
            inst_freq = self.instantaneous_frequency
        else:
            inst_amp = self.instantaneous_amplitude(data)
            inst_freq = self.instantaneous_frequency(data)

        with np.errstate(divide="ignore", invalid="ignore"):
            sweetness = inst_amp / inst_freq
            sweetness = np.nan_to_num(sweetness, nan=0.0, posinf=0.0, neginf=0.0)

        return sweetness

    def calc_absolute_gradient(self, data=None):
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

    def calc_average_energy(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
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

        data = self._get_data(data)
        squared_data = data**2

        # Use uniform filter (moving average) which is much faster
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)
        return uniform_filter(squared_data, size=window_size, mode=mode)

    def calc_rms_amplitude(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
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
        if self._precalc and self.average_energy is not None:
            avg_energy = self.average_energy
        else:
            avg_energy = self.calc_average_energy(data, x_dis, y_dis, mode)
        return np.sqrt(avg_energy)

    def calc_coherence(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
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

    def calc_entropy(self, data=None, x_dis=1, y_dis=1, mode="mirror", bins=16):
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

        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)

        # Define a function to calculate entropy of a window
        def calc_calc_entropy(window):
            hist, _ = np.histogram(window, bins=bins, density=True)
            nonzero = hist > 0
            if np.any(nonzero):
                return -np.sum(hist[nonzero] * np.log2(hist[nonzero]))
            return 0

        # Apply the filter
        return generic_filter(data, calc_calc_entropy, size=window_size, mode=mode)

    def calc_mean(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
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

        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)
        return uniform_filter(data, size=window_size, mode=mode)

    def calc_mean_sq(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the mean of squared values within a window (optimized version).

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
            2D array of mean squared values
        """
        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)
        return uniform_filter(data**2, size=window_size, mode=mode)

    def calc_median(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
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

        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)
        return median_filter(data, size=window_size, mode=mode)

    def calc_std(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
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

        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)

        # Use precomputed mean and mean_sq if available
        if self._precalc and self.mean is not None and self.mean_sq is not None:
            mean = self.mean
            mean_sq = self.mean_sq
        else:
            mean = uniform_filter(data, size=window_size, mode=mode)
            mean_sq = uniform_filter(data**2, size=window_size, mode=mode)

        # Calculate std dev
        variance = mean_sq - mean**2
        # Fix potential small negative values due to floating-point errors
        variance = np.maximum(variance, 0)
        return np.sqrt(variance)

    def calc_skewness(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the skewness within a window using central moments formula.

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
            2D array of skewness values
        """
        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)

        # Use precomputed mean and std if available
        if self._precalc and self.mean is not None and self.std is not None:
            local_mean = self.mean
            local_std = self.std
        else:
            local_mean = self.mean(data, x_dis, y_dis, mode)
            local_std = self.std(data, x_dis, y_dis, mode)

        # Calculate third central moment using uniform_filter
        # (x-μ)^3
        diff_cubed = (data - local_mean) ** 3
        third_moment = uniform_filter(diff_cubed, size=window_size, mode=mode)

        # Normalize by std^3 to get skewness
        with np.errstate(divide="ignore", invalid="ignore"):
            skew = third_moment / (local_std**3)

        # Fix potential NaN or inf values
        return np.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0)

    def calc_kurtosis(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the kurtosis within a window using central moments formula.

        Returns kurtosis - 3 (excess kurtosis) to match scipy.stats.kurtosis.

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
            2D array of excess kurtosis values
        """
        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)

        # Use precomputed mean and std if available
        if self._precalc and self.mean is not None and self.std is not None:
            local_mean = self.mean
            local_std = self.std
        else:
            local_mean = self.mean(data, x_dis, y_dis, mode)
            local_std = self.std(data, x_dis, y_dis, mode)

        # Calculate fourth central moment using uniform_filter
        # (x-μ)^4
        diff_to_4th = (data - local_mean) ** 4
        fourth_moment = uniform_filter(diff_to_4th, size=window_size, mode=mode)

        # Normalize by std^4 to get kurtosis
        with np.errstate(divide="ignore", invalid="ignore"):
            kurt = fourth_moment / (local_std**4)

        # Subtract 3 to get excess kurtosis (normal distribution has kurtosis=3)
        excess_kurt = kurt - 3

        # Fix potential NaN or inf values
        return np.nan_to_num(excess_kurt, nan=0.0, posinf=0.0, neginf=0.0)

    def calc_min(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the minimum value within a window (optimized version).

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
            2D array of minimum values
        """

        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)
        return minimum_filter(data, size=window_size, mode=mode)

    def calc_max(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the maximum value within a window (optimized version).

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
            2D array of maximum values
        """

        data = self._get_data(data)
        window_size = (2 * y_dis + 1, 2 * x_dis + 1)
        return maximum_filter(data, size=window_size, mode=mode)

    def calc_range(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
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
            Padding mode for numpy.pad, defaults to 'mirror'

        Returns:
        --------
        ndarray
            2D array of range values
        """
        if self._precalc and self.max is not None and self.min is not None:
            max_vals = self.max
            min_vals = self.min
        else:
            max_vals = self.max(data, x_dis, y_dis, mode)
            min_vals = self.min(data, x_dis, y_dis, mode)
        return max_vals - min_vals

    # ---- Helper methods ----

    def apply_gain(
        self,
        data=None,
        method="exponential",
        params=None,
        window_size=None,
        visualize=False,
    ):
        """
        Apply a gain function to compensate for signal attenuation.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        method : str, optional
            Method for fitting the damping function:
            - 'exponential': A*exp(-α*t)
            - 'power': A*t^(-α)
            - 'combined': A*t^(-1)*exp(-α*t)
            - 'polynomial': polynomial of degree specified in params
            - 'moving_average': smoothed empirical function
            Defaults to 'exponential'
        params : dict, optional
            Additional parameters for the fitting method:
            - For 'polynomial': {'degree': n} where n is the polynomial degree
            - For 'moving_average': {'window_length': w} for smoothing window
        window_size : int, optional
            Size of averaging window for smoothing the mean trace.
            If None, no smoothing is applied. Defaults to None.
        visualize : bool, optional
            If True, displays plots of mean trace, fitted function,
            and before/after comparison. Defaults to False.

        Returns:
        --------
        ndarray
            Gain-corrected data with same shape as input
        """

        # Get data and compute mean trace
        data = self._get_data(data)
        data = self.instantaneous_amplitude(data)
        mean_trace = np.mean(data, axis=1)

        # Smooth mean trace if window_size provided
        if window_size is not None:
            mean_trace = signal.savgol_filter(mean_trace, window_size, 2)

        # Time axis (sample indices)
        t = np.arange(len(mean_trace))

        # Define fitting functions based on method
        if method == "exponential":
            # A*exp(-α*t)
            def damping_func(t, A, alpha):
                return A * np.exp(-alpha * t)

            # Initial guess
            p0 = [np.max(mean_trace), 0.01]

            # Fit the function to mean trace
            try:
                popt, _ = curve_fit(
                    damping_func,
                    t,
                    mean_trace,
                    p0=p0,
                    bounds=([0, 0], [np.inf, np.inf]),
                )
                fitted_curve = damping_func(t, *popt)

                plt.figure(figsize=(4, 8))
                plt.plot(mean_trace, t, "b-", label="Mean Trace")
                plt.plot(fitted_curve, t, "r--", label="Fitted Curve")
                plt.gca().invert_yaxis()
                plt.xlabel("Amplitude")
                plt.ylabel("Time (samples)")
                plt.title("Mean Trace and Fitted Curve")
                plt.legend()
                plt.grid(True)
                plt.show()

                gain_factor = 1.0 / fitted_curve
                gain_factor = gain_factor / gain_factor[0]  # Normalize
            except RuntimeError:
                print("Warning: Exponential fitting failed, using empirical gain")
                # Fallback to empirical approach
                fitted_curve = mean_trace
                gain_factor = np.max(mean_trace) / mean_trace
                gain_factor = np.clip(gain_factor, 1.0, 1000.0)  # Limit extreme values

        elif method == "power":
            # A*t^(-α)
            def damping_func(t, A, alpha):
                # Avoid division by zero
                t_safe = np.maximum(t, 0.1)
                return A * t_safe ** (-alpha)

            # Initial guess
            p0 = [np.max(mean_trace) * 10, 0.5]

            # Fit the function to mean trace (skip first few samples)
            start_idx = 5  # Skip first few samples
            try:
                popt, _ = curve_fit(
                    damping_func,
                    t[start_idx:],
                    mean_trace[start_idx:],
                    p0=p0,
                    bounds=([0, 0], [np.inf, np.inf]),
                )
                fitted_curve = np.zeros_like(mean_trace)
                fitted_curve[start_idx:] = damping_func(t[start_idx:], *popt)
                fitted_curve[:start_idx] = fitted_curve[
                    start_idx
                ]  # Fill in skipped values
                gain_factor = np.max(mean_trace) / fitted_curve
            except RuntimeError:
                print("Warning: Power law fitting failed, using empirical gain")
                # Fallback
                fitted_curve = mean_trace
                gain_factor = np.max(mean_trace) / mean_trace
                gain_factor = np.clip(gain_factor, 1.0, 1000.0)

        elif method == "combined":
            # A*t^(-1)*exp(-α*t)
            def damping_func(t, A, alpha):
                # Avoid division by zero
                t_safe = np.maximum(t, 0.1)
                return A * t_safe ** (-1) * np.exp(-alpha * t_safe)

            # Initial guess
            p0 = [np.max(mean_trace) * 10, 0.01]

            # Fit the function
            start_idx = 5  # Skip first few samples
            try:
                popt, _ = curve_fit(
                    damping_func,
                    t[start_idx:],
                    mean_trace[start_idx:],
                    p0=p0,
                    bounds=([0, 0], [np.inf, np.inf]),
                )
                fitted_curve = np.zeros_like(mean_trace)
                fitted_curve[start_idx:] = damping_func(t[start_idx:], *popt)
                fitted_curve[:start_idx] = fitted_curve[
                    start_idx
                ]  # Fill in skipped values
                gain_factor = np.max(mean_trace) / fitted_curve
            except RuntimeError:
                print("Warning: Combined fitting failed, using empirical gain")
                # Fallback
                fitted_curve = mean_trace
                gain_factor = np.max(mean_trace) / mean_trace
                gain_factor = np.clip(gain_factor, 1.0, 1000.0)

        elif method == "polynomial":
            degree = 3  # Default polynomial degree
            if params and "degree" in params:
                degree = params["degree"]

            # Fit polynomial
            poly_coeffs = np.polyfit(t, mean_trace, degree)
            fitted_curve = np.polyval(poly_coeffs, t)

            # Generate gain factor
            gain_factor = np.max(mean_trace) / fitted_curve
            # Limit extreme values
            gain_factor = np.clip(gain_factor, 1.0, 1000.0)

        elif method == "moving_average":
            window_length = 51  # Default
            if params and "window_length" in params:
                window_length = params["window_length"]

            fitted_curve = signal.savgol_filter(mean_trace, window_length, 2)
            gain_factor = np.max(mean_trace) / fitted_curve
            gain_factor = np.clip(gain_factor, 1.0, 1000.0)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply gain to data
        gained_data = data * gain_factor[:, np.newaxis]  # Broadcasting

        # Normalize output data to have similar amplitude range as input
        max_val = np.max(np.abs(data))
        gained_data = gained_data * (max_val / np.max(np.abs(gained_data)))

        # Visualize if requested
        if visualize:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Plot mean trace and fitted curve
            axes[0].plot(t, mean_trace, "b-", label="Mean Trace")
            axes[0].plot(t, fitted_curve, "r--", label="Fitted Curve")
            axes[0].set_title("Mean Trace and Fitted Curve")
            axes[0].set_xlabel("Sample")
            axes[0].set_ylabel("Amplitude")
            axes[0].legend()

            # Plot gain factor
            axes[1].plot(t, gain_factor)
            axes[1].set_title("Gain Factor")
            axes[1].set_xlabel("Sample")
            axes[1].set_ylabel("Gain")

            # Plot before/after comparison
            vmax = np.percentile(np.abs(gained_data), 99.5)
            axes[2].imshow(data, aspect="auto", cmap="gray", vmax=vmax)
            axes[2].set_title("Before")
            axes[2].set_xlabel("Trace")
            axes[2].set_ylabel("Time Sample")

            fig.tight_layout()

            # Second figure for gained data
            fig2, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(gained_data, aspect="auto", cmap="gray", vmax=vmax)
            ax.set_title("After Gain")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Time Sample")
            fig2.colorbar(im)
            plt.show()

        self.data = gained_data

    def _get_data(self, data):
        """Helper method to get the data array to use."""
        if data is None:
            if self.data is None:
                raise ValueError("No data provided and no data stored in object")
            return self.data
        return data

    def __getattr__(self, name):
        # Wird nur aufgerufen, wenn das Attribut nicht normal gefunden wird
        # Prüfe, ob es ein vorberechnetes Attribut gibt
        if self._precalc and name in self.__dict__:
            return self.__dict__[name]
        # Prüfe, ob es eine Berechnungsmethode gibt
        calc_method = f"calc_{name}"
        if hasattr(self, calc_method):
            return getattr(self, calc_method)()
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

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
    None
