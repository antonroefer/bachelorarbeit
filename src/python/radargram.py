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
from scipy.signal import morlet2, ricker, cwt, argrelextrema
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        self.dip = None
        self.azimuth = None

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
            print("Calculating Quadrature ...")
            self.quadrature = self.calc_quadrature()
            print("Calculating Instantaneous Q ...")
            self.instantaneous_q = self.calc_instantaneous_q()
            self.sweetness = self.calc_sweetness()
            print("Calculating FFT ...")
            self.fft = self.calc_fft()
            print("Calculating Absolute Gradient ...")
            self.absolute_gradient = self.calc_absolute_gradient()
            print("Calculating Average Energy ...")
            self.average_energy = self.calc_average_energy(x_dis=x_dis, y_dis=y_dis)
            print("Calculating RMS Amplitude ...")
            self.rms_amplitude = self.calc_rms_amplitude(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Coherence ...")
            self.coherence = self.calc_coherence(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Entropy ...")
            # self.entropy = self.calc_entropy(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Semblance ...")
            # self.semblance = self.calc_semblance(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Mean ...")
            self.mean = self.calc_mean(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Mean of Squared Values ...")
            self.mean_sq = self.calc_mean_sq(x_dis=x_dis, y_dis=y_dis)
            print("Calculating Median ...")
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
            print("Calculating Dip ...")
            self.dip_real, self.dip_imag = self.calc_dip()
            print("Calculating Azimuth ...")
            self.azimuth = self.calc_azimuth()
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

    def calc_quadrature(self, data=None):
        """
        Calculate the quadrature component (imaginary part of analytic signal) using Hilbert transform.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data

        Returns:
        --------
        ndarray
            2D array of quadrature (imaginary) values
        """
        data = self._get_data(data)
        quadrature = np.zeros_like(data, dtype=float)

        # Process each trace (column) separately
        for i in range(data.shape[1]):
            trace = data[:, i]
            analytic_signal = signal.hilbert(trace)
            quadrature[:, i] = np.imag(analytic_signal)

        return quadrature

    def calc_instantaneous_q(self, data=None, dt=0.1173):
        """
        Calculate the instantaneous Q factor (quality factor) using amplitude and frequency.

        Q = (pi * instantaneous frequency * envelope) / |d(envelope)/dt|

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        dt : float, optional
            Time sampling interval, defaults to 0.1173

        Returns:
        --------
        ndarray
            2D array of instantaneous Q values
        """
        if (
            self._precalc
            and self.instantaneous_amplitude is not None
            and self.instantaneous_frequency is not None
        ):
            envelope = self.instantaneous_amplitude
            freq = self.instantaneous_frequency
        else:
            envelope = self.calc_instantaneous_amplitude(data)
            freq = self.calc_instantaneous_frequency(data, dt=dt)

        # Compute derivative of envelope along time axis (axis=0)
        d_env = np.gradient(envelope, dt, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            Q = (np.pi * freq * envelope) / np.abs(d_env)
            Q = np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
        return Q

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

    def calc_fft(self, data=None, shift=False, log_scale=True):
        """
        Calculate the 2D FFT (Fast Fourier Transform) of the radargram.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        shift : bool, optional
            If True, shift the zero-frequency component to the center
        log_scale : bool, optional
            If True, return the log-magnitude spectrum

        Returns:
        --------
        ndarray
            2D FFT magnitude (optionally log-scaled)
        """
        data = self._get_data(data)
        fft2 = np.fft.fft2(data)
        if shift:
            fft2 = np.fft.fftshift(fft2)
        magnitude = np.abs(fft2)
        if log_scale:
            magnitude = np.log1p(magnitude)
        return magnitude

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

        # Use precomputed mean and std if available
        if self._precalc and self.mean is not None and self.std is not None:
            local_mean = self.mean
            local_std = self.std
        else:
            local_mean = uniform_filter(data, size=window_size, mode=mode)
            local_mean_sq = uniform_filter(data**2, size=window_size, mode=mode)
            local_std = np.sqrt(local_mean_sq - local_mean**2)

        # Calculate coherence
        with np.errstate(divide="ignore", invalid="ignore"):
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
        def entropy_func(window):
            hist, _ = np.histogram(window, bins=bins, density=True)
            nonzero = hist > 0
            if np.any(nonzero):
                return -np.sum(hist[nonzero] * np.log2(hist[nonzero]))
            return 0

        # Apply the filter
        return generic_filter(data, entropy_func, size=window_size, mode=mode)

    def calc_semblance(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the semblance attribute within a window.

        Semblance is a measure of similarity of traces in a window, often used in seismic processing.

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
            2D array of semblance values
        """
        data = self._get_data(data)
        window_shape = (2 * y_dis + 1, 2 * x_dis + 1)
        n_samples = window_shape[0] * window_shape[1]

        def semblance_func(window):
            # window is flattened
            if np.all(window == 0):
                return 0.0
            sum_traces = np.sum(window)
            sum_traces_sq = sum_traces**2
            sum_sq_traces = np.sum(window**2)
            if sum_sq_traces == 0:
                return 0.0
            return sum_traces_sq / (n_samples * sum_sq_traces)

        return generic_filter(data, semblance_func, size=window_shape, mode=mode)

    def calc_correlation(self, data=None, x_dis=1, y_dis=1, mode="mirror"):
        """
        Calculate the local correlation coefficient between each trace and the mean trace in a window.

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
            2D array of local correlation coefficients
        """
        data = self._get_data(data)
        window_shape = (2 * y_dis + 1, 2 * x_dis + 1)

        def corr_func(window):
            window = window.reshape(window_shape)
            # Correlate center trace with mean of window
            center_idx = window_shape[1] // 2
            center_trace = window[:, center_idx]
            mean_trace = np.mean(window, axis=1)
            # Remove mean
            ct = center_trace - np.mean(center_trace)
            mt = mean_trace - np.mean(mean_trace)
            denom = np.linalg.norm(ct) * np.linalg.norm(mt)
            if denom == 0:
                return 0.0
            return np.dot(ct, mt) / denom

        return generic_filter(data, corr_func, size=window_shape, mode=mode)

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

    def calc_dip(self, data=None):
        """
        Calculate the local dip (in radians) within a window using the Sobel operator.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data

        Returns:
        --------
        ndarray
            2D array of dip angles (in radians)
        """
        data = self._get_data(data)
        # Compute gradients
        grad_y = sobel(data, axis=0)
        grad_x = sobel(data, axis=1)
        # Dip is arctangent of vertical over horizontal gradient
        dip = np.arctan2(grad_y, grad_x)
        return np.cos(dip), np.sin(dip)

    def calc_azimuth(self, data=None):
        """
        Calculate the local azimuth (in degrees) within a window using the Sobel operator.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data

        Returns:
        --------
        ndarray
            2D array of azimuth angles (in degrees, 0 = x-direction, 90 = y-direction)
        """
        data = self._get_data(data)
        grad_y = sobel(data, axis=0)
        grad_x = sobel(data, axis=1)
        # Azimuth is angle in the x-y plane, measured from x-axis
        azimuth = np.degrees(np.arctan2(grad_y, grad_x))
        # Normalize to [0, 180)
        azimuth = np.mod(azimuth, 180)
        return azimuth

    def calc_cwt(
        self, data=None, widths=[5, 10, 20, 35, 55, 80], wavelet="ricker", axis=0
    ):
        """
        Calculate the Continuous Wavelet Transform (CWT) of the radargram.

        Parameters:
        -----------
        data : ndarray, optional
            2D radar amplitude data. If None, uses self.data
        widths : array-like, optional
            Widths (scales) to use for the CWT. If None, uses np.arange(1, 32)
        wavelet : str or callable, optional
            Wavelet function to use ('morlet', 'ricker', or a custom function). Defaults to 'ricker'
        axis : int, optional
            Axis along which to apply the CWT (0: along traces, 1: along samples). Defaults to 0

        Returns:
        --------
        ndarray
            3D array of CWT coefficients with shape (n_scales, n_samples, n_traces)
        """

        data = self._get_data(data)

        if wavelet == "morlet":
            # Use Morlet wavelet with default omega0=6
            def wavelet_func(M, s):
                return morlet2(M, s, w=6)
        elif wavelet == "ricker":
            wavelet_func = ricker
        elif callable(wavelet):
            wavelet_func = wavelet
        else:
            raise ValueError("Unknown wavelet type: {}".format(wavelet))

        # Apply CWT along the specified axis for each trace/sample
        if axis == 0:
            # Along each trace (column)
            n_traces = data.shape[1]
            n_samples = data.shape[0]
            cwt_coeffs = np.zeros((len(widths), n_samples, n_traces), dtype=complex)
            for i in range(n_traces):
                cwt_coeffs[:, :, i] = cwt(data[:, i], wavelet_func, widths)
        else:
            # Along each sample (row)
            n_samples = data.shape[0]
            n_traces = data.shape[1]
            cwt_coeffs = np.zeros((len(widths), n_samples, n_traces), dtype=complex)
            for i in range(n_samples):
                cwt_coeffs[:, i, :] = cwt(data[i, :], wavelet_func, widths)

        return cwt_coeffs

    # ---- Helper methods ----

    def apply_gain(
        self,
        data=None,
        method="exponential",
        params=None,
        window_size=None,
        visualize=True,
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
        # Set the first five samples of each trace to 0
        data[:5, :] = 0

        # Apply bandpass filter before gain correction

        # Sampling interval in nanoseconds, convert to seconds
        dt = 0.1173e-9  # seconds
        fs = 1.0 / dt  # Hz

        # Bandpass filter design: 100 MHz to 800 MHz
        lowcut = 1e6
        highcut = 800e6
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        b, a = butter(N=4, Wn=[low, high], btype="band")

        # Apply filter to each trace (column)
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = filtfilt(b, a, data[:, i])

        mean_trace = np.mean(filtered_data, axis=1)

        # Plot the Fourier Transform of the mean trace
        mean_trace_fft = np.fft.fft(mean_trace)
        freqs = np.fft.fftfreq(mean_trace.size, d=dt)
        plt.figure(figsize=(10, 5))
        plt.loglog(
            freqs[: mean_trace.size // 2] / 1e6,
            np.abs(mean_trace_fft[: mean_trace.size // 2]),
        )
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude")
        plt.title("Fourier Transform of Mean Trace")
        plt.grid(True, which="both", linestyle="--")
        plt.tight_layout()
        plt.show()

        # Find local maxima of the mean trace
        local_max_indices = argrelextrema(mean_trace, np.greater)[0]
        # Only keep maxima above zero
        mask = (mean_trace[local_max_indices] > -1) & (local_max_indices > 50)
        local_max_indices = local_max_indices[mask]
        local_max_values = mean_trace[local_max_indices]

        # Fit an exponential function to the local maxima using curve_fit
        def exp_func(x, alpha):
            return local_max_values[0] * np.exp(-alpha * (x - local_max_indices[0]))

        x_data = local_max_indices
        y_data = local_max_values

        try:
            popt, _ = curve_fit(exp_func, x_data, y_data, p0=[1e-2], maxfev=10000)
            print(f"Fitted parameters: {popt}")
            fitted_curve = exp_func(np.arange(mean_trace.size), *popt)

        except Exception as e:
            print(f"Exponential fit failed: {e}")
            fitted_curve = np.ones_like(mean_trace)

        # Apply gain correction based on the selected method
        # Only apply gain correction for indices >= local_max_indices[0]
        gain = np.ones_like(mean_trace)
        start_idx = local_max_indices[0]
        gain[start_idx:] = local_max_values[0] / fitted_curve[start_idx:]
        np.clip(gain, None, local_max_values[0], out=gain)  # Limit gain to max value

        # Apply gain to each trace (column)

        corrected_data = filtered_data * gain[:, np.newaxis]

        # Calculate the envelope (instantaneous amplitude) of the filtered data
        envelope_filtered = self.calc_instantaneous_amplitude(filtered_data)

        # Plot comparison between mean trace of original and filtered data
        plt.figure(figsize=(10, 5))
        plt.plot(np.mean(data, axis=1), label="Original Mean Trace", color="blue")
        plt.plot(
            np.mean(filtered_data, axis=1), label="Filtered Mean Trace", color="orange"
        )
        plt.plot(
            envelope_filtered.mean(axis=1),
            label="Envelope of Filtered Data",
            color="red",
            linestyle="--",
        )
        plt.xlabel("Sample Index")
        plt.ylabel("Mean Amplitude")
        plt.title("Comparison of Mean Trace: Original vs. Filtered Data")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot mean trace and local maxima
        if visualize:
            plt.figure(figsize=(10, 5))
            plt.plot(mean_trace, label="Mean Trace")
            plt.plot(
                gain,
                label="Gain Function",
                linestyle="--",
                color="purple",
            )
            plt.plot(
                fitted_curve,
                label="Fitted Exponential Function",
                linestyle="--",
                color="orange",
            )
            plt.plot(
                np.mean(corrected_data, axis=1),
                label="Corrected Mean Trace",
                color="green",
            )
            plt.scatter(
                local_max_indices, local_max_values, color="red", label="Local Maxima"
            )
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.title("Mean Trace with Local Maxima")
            plt.grid(True)
            plt.legend()
            plt.show()

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


# Example usage:
if __name__ == "__main__":
    import h5py
    from scipy.signal import butter, filtfilt

    raw = True  # Set to True if using raw data

    # Load MAT file in v7.3 format using h5py
    file_path = (
        "./../../data/raw/radargrams.mat"
        if raw
        else "./../../../GPR_Daten_mat/radargrams.mat"
    )

    # First, explore the structure of the file
    with h5py.File(file_path, "r") as f:
        print("Top-level keys:", list(f.keys()))

        # Explore first level of structure
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                print(f"{key} (Group): {list(f[key].keys())}")
            else:
                print(f"{key} (Dataset): shape={f[key].shape}, dtype={f[key].dtype}")

        # Load data from the first available key
        first_key = list(f.keys())[0]  # second key

        if isinstance(f[first_key], h5py.Group):
            # If it's a group, look for a dataset inside
            nested_keys = list(f[first_key].keys())
            if nested_keys:
                data_path = f"{first_key}/{nested_keys[30]}"
                print(f"Loading data from: {data_path}")
                data = np.array(
                    f[data_path][:]
                ).T  # Transpose to match MATLAB's orientation
        else:
            # If it's directly a dataset
            data = np.array(f[first_key][:]).T
            print(f"Loading data from: {first_key}")

        # Print data shape
        print(f"Data shape: {data.shape}")

    # Create Radargram instance
    rg = Radargram(data)

    rg.apply_gain()
