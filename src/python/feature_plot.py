from radargram import Radargram
import matplotlib.pyplot as plt
import numpy as np
import h5py

raw = False

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
rg = Radargram(data, 20, 20)
rg.apply_gain() if raw else None

# Plot original data
plt.figure(figsize=(10, 6))
plt.imshow(data, aspect="auto", cmap="jet")
plt.title("Original Radargram")
plt.colorbar(label="Amplitude")
plt.xlabel("Trace")
plt.ylabel("Sample")

# List of all attributes to calculate and plot, now with vmin and vmax
attributes = [
    # Attribute name, title, vmin, vmax
    ("instantaneous_amplitude", "Instantaneous Amplitude (Envelope)", None, None),
    ("instantaneous_phase_real", "Instantaneous Phase (Real Part)", None, None),
    ("instantaneous_phase_imag", "Instantaneous Phase (Imaginary Part)", None, None),
    ("instantaneous_frequency", "Instantaneous Frequency", 0.5, -0.5),
    ("quadrature", "Quadrature", None, None),
    ("instantaneous_q", "Instantaneous Q", 50, -50),
    ("fft", "FFT (Magnitude Spectrum)", None, None),
    ("absolute_gradient", "Absolute Gradient", None, None),
    ("average_energy", "Average Energy", None, None),
    ("rms_amplitude", "RMS Amplitude", None, None),
    ("coherence", "Coherence", 100, -100),
    # ("semblance", "Semblance", None, None),
    # ("entropy", "Entropy", None, None),
    ("mean", "Mean", None, None),
    # ("median", "Median", None, None),
    ("std", "Standard Deviation", None, None),
    ("skewness", "Skewness", None, None),
    ("kurtosis", "Kurtosis", 5, -5),
    ("max", "Maximum", None, None),
    ("min", "Minimum", None, None),
    ("range", "Range (Max-Min)", None, None),
]

# Plot each attribute in a separate figure
for attr_name, title, vmax, vmin in attributes:
    print(f"Processing {title}...")
    plt.figure(figsize=(10, 6))

    try:
        # Get attribute data using Radargram's __getattr__
        attr_data = getattr(rg, attr_name)
        label = "Value"

        # Plot the attribute
        plt.imshow(
            attr_data,
            aspect="auto",
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
        )
        plt.title(title)
        plt.get_current_fig_manager().set_window_title(title)
        plt.colorbar(label=label)
        plt.xlabel("Trace")
        plt.ylabel("Sample")
        plt.draw()
        plt.pause(0.01)  # Pause to allow the plot to update

    except Exception as e:
        print(f"Error processing {attr_name}: {e}")
        plt.title(title)
        plt.colorbar(label=label)
        plt.xlabel("Trace")
        plt.ylabel("Sample")
        plt.draw()
        plt.pause(0.01)  # Pause to allow the plot to update

    except Exception as e:
        print(f"Error processing {attr_name}: {e}")

input("Press Enter to close all figures ...")
plt.close("all")
