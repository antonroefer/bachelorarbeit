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
rg = Radargram(data)
rg.apply_gain() if raw else None

# Plot original data
plt.figure(figsize=(10, 6))
plt.imshow(rg.data, aspect="auto", cmap="jet")
plt.title("Original Radargram")
plt.colorbar(label="Amplitude")
plt.xlabel("Trace")
plt.ylabel("Sample")

# List of all attributes to calculate and plot
attributes = [
    # Attribute name, title, is_tuple, tuple_index
    ("instantaneous_amplitude", "Instantaneous Amplitude (Envelope)", False, None),
    ("instantaneous_phase", "Instantaneous Phase (Real Part)", True, 0),
    ("instantaneous_phase", "Instantaneous Phase (Imaginary Part)", True, 1),
    ("instantaneous_frequency", "Instantaneous Frequency", False, None),
    ("absolute_gradient", "Absolute Gradient", False, None),
    ("average_energy", "Average Energy", False, None),
    ("rms_amplitude", "RMS Amplitude", False, None),
    ("coherence", "Coherence", False, None),
    ("entropy", "Entropy", False, None),
    ("mean", "Mean", False, None),
    ("median", "Median", False, None),
    ("std", "Standard Deviation", False, None),
    ("skewness", "Skewness", False, None),
    ("kurtosis", "Kurtosis", False, None),
    ("max", "Maximum", False, None),
    ("min", "Minimum", False, None),
    ("range", "Range (Max-Min)", False, None),
]

plt.ion()

# Plot each attribute in a separate figure
for attr_name, title, is_tuple, tuple_idx in attributes:
    print(f"Processing {title}...")
    plt.figure(figsize=(10, 6))

    try:
        # Get attribute data
        if is_tuple:
            # For methods that return tuples (like instantaneous_phase)
            attr_data = getattr(rg, attr_name)()[tuple_idx]
            if tuple_idx == 0:
                label = "Real Part"
            else:
                label = "Imaginary Part"
        else:
            # For methods that return a single array
            attr_data = getattr(rg, attr_name)()
            label = "Value"

        # Plot the attribute
        plt.imshow(attr_data, aspect="auto", cmap="jet")
        plt.title(title)
        plt.colorbar(label=label)
        plt.xlabel("Trace")
        plt.ylabel("Sample")
        plt.draw()
        plt.pause(0.1)  # Pause to allow the plot to update

    except Exception as e:
        print(f"Error processing {attr_name}: {e}")

input("Press Enter to close all figures ...")
plt.close("all")
