import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os
import h5py

i_rg = 6  # Radargram number

cwd_path = os.getcwd()
folder = "radargram_plots"
base_data_dir = os.path.join(cwd_path, "..", "..", "data", "processed")
x_path = os.path.join(base_data_dir, "x.mat")
t_path = os.path.join(base_data_dir, "t.mat")


# First, explore the structure of the file
with h5py.File(x_path, "r") as f:
    print("Top-level keys:", list(f.keys()))

    # Explore first level of structure
    for key in f.keys():
        if isinstance(f[key], h5py.Group):
            print(f"{key} (Group): {list(f[key].keys())}")
        else:
            print(f"{key} (Dataset): shape={f[key].shape}, dtype={f[key].dtype}")

    # Load x from the first available key
    first_key = list(f.keys())[0]  # second key

    if isinstance(f[first_key], h5py.Group):
        # If it's a group, look for a dataset inside
        nested_keys = list(f[first_key].keys())
        if nested_keys:
            x_path = f"{first_key}/{nested_keys[i_rg]}"
            print(f"Loading data from: {x_path}")
            x = np.array(f[x_path][:]).T  # Transpose to match MATLAB's orientation
    else:
        # If it's directly a dataset
        x = np.array(f[first_key][:]).T
        print(f"Loading x from: {first_key}")

    # Print data shape
    x = x.squeeze()  # Ensure x is a 1D array
    print(f"X shape: {x.shape}")

# First, explore the structure of the file
with h5py.File(t_path, "r") as f:
    print("Top-level keys:", list(f.keys()))

    # Explore first level of structure
    for key in f.keys():
        if isinstance(f[key], h5py.Group):
            print(f"{key} (Group): {list(f[key].keys())}")
        else:
            print(f"{key} (Dataset): shape={f[key].shape}, dtype={f[key].dtype}")

    # Load x from the first available key
    first_key = list(f.keys())[0]  # second key

    if isinstance(f[first_key], h5py.Group):
        # If it's a group, look for a dataset inside
        nested_keys = list(f[first_key].keys())
        if nested_keys:
            t_path = f"{first_key}/{nested_keys[i_rg]}"
            print(f"Loading data from: {t_path}")
            t = np.array(f[t_path][:]).T  # Transpose to match MATLAB's orientation
    else:
        # If it's directly a dataset
        t = np.array(f[first_key][:]).T
        print(f"Loading t from: {first_key}")

    # Print data shape
    t = t.squeeze()  # Ensure t is a 1D array

fig_path = os.path.join(
    cwd_path, "..", "..", "..", "GPR_Daten_mat", "Figures", "georef"
)
fig_file = f"Radargram_{int(i_rg + 1)}.png"
image = img.imread(os.path.join(fig_path, fig_file))

cut = True

# Finde den Index, bei dem x zum ersten Mal größer als 60 ist
first_index_above_60 = np.argmax(x > 60)
# Erstelle die geschnittenen Arrays
cut_x = x[:first_index_above_60]
cut_image = image[:, :first_index_above_60]
# Überschreibe die Originaldaten mit den geschnittenen Daten
x = cut_x if cut else x
image = cut_image if cut else image

fig = plt.figure(figsize=(15, 6))
im = plt.imshow(
    image, cmap="gray_r", extent=[x.min(), x.max(), t.max(), t.min()], aspect="auto"
)
plt.xlabel("Distance (m)", fontsize=18)
plt.ylabel("Time (ns)", fontsize=18)

cbar = fig.colorbar(im, fraction=0.046, pad=0.04)

# Ticks ausblenden und eigene Labels setzen
cbar.set_ticks([im.get_clim()[0], im.get_clim()[1]])
cbar.set_ticklabels(["negativ", "positiv"])
cbar.set_label("Amplitude", fontsize=18)
cbar.ax.tick_params(labelsize=14)
os.makedirs(folder, exist_ok=True)
plt.savefig(
    os.path.join(
        cwd_path, folder, f"Radargram_{int(i_rg + 1)}{'_cut' if cut else ''}.png"
    ),
    dpi=300,
)
plt.show()
