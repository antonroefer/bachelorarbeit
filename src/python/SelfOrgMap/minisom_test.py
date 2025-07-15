from minisom import MiniSom  # Importiere deine aktualisierte Klasse
from sklearn.datasets import load_iris
import numpy as np
import os
import h5py
from pycolormap_2d import ColorMap2DZiegler, ColorMap2DTeuling2, ColorMap2DBremm


def min_max_scale(arr, new_min=0, new_max=1):
    """
    Skaliert ein NumPy-Array auf einen neuen Wertebereich (new_min, new_max).

    Args:
        arr (ndarray): Das Eingangs-Array.
        new_min (float): Der gewünschte minimale Wert des neuen Bereichs.
        new_max (float): Der gewünschte maximale Wert des neuen Bereichs.

    Returns:
        ndarray: Das skalierte Array.
    """
    # Finde den originalen Minimal- und Maximalwert des Arrays
    original_min = arr.min()
    original_max = arr.max()

    # Vermeide Division durch Null, falls alle Werte im Array gleich sind
    if original_max == original_min:
        # Wenn alle Werte gleich sind, sind sie im neuen Bereich einfach der Mittelwert
        return np.full_like(arr, (new_min + new_max) / 2)

    # Führe die Min-Max-Skalierung durch
    scaled_arr = ((arr - original_min) / (original_max - original_min)) * (
        new_max - new_min
    ) + new_min
    return scaled_arr


raw = False  # Use raw data or processed data
i_rg = 6  # Radargram number

# Load MAT file in v7.3 format using h5py
script_dir = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(script_dir, "..", "..", "..", "data")
data_type = "raw" if raw else "processed"

file_path = os.path.join(base_data_dir, data_type, "radargrams.mat")
x_path = os.path.join(base_data_dir, "raw" if raw else "processed", "x.mat")
t_path = os.path.join(base_data_dir, "raw" if raw else "processed", "t.mat")

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
            data_path = f"{first_key}/{nested_keys[i_rg]}"
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
    original_data = data.copy()  # Save original data for later use

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
    print(f"T shape: {t.shape}")

ratio = x.max() / t.max() * (9 / 16)


# Eigene Daten laden
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = "feature_vectors.npz"
data_path = os.path.join(script_dir, data_file)
with np.load(data_path) as npzfile:
    # Gib die Namen der Arrays in der .npz-Datei aus
    print("Arrays in der Datei:", npzfile.files)
    # Lade die Daten mit dem korrekten Schlüssel 'feature_stack'
    data = npzfile["feature_stack"]

    # --- NEU: Feature-Auswahl ---
    # Liste der gewünschten Features
    desired_features = [
        "inst_amp",
        "inst_freq",
        "semblance",
        "kurtosis",
        "inst_phase_real",
        "inst_phase_imag",
        "inst_q",
    ]

    # Annahme: Die Namen der Features sind in der .npz-Datei unter dem Schlüssel 'feature_names' gespeichert
    if "feature_names" in npzfile.files:
        all_feature_names = list(npzfile["feature_names"])
        print("Verfügbare Features:", all_feature_names)

        # Finde die Indizes der gewünschten Features
        try:
            indices_to_keep = [
                all_feature_names.index(name) for name in desired_features
            ]
            print(f"Indizes der ausgewählten Features: {indices_to_keep}")

            # Wähle nur die gewünschten Features aus den Daten aus
            # Die Features sind die letzte Dimension im 3D-Array
            if data.ndim == 3:
                data = data[:, :, indices_to_keep]
                print(f"Form der Daten nach Feature-Auswahl: {data.shape}")
            else:
                print(
                    "Warnung: Daten sind nicht 3D, Feature-Auswahl wird übersprungen."
                )

        except ValueError as e:
            print(
                f"Fehler bei der Feature-Auswahl: {e}. Stelle sicher, dass alle gewünschten Features in 'feature_names' vorhanden sind."
            )
            # Beende das Skript oder fahre mit allen Features fort
            exit()
    else:
        print(
            "Warnung: 'feature_names' nicht in .npz-Datei gefunden. Feature-Auswahl nicht möglich."
        )

# Finde den Index, bei dem x zum ersten Mal größer als 60 ist
first_index_above_60 = np.argmax(x > 60)
# Erstelle die geschnittenen Arrays
cut_x = x[:first_index_above_60]
cut_data = data[:, :first_index_above_60]
# Überschreibe die Originaldaten mit den geschnittenen Daten
x = cut_x
data = cut_data
# Gib die neuen Formen aus
print(f"Neue X shape: {x.shape}")
print(f"Neue Data shape: {data.shape}")

# --- NEU: Daten umformen und normalisieren ---
# Forme das 3D-Array (höhe, breite, merkmale) in ein 2D-Array (punkte, merkmale) um
if data.ndim == 3:
    print(f"Originale Datenform: {data.shape}")
    data = data.reshape(-1, data.shape[2])
    print(f"Neue Datenform für SOM: {data.shape}")

num_features = data.shape[1]
print(f"Anzahl der erkannten Merkmale: {num_features}")

# SOM initialisieren
# Für ein 8x10 Gitter wie im MATLAB-Beispiel
apx = "10"

som = MiniSom(
    x=30,
    y=30,
    input_len=num_features,
    sigma=5,
    learning_rate=0.5,
    topology="hexagonal",
    sigma_decay_function="inverse_decay_to_one",
    random_seed=42,
)

num_epochs = 10

# Normalisiere jede Spalte mit einer For-Schleife und dem MiniSom min_max_scaler
for i in range(data.shape[1]):
    data[:, i] = min_max_scale(data[:, i])

# Gewichte initialisieren (z.B. zufällig oder PCA)
som.normalize_random_weights_init(data)

# SOM trainieren
print("Starte Training...")
# Setze eine realistische Anzahl von Epochen (z.B. 1 oder 2, nicht 10.000)
som.train(
    data, num_iteration=num_epochs, random_order=True, use_epochs=True, verbose=True
)
print("Training beendet.")

# Speichere das trainierte Modell
model_name = f"trained_som_{apx}.pkl"
model_path = os.path.join(script_dir, model_name)
som.save_model(model_path)

# Den Plot erstellen
save_plots = True  # Setze auf True, um die Plots zu speichern
cmap = ColorMap2DZiegler

som.plot_u_matrix(save=save_plots, appendix=apx)  # U-Matrix Plot
som.plot_som_neighbor_distances(
    cmap="hot", figsize=(10, 8), save=save_plots, appendix=apx
)  # cmap='hot' ist gut für Distanzen
som.plot_som_hits(data, save=save_plots, appendix=apx, colormap=cmap)  # SOM Hits Plot
som.plot_som_planes(save=save_plots, appendix=apx)
som.plot_bmu_radargram(
    data=data, x=x, t=t, save=save_plots, appendix=apx, cmap_2d_class=cmap
)
