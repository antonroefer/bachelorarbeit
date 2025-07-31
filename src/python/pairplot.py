import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np


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


# Eigene Daten laden
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = "feature_vectors_raw.npz"
data_path = os.path.join(script_dir, data_file)
with np.load(data_path) as npzfile:
    # Gib die Namen der Arrays in der .npz-Datei aus
    print("Arrays in der Datei:", npzfile.files)
    # Lade die Daten mit dem korrekten Schlüssel 'feature_stack'
    data = npzfile["feature_stack"]

    # --- NEU: Feature-Auswahl ---
    # Liste der gewünschten Features
    desired_features = [
        # "inst_amp",
        # "avg_energy",
        # "rms_amp",
        # "quadrature",
        # "inst_q",
        # "inst_phase_real",
        # "inst_phase_imag",
        # "inst_freq",
        # "semblance",
        "skewness",
        "kurtosis",
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

# Überführe die 3D-Daten (x, t, features) in ein 2D-Array (samples, features)
# Dabei werden x und t zu einer gemeinsamen "samples"-Achse zusammengefasst
num_x, num_t, num_features = data.shape
data_2d = data.reshape(num_x * num_t, num_features)
# Nehme nur jedes 10. Sample für schnellere Verarbeitung
data_2d = data_2d[::15]

# Erstelle ein DataFrame aus den gewünschten Features
df = pd.DataFrame(data_2d, columns=desired_features)

# Pairplot mit Dichte-Diagonalen
sns.pairplot(df, diag_kind="kde", plot_kws={"alpha": 0.3})
# plt.suptitle("Seaborn Density Plot der Radargramm-Features", y=1.02, fontsize=16)
plt.savefig("pairplot_stats.png", dpi=300)
