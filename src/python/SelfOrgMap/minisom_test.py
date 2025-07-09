from minisom import MiniSom  # Importiere deine aktualisierte Klasse
from sklearn.datasets import load_iris
import numpy as np
import os

iris = load_iris()
data = iris.data
# Daten normalisieren (spaltenweise)
data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

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
    desired_features = ["inst_amp", "inst_freq_raw", "semblance", "kurtosis", "dip"]

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

# --- NEU: Daten umformen und normalisieren ---
# Forme das 3D-Array (höhe, breite, merkmale) in ein 2D-Array (punkte, merkmale) um
if data.ndim == 3:
    print(f"Originale Datenform: {data.shape}")
    data = data.reshape(-1, data.shape[2])
    print(f"Neue Datenform für SOM: {data.shape}")

# Entferne Zeilen mit NaN-Werten, die beim Normalisieren Probleme verursachen könnten
data = data[~np.isnan(data).any(axis=1)]

# Daten normalisieren (spaltenweise)
data_min = data.min(axis=0)
data_max = data.max(axis=0)
data = (data - data_min) / (data_max - data_min)

num_features = data.shape[1]
print(f"Anzahl der erkannten Merkmale: {num_features}")

# SOM initialisieren
# Für ein 8x10 Gitter wie im MATLAB-Beispiel
som = MiniSom(
    x=12,
    y=12,
    input_len=num_features,  # Verwende die tatsächliche Anzahl der Merkmale
    sigma=0.6,
    learning_rate=0.9,
    topology="hexagonal",
    random_seed=42,
)

# Gewichte initialisieren (z.B. zufällig oder PCA)
som.random_weights_init(data)

# SOM trainieren
print("Starte Training...")
# Setze eine realistische Anzahl von Epochen (z.B. 1 oder 2, nicht 10.000)
num_epochs = 50
som.train(
    data, num_iteration=num_epochs, random_order=True, use_epochs=True, verbose=True
)
print("Training beendet.")

# Den Plot erstellen
save_plots = True  # Setze auf True, um die Plots zu speichern

som.plot_u_matrix(save=save_plots)  # U-Matrix Plot
som.plot_som_neighbor_distances(
    cmap="hot", figsize=(10, 8), save=save_plots
)  # cmap='hot' ist gut für Distanzen
som.plot_som_hits(data, save=save_plots)
som.plot_som_planes(save=save_plots)
