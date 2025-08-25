import numpy as np
import matplotlib.pyplot as plt


# Funktion zur Berechnung der Normalverteilungs-Dichte
def normal_dist(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# X-Achsen-Werte erstellen
x = np.linspace(-5, 5, 1000)

# Parameter für die drei Kurven
mu = 0
# Mesokurtisch (Standard-Normalverteilung)
sigma_meso = 1.0
# Leptokurtisch (Spitze Kurve)
sigma_lepto = 0.5
# Platykurtisch (Flache Kurve)
sigma_platy = 2.0

# Dichtefunktionen berechnen
y_meso = normal_dist(x, mu, sigma_meso)
y_lepto = normal_dist(x, mu, sigma_lepto)
y_platy = normal_dist(x, mu, sigma_platy)

# Plotten der Kurven
plt.figure(figsize=(10, 6))
plt.plot(x, y_meso, label="Mesokurtisch (Kurtosis = 3)", linewidth=2)
plt.plot(x, y_lepto, label="Leptokurtisch (Kurtosis > 3)", linewidth=2)
plt.plot(x, y_platy, label="Platykurtisch (Kurtosis < 3)", linewidth=2)

# Beschriftungen und Titel hinzufügen
plt.title("Visualisierung der Wölbung (Kurtosis)")
plt.xlabel("X-Werte")
plt.ylabel("Y-Werte")
plt.legend()
plt.grid(True)

# Plot anzeigen
plt.show()
