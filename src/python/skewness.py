import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# X-Achsen-Werte auf einer linearen Skala erstellen
x = np.linspace(0, 5, 500)

# Parameter für die drei Kurven
# Der 's'-Parameter steuert die Schiefe: größere s-Werte führen zu stärkerer Rechtsschiefe
s_low = 0.5  # Geringe Schiefe

# Dichtefunktionen mit der lognorm-Funktion von SciPy berechnen
# Der 'scale'-Parameter verschiebt die Kurve auf der X-Achse
y_left = lognorm.pdf(x, s_low, scale=1.0)  #
y_right = y_left[::-1]  # y_left nur in der Reihenfolge umgekehrt
y = 2 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - 2.5) ** 2) ** 2

x = np.linspace(-2.5, 2.5, 500)

# Plotten der Kurven
plt.figure(figsize=(10, 7))
plt.plot(x, y_left, label="linksschief", linewidth=2)
plt.plot(x, y, label="gerade", linewidth=2)
plt.plot(x, y_right, label="rechtsschief", linewidth=2)

# Beschriftungen und Titel hinzufügen
plt.title("Visualisierung der Schiefe (Skewness)")
plt.xlabel("X-Werte")
plt.ylabel("Y-Werte")
plt.legend()
plt.grid(True)

# Plot anzeigen
plt.show()
