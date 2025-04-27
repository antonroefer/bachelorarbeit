import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    norm,
    cauchy,
    expon,
    gamma,
    rayleigh,
    weibull_min,
    lognorm,
    laplace,
)

# x-Achse
x = np.linspace(-5, 10, 1000)

# Verteilungen
pdf_norm = norm.pdf(x, loc=0, scale=1)
pdf_cauchy = cauchy.pdf(x, loc=0, scale=1)
pdf_expon = expon.pdf(x, loc=0, scale=1)
pdf_gamma = gamma.pdf(x, a=2, loc=0, scale=1)
pdf_rayleigh = rayleigh.pdf(x, loc=0, scale=1)
pdf_weibull = weibull_min.pdf(x, c=1.5, loc=0, scale=1)
pdf_lognorm = lognorm.pdf(x, s=0.8, loc=0, scale=1)
pdf_laplace = laplace.pdf(x, loc=0, scale=1)

# Plot
plt.figure(figsize=(12, 7))

plt.plot(x, pdf_norm, label="Normal (Gauß)", linewidth=2)
plt.plot(x, pdf_cauchy, label="Cauchy", linewidth=2)
plt.plot(x, pdf_expon, label="Exponential", linewidth=2)
plt.plot(x, pdf_gamma, label="Gamma (a=2)", linewidth=2)
plt.plot(x, pdf_rayleigh, label="Rayleigh", linewidth=2)
plt.plot(x, pdf_weibull, label="Weibull (c=1.5)", linewidth=2)
plt.plot(x, pdf_lognorm, label="Log-Normal (s=0.8)", linewidth=2)
plt.plot(x, pdf_laplace, label="Laplace", linewidth=2)

plt.title("Vergleich gängiger Wahrscheinlichkeitsdichte-Funktionen", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("Dichte", fontsize=12)
plt.grid(True)
plt.legend()
plt.ylim(0, 1.5)
plt.xlim(-5, 10)
plt.show()
