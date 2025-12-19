import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------
# MODELLO
# ---------------------------
def malus(x, A, theta0):
    return A * np.cos(np.pi/180 * (x - theta0))**2

# ---------------------------
# LETTURA DATI
# ---------------------------
df = pd.read_table("dati_corretti_gamma.txt", delim_whitespace=True, names=["x", "y", "sigma"])
x = df["x"].values
y = df["y"].values
sigma = df["sigma"].values
N = len(x)

# ---------------------------
# FIT MLE + MATRICE DI COVARIANZA
# ---------------------------
p0 = [y.max(), 0]
pars, cov = curve_fit(malus, x, y, sigma=sigma, p0=p0, absolute_sigma=True)

A_fit, theta_fit = pars
sigma_A, sigma_theta = np.sqrt(np.diag(cov))

print("\n===== RISULTATI FIT (MLE) =====")
print(f"A       = {A_fit:.6g} ± {sigma_A:.6g}")
print(f"theta0  = {theta_fit:.6g}° ± {sigma_theta:.6g}")

# ---------------------------
# CHI QUADRO
# ---------------------------
y_teo = malus(x, A_fit, theta_fit)

chi2 = np.sum(((y - y_teo) / sigma)**2)
chi2_red = chi2 / (N - 2)

print("\n===== TEST DEL CHI-QUADRO =====")
print(f"Chi²        = {chi2:.4f}")
print(f"Chi² ridotto = {chi2_red:.4f}")

# ---------------------------
# TABELLA
# ---------------------------
print("\n===== TABELLA DATI =====")
print("x_deg\t y_corr\t\t y_teo\t\t residuo")
for i in range(N):
    print(f"{x[i]:.0f}\t {y[i]:.10g}\t {y_teo[i]:.10g}\t {y[i] - y_teo[i]:.10g}")

print("\nParametri con precisione massima:")
print("A_fit =", repr(A_fit))
print("theta_fit =", repr(theta_fit))

# ---------------------------
# PLOT FINALE
# - curva centrale (fit)
# - curva minore   (A - σA, θ - σθ)
# ---------------------------
x_plot = np.linspace(min(x), max(x), 500)

y_fit = malus(x_plot, A_fit, theta_fit)
y_low = malus(x_plot, A_fit - sigma_A, theta_fit - sigma_theta)

plt.figure(figsize=(9,6))

# dati sperimentali
plt.errorbar(x, y, sigma, fmt='o', label="Dati", zorder=3)

# curva del fit
plt.plot(x_plot, y_fit, 'r-', lw=2, label="Fit MLE")

# curva con parametri minori
plt.plot(x_plot, y_low, 'g--', lw=2, label="Fit (A-σA, θ-σθ)")

plt.xlabel("Angolo (°)")
plt.ylabel("Intensità")
plt.title("Fit della legge di Malus (con matrice di covarianza)")
plt.grid(True)
plt.legend()
plt.show()
