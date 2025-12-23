import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})
# ------------------------------------------------------
# MODELLO COS^n (versione stabile con |cos|^n)
# ------------------------------------------------------
def malus_n(x, A, theta0, n):
    return A * np.abs(np.cos(np.pi/180 * (x - theta0)))**n

# Modello cos^2 per un primo guess
def malus2(x, A, theta0):
    return A * np.cos(np.pi/180 * (x - theta0))**2


# ------------------------------------------------------
# LETTURA DATI (spazi)
# ------------------------------------------------------
df = pd.read_table("dati.txt", 
                   delim_whitespace=True,
                   header=None,
                   names=["x", "y", "sigma"])

x = df["x"].values
y = df["y"].values
sigma = df["sigma"].values

# ------------------------------------------------------
# FIT INIZIALE (cos^2)
# ------------------------------------------------------
p0_simple, _ = curve_fit(
    malus2, x, y, sigma=sigma,
    p0=[y.max(), 0],
    absolute_sigma=True
)

A_guess, theta_guess = p0_simple

# ------------------------------------------------------
# FIT COMPLETO: A, theta0, n
#  - bounds: A ≥ 0, n ≥ 0
# ------------------------------------------------------
p0 = [A_guess, theta_guess, 2]  # starting values

pars, cov = curve_fit(
    malus_n,
    x,
    y,
    sigma=sigma,
    p0=p0,
    bounds=([0, -np.inf, 0], [np.inf, np.inf, 20]),  # n max 20
    absolute_sigma=True,
    maxfev=50000
)

A_fit, theta_fit, n_fit = pars
sigma_A, sigma_theta, sigma_n = np.sqrt(np.diag(cov))

# Stima gamma
gamma_est = n_fit / 2
gamma_err = sigma_n / 2

# ------------------------------------------------------
# RISULTATI
# ------------------------------------------------------
print("\n=======================")
print("   RISULTATI FIT cos^n")
print("=======================")
print(f"A       = {A_fit:.6g} ± {sigma_A:.3g}")
print(f"theta0  = {theta_fit:.6g}° ± {sigma_theta:.3g}")
print(f"n       = {n_fit:.6g} ± {sigma_n:.3g}")
print("-----------------------")
print(f"STIMA GAMMA = n/2 = {gamma_est:.6g} ± {gamma_err:.3g}")
print("=======================\n")

# ------------------------------------------------------
# PLOT
# ------------------------------------------------------
x_fit = np.linspace(min(x), max(x), 500)
y_fit = malus_n(x_fit, A_fit, theta_fit, n_fit)

plt.errorbar(x, y, sigma, fmt='o', label="Dati")
plt.plot(x_fit, y_fit, '-', label=f"Fit cos^{n_fit:.2f}")
plt.xlabel("Angolo [°]")
plt.ylabel("Intensità [u.a.]")
plt.title("Fit per stima dell’esponente n")
plt.grid(True)
plt.legend()
plt.show()
