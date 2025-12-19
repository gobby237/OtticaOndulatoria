import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# 1. MODELLO DI MALUS
# ---------------------------
def malus(x, A, theta0):
    return A * np.cos(np.pi/180 * (x - theta0))**2


# ---------------------------
# 2. LETTURA FILE
# ---------------------------
df = pd.read_table("dati.txt", delim_whitespace=True, names=["x", "y", "sigma"])

x = df["x"].values
y = df["y"].values
sigma = df["sigma"].values

# ---------------------------
# 3. FIT per ottenere il minimo
# ---------------------------
p0 = [y.max(), 0]
pars, cov = curve_fit(malus, x, y, sigma=sigma, p0=p0, absolute_sigma=True)
A_fit, theta0_fit = pars

# ---------------------------
# 4. COSTRUZIONE GRIGLIA PER CHI2
# ---------------------------

A_range = np.linspace(0.8*A_fit, 1.2*A_fit, 200)
theta_range = np.linspace(theta0_fit - 30, theta0_fit + 30, 200)

A_grid, theta_grid = np.meshgrid(A_range, theta_range)
chi2_grid = np.zeros_like(A_grid)

for i in range(A_grid.shape[0]):
    for j in range(A_grid.shape[1]):
        y_model = malus(x, A_grid[i, j], theta_grid[i, j])
        chi2_grid[i, j] = np.sum(((y - y_model)/sigma)**2)

# ---------------------------
# 5. TROVA MINIMO E LIVELLO χ² + 1
# ---------------------------
chi2_min = np.min(chi2_grid)
idx_min = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
A_best = A_grid[idx_min]
theta_best = theta_grid[idx_min]

chi2_target = chi2_min + 1  # criterio Δχ² = 1

# ---------------------------
# 6. STIMA DELLE INCERTEZZE
# ---------------------------

# Tutti i punti con chi² = chi²_min + 1 ± una piccola tolleranza
tol = 0.02  # margine
mask = np.abs(chi2_grid - chi2_target) < tol

A_values = A_grid[mask]
theta_values = theta_grid[mask]

sigma_A = np.abs(A_values - A_best).max()
sigma_theta = np.abs(theta_values - theta_best).max()

# ---------------------------
# 7. PLOT 3D DELLA SUPERFICIE
# ---------------------------
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A_grid, theta_grid, chi2_grid, cmap="viridis", alpha=0.8)
ax.set_xlabel("A")
ax.set_ylabel("theta0 (°)")
ax.set_zlabel("χ²(A, θ₀)")
ax.set_title("Superficie del χ² con fit Malus")
plt.show()

# ---------------------------
# 8. CONTOUR PLOT 2D con livello χ²_min + 1
# ---------------------------
plt.figure(figsize=(8,6))
cs = plt.contour(A_grid, theta_grid, chi2_grid, 
                 levels=[chi2_min, chi2_target], colors=["red", "blue"])

plt.clabel(cs)
plt.scatter([A_best], [theta_best], color="black", label="Minimo globale")
plt.xlabel("A")
plt.ylabel("theta0 (°)")
plt.title("Livelli di χ² – Inclusi χ²_min e χ²_min + 1")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# 9. TABELLA FINALE
# ---------------------------
print("\n=== RISULTATI FINALI (METODO Δχ² = 1) ===")
print(f"Parametro   Valore ± Errore")
print(f"A        = {A_best:.6g} ± {sigma_A:.6g}")
print(f"theta0   = {theta_best:.6g}° ± {sigma_theta:.6g}°")
print(f"χ²_min   = {chi2_min:.6g}")
