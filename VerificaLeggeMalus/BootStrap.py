import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import shapiro, probplot
import statsmodels.stats.stattools as stattools

# ---------------------------
# MODELLO
# ---------------------------
def malus(x, A, theta0):
    return A * np.cos(np.pi/180 * (x - theta0))**2

# ---------------------------
# LETTURA DATI
# ---------------------------
df = pd.read_table("dati_corretti_gamma.txt", 
                   delim_whitespace=True, 
                   names=["x", "y", "sigma"])

x = df["x"].values
y = df["y"].values
sigma = df["sigma"].values
N = len(x)

# ---------------------------
# FIT ORIGINALE (MLE)
# ---------------------------
p0 = [y.max(), 0]
pars, cov = curve_fit(malus, x, y, sigma=sigma, p0=p0, absolute_sigma=True)
A_fit, theta0_fit = pars
sigma_A, sigma_theta0 = np.sqrt(np.diag(cov))

print("\n===== RISULTATI FIT MLE =====")
print(f"A       = {A_fit:.6g} ± {sigma_A:.6g}")
print(f"theta0  = {theta0_fit:.6g} ± {sigma_theta0:.6g}")

# ---------------------------
# CHI-QUADRO
# ---------------------------
y_teo = malus(x, A_fit, theta0_fit)
chi2 = np.sum(((y - y_teo)/sigma)**2)
chi2_red = chi2 / (N - 2)

print("\n===== TEST DEL CHI-QUADRO =====")
print(f"Chi²        = {chi2:.4f}")
print(f"Chi² ridotto = {chi2_red:.4f}")

# ---------------------------
# RESIDUI
# ---------------------------
residui = y - y_teo
residui_norm = residui / sigma

# ---------------------------
# TABELLA RESIDUI
# ---------------------------
print("\n===== TABELLA DATI + RESIDUI =====")
print("x_deg\t y_corr\t\t y_teo\t\t residuo")
for i in range(N):
    print(f"{x[i]:.0f}\t {y[i]:.10g}\t {y_teo[i]:.10g}\t {residui[i]:.10g}")

# ---------------------------
# PLOT RESIDUI
# ---------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.axhline(0, color="black", lw=1)
plt.scatter(x, residui, color="blue")
plt.title("Residui = y - y_teo")
plt.xlabel("Angolo (°)")
plt.ylabel("Residuo")
plt.grid(True)

plt.subplot(1,2,2)
plt.axhline(0, color="black", lw=1)
plt.scatter(x, residui_norm, color="red")
plt.title("Residui normalizzati")
plt.xlabel("Angolo (°)")
plt.ylabel("(y - y_teo)/σ")
plt.grid(True)

plt.tight_layout()
plt.show()

# ---------------------------
# QQ-PLOT
# ---------------------------
plt.figure(figsize=(6,6))
probplot(residui_norm, dist="norm", plot=plt)
plt.title("QQ-plot dei residui normalizzati")
plt.grid(True)
plt.show()

# ---------------------------
# TEST DI NORMALITÀ (Shapiro–Wilk)
# ---------------------------
W, p_shapiro = shapiro(residui_norm)
print("\n===== TEST DI NORMALITÀ (Shapiro–Wilk) =====")
print(f"W = {W:.4f},  p-value = {p_shapiro:.4f}")
if p_shapiro < 0.05:
    print("✘ I residui NON sono gaussiani (p < 0.05).")
else:
    print("✔ I residui sono compatibili con una gaussiana.")

# ---------------------------
# TEST DI INDIPENDENZA (Durbin–Watson)
# ---------------------------
DW = stattools.durbin_watson(residui_norm)
print("\n===== TEST DI INDIPENDENZA (Durbin–Watson) =====")
print(f"Durbin–Watson = {DW:.4f}")

if DW < 1.5:
    print("✘ Residui positivamente correlati (pattern sistematico).")
elif DW > 2.5:
    print("✘ Residui negativamente correlati (oscillazioni regolari).")
else:
    print("✔ Residui non mostrano forte autocorrelazione.")

# -----------------------------------------------------------
# PLOT FINALE FIT + BANDE DI ERRORE (da matrice di covarianza)
# -----------------------------------------------------------
x_fit = np.linspace(min(x), max(x), 500)

y_fit = malus(x_fit, A_fit, theta0_fit)
y_upper = malus(x_fit, A_fit + sigma_A, theta0_fit + sigma_theta0)
y_lower = malus(x_fit, A_fit - sigma_A, theta0_fit - sigma_theta0)

plt.figure(figsize=(8,6))
plt.errorbar(x, y, sigma, fmt='o', label="Dati")
plt.plot(x_fit, y_fit, 'r-', lw=2, label="Fit MLE")
plt.fill_between(x_fit, y_lower, y_upper, color='red', alpha=0.25, 
                 label="Incertezza parametri (±1σ)")
plt.xlabel("Angolo (°)")
plt.ylabel("Intensità")
plt.title("Fit Malus (MLE) + banda da covarianza")
plt.grid(True)
plt.legend()
plt.show()
