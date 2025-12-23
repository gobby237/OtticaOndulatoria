import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import probplot

# ---------------------------
# FONT (stessa grandezza di prima)
# ---------------------------
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

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
# FIT INIZIALE (solo per starting values)
# ---------------------------
p0 = [y.max(), 0]
pars_init, _ = curve_fit(malus, x, y, sigma=sigma, p0=p0, absolute_sigma=True)
A_guess, theta_guess = pars_init

print("\n===== VALORI INIZIALI (solo guess) =====")
print("A_guess =", A_guess)
print("theta_guess =", theta_guess)

# ---------------------------
# BOOTSTRAP
# ---------------------------
n_boot = 2000
A_samples = []
theta_samples = []

for _ in range(n_boot):
    idx = np.random.randint(0, N, N)
    x_bs = x[idx]
    y_bs = y[idx]
    sigma_bs = sigma[idx]

    try:
        pars_bs, _ = curve_fit(
            malus, x_bs, y_bs, sigma=sigma_bs,
            p0=[A_guess, theta_guess],
            absolute_sigma=True,
            maxfev=5000
        )
        A_samples.append(pars_bs[0])
        theta_samples.append(pars_bs[1])
    except:
        pass

A_samples = np.array(A_samples)
theta_samples = np.array(theta_samples)

# ---------------------------
# PARAMETRI FINALI: bootstrap
# ---------------------------
A_mean = A_samples.mean()
theta_mean = theta_samples.mean()
A_err = A_samples.std(ddof=1)
theta_err = theta_samples.std(ddof=1)

print("\n===== RISULTATI BOOTSTRAP =====")
print(f"A      = {A_mean:.6g} ± {A_err:.6g}")
print(f"theta0 = {theta_mean:.6g}° ± {theta_err:.6g}")

# ---------------------------
# VALORI TEORICI
# ---------------------------
y_teo = malus(x, A_mean, theta_mean)

# ---------------------------
# RESIDUI
# ---------------------------
residui = y - y_teo
residui_norm = residui / sigma

# ---------------------------
# CHI-QUADRO
# ---------------------------
chi2 = np.sum(((y - y_teo)/sigma)**2)
chi2_red = chi2 / (N - 2)

print("\n===== CHI-QUADRO =====")
print(f"Chi²        = {chi2:.4f}")
print(f"Chi² ridotto = {chi2_red:.4f}")

# ---------------------------
# QQ-PLOT + FIT LINEARE DEI QUANTILI
# ---------------------------
(osm, osr), (slope, intercept, r) = probplot(residui_norm, dist="norm", fit=True)

# calcolo incertezze su slope e intercept (OLS)
X = np.vstack([osm, np.ones_like(osm)]).T
res_lin = osr - (slope * osm + intercept)
s2 = np.sum(res_lin**2) / (len(osm) - 2)
cov_beta = s2 * np.linalg.inv(X.T @ X)
slope_err = np.sqrt(cov_beta[0,0])
intercept_err = np.sqrt(cov_beta[1,1])

print("\n===== ANALISI QQ-PLOT =====")
print(f"Retta stimata: y = ({slope:.4f} ± {slope_err:.4f}) x + ({intercept:.4f} ± {intercept_err:.4f})")

plt.figure(figsize=(6,6))
plt.scatter(osm, osr, color="blue", label="Residui normalizzati")

x_line = np.linspace(min(osm), max(osm), 200)

# retta teorica y = x
plt.plot(x_line, x_line, 'r--', label="Retta teorica y=x")

# retta stimata
plt.plot(x_line, slope*x_line + intercept, 'g-', label="Retta stimata")

plt.title("QQ-plot dei residui normalizzati")
plt.xlabel("Quantili teorici N(0,1)")
plt.ylabel("Quantili empirici")
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------
# FIT + BANDE BOOTSTRAP
# ---------------------------
x_fit = np.linspace(min(x), max(x), 500)
y_fit = malus(x_fit, A_mean, theta_mean)

y_upper = malus(x_fit, A_mean + A_err, theta_mean + theta_err)
y_lower = malus(x_fit, A_mean - A_err, theta_mean - theta_err)

plt.figure(figsize=(9,6))
plt.errorbar(x, y, sigma, fmt='o', label="Dati")
plt.plot(x_fit, y_fit, "r-", lw=2, label="Fit bootstrap")
plt.fill_between(x_fit, y_lower, y_upper, color='red', alpha=0.25, label="Banda (±1σ bootstrap)")
plt.xlabel("Angolo [°]")
plt.ylabel("Intensità [u.a.]")
plt.title("Fit Legge di Malus — metodo Bootstrap")
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------
# ISTOGRAMMI DEI PARAMETRI
# ---------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(A_samples, bins=40, color="C0", alpha=0.7)
plt.axvline(A_mean, color="black", label="media bootstrap")
plt.title("Distribuzione bootstrap di A")
plt.xlabel("A")
plt.legend()

plt.subplot(1,2,2)
plt.hist(theta_samples, bins=40, color="C1", alpha=0.7)
plt.axvline(theta_mean, color="black", label="media bootstrap")
plt.title("Distribuzione bootstrap di θ₀")
plt.xlabel("theta0 [°]")
plt.legend()

plt.tight_layout()
plt.show()
