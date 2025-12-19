import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import f

# ---------------------------
# MODELLI
# ---------------------------

def model_const(x, C):
    return np.full_like(x, C)

def model_sin(x, C, A, theta0):
    return C + A * np.cos(2 * np.pi/180 * x - theta0)


# ---------------------------
# LETTURA DATI
# ---------------------------
df = pd.read_csv("dati.txt", sep="\t", header=None, names=["x", "y", "sigma"])
x, y, sigma = df["x"].values, df["y"].values, df["sigma"].values
N = len(x)

# ---------------------------
# FIT COSTANTE
# ---------------------------
pars_const, cov_const = curve_fit(
    model_const,
    x,
    y,
    sigma=sigma,
    absolute_sigma=True
)
C0 = pars_const[0]
C0_err = np.sqrt(cov_const[0,0])  # errore parametro C

y_const = model_const(x, C0)
SSE_const = np.sum(((y - y_const)/sigma)**2)

# ---------------------------
# FIT SINUSOIDALE
# ---------------------------
p0 = [C0, (y.max()-y.min())/2, 0]

pars_sin, cov_sin = curve_fit(
    model_sin,
    x,
    y,
    p0=p0,
    sigma=sigma,
    absolute_sigma=True,
    maxfev=20000
)

C1, A1, theta1 = pars_sin
C1_err, A1_err, theta1_err = np.sqrt(np.diag(cov_sin))  # errori dei parametri

y_sin = model_sin(x, C1, A1, theta1)
SSE_sin = np.sum(((y - y_sin)/sigma)**2)

# ---------------------------
# TEST F
# ---------------------------
p_const = 1
p_sin = 3

df1 = p_sin - p_const
df2 = N - p_sin

F_value = ((SSE_const - SSE_sin)/df1) / (SSE_sin/df2)
p_value = 1 - f.cdf(F_value, df1, df2)
F_crit = f.ppf(0.95, df1, df2)

# ---------------------------
# RISULTATI
# ---------------------------
print("\n=== RISULTATI FIT ===")
print(f"Costante C      = {C1:.6g} ± {C1_err:.3g}")
print(f"Ampiezza A      = {A1:.6g} ± {A1_err:.3g}")
print(f"Fase theta0     = {theta1:.6g} ± {theta1_err:.3g} rad\n")

print("=== TEST F ===")
print(f"SSE costante    = {SSE_const:.4f}")
print(f"SSE sinusoide   = {SSE_sin:.4f}")
print(f"F-value         = {F_value:.4f}")
print(f"F critico (5%)  = {F_crit:.4f}")
print(f"p-value         = {p_value:.6f}\n")

if F_value > F_crit:
    print(">>> La sinusoide è SIGNIFICATIVA → polarizzazione non circolare.\n")
else:
    print(">>> La sinusoide NON è significativa → polarizzazione circolare compatibile.\n")

# ---------------------------
# GRAFICO
# ---------------------------
x_fit = np.linspace(min(x), max(x), 500)
plt.errorbar(x, y, sigma, fmt="o", label="Dati")
plt.plot(x_fit, model_const(x_fit, C0), lw=2, label="Modello costante")
plt.plot(x_fit, model_sin(x_fit, C1, A1, theta1), lw=2, label="Fit sinusoidale")
plt.grid(True)
plt.xlabel("Angolo (°)")
plt.ylabel("Intensità")
plt.legend()
plt.show()
