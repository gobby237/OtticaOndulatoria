import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# FONT (leggermente più grande per tutto)
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
# LETTURA DATI (x y erry)
# ---------------------------
file_txt = "dati.txt"  # <-- cambia qui il nome del file
df = pd.read_table(file_txt, delim_whitespace=True, names=["x", "y", "erry"])

x = df["x"].values
y = df["y"].values
erry = df["erry"].values

# ---------------------------
# MEDIA ARITMETICA
# ---------------------------
y_mean = np.mean(y)

# ---------------------------
# PLOT CON INCERTEZZE + RETTA MEDIA
# ---------------------------
plt.figure(figsize=(9, 6))

plt.errorbar(x, y, yerr=erry, fmt='o', capsize=3, label="Dati")
plt.axhline(y_mean, linestyle="--", lw=2, label=f"Media = {y_mean:.3g}")

plt.xlabel("Angolo [°]")
plt.ylabel("Intensità [u.a.]")
plt.title("Dati sperimentali con incertezze e media aritmetica")
plt.grid(True)

# Aumenta leggermente la scala in y (margine 10% sopra/sotto)
ymin = np.min(y - erry)
ymax = np.max(y + erry)
pad = 0.10 * (ymax - ymin)
plt.ylim(ymin - pad, ymax + pad)

plt.legend()
plt.tight_layout()
plt.show()
