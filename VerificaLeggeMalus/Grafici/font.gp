# ----------------------------------------------
# CONFIGURAZIONE FONT E STILE GLOBALE (Da immagine)
# ----------------------------------------------

# --- 1. Titolo del Grafico (22) ---
# Imposta la dimensione del font per il titolo principale.
set title font ",22"

# --- 2. Assi / Etichette (18) ---
# Imposta la dimensione del font per le etichette degli assi (X, Y).
# Le dimensioni del font per xlabel e ylabel sono impostate a 18.
set xlabel font ",18"
set ylabel font ",18"

# --- 3. Dati / Ticks (12) ---
# Imposta la dimensione del font per i valori sugli assi (xtics e ytics).
# La dimensione del font per le tacche degli assi è impostata a 12.
set xtics font ",12"
set ytics font ",12"

# --- 4. Legenda / Chiave (14) ---
# Imposta la dimensione del font per la chiave (key / legenda).
set key font ",14"

# --- 5. Pointsize (ps) (2) ---
# Imposta il Pointsize predefinito (ps) per tutti i dati plottati.
# Questo è il valore standard se non specificato diversamente nel comando plot.
set pointsize 2

# NOTA: Per applicare i font in modo coerente su tutti i terminali
# (soprattutto se salvi in PDF o PNG), potresti aver bisogno di specificare
# il terminale prima di plottare. Esempio:
# set terminal push # Salva le impostazioni attuali del terminale
# set terminal pngcairo font "Arial,12" size 800,600
# set output "nome_file.png"
# ... plot comando ...
# set output
# set terminal pop # Ripristina il terminale precedente

# Esempio di come plottare (decommentare e modificare se necessario):
# plot 'dati.dat' with points title 'Misurazioni'