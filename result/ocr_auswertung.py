import matplotlib.pyplot as plt
import pandas as pd
import seaborn

file_path = 'result/ocr_messdaten.csv'
df = pd.read_csv(file_path, sep=";", decimal=",")

# Qualitätsstufen aus Dateinamen extrahieren
def extrahiere_qualitaet(dokument_name):
    if "scan" in dokument_name:
        return "Scan"
    elif "unscharf" in dokument_name:
        return "Unscharf"
    elif "bild" in dokument_name:
        return "Foto"
    else:
        return "Unbekannt"

# Neue Spalte für Qualitätsstufe
df["Qualität"] = df["Dokument"].apply(extrahiere_qualitaet)

def plot_metric(df, qualitätsstufe, metric, title, ylabel):
    daten = df[df["Qualität"] == qualitätsstufe]
    plt.figure(figsize=(5, 5))
    seaborn.barplot(data=daten, x="Tool", y=metric)
    plt.title(f"{title} - {qualitätsstufe}")
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

qualitätsstufen = df["Qualität"].unique()

for qualität in qualitätsstufen:
    plot_metric(df, qualität, "Zeit", "Verarbeitungszeit (s)", "Sekunden")
    plot_metric(df, qualität, "CER", "Character Error Rate (CER)", "CER")
    plot_metric(df, qualität, "WER", "Word Error Rate (WER)", "WER")