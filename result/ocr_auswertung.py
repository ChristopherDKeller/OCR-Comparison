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

# Plot-Funktion
def plot_metrics(df, qualitätsstufe):
    daten = df[df["Qualität"] == qualitätsstufe]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"OCR Tool Performance - {qualitätsstufe}", fontsize=16)

    seaborn.barplot(data=daten, x="Tool", y="Zeit", ax=axs[0])
    axs[0].set_title("Verarbeitungszeit (s)")
    axs[0].set_ylabel("Sekunden")

    seaborn.barplot(data=daten, x="Tool", y="CER", ax=axs[1])
    axs[1].set_title("Character Error Rate (CER)")
    axs[1].set_ylabel("CER")

    seaborn.barplot(data=daten, x="Tool", y="WER", ax=axs[2])
    axs[2].set_title("Word Error Rate (WER)")
    axs[2].set_ylabel("WER")

    for ax in axs:
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# Plots
qualitätsstufen = df["Qualität"].unique()
figuren = [plot_metrics(df, qualität) for qualität in qualitätsstufen]
plt.show()
