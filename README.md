# OCR-Comparison

Ein Python-Tool zur Bewertung und Visualisierung der Leistung von vier OCR-Engines:
**Tesseract**, **EasyOCR**, **PaddleOCR** und **Surya OCR**.

## Funktionen

- Vergleich von OCR-Tools hinsichtlich Laufzeit, CER & WER
- Visualisierung erkannter Texte im Bild
- Normalisierung & Auswertung gegen Ground Truth

## Voraussetzungen

- Python 3.12
- Optional: GPU für PaddleOCR, EasyOCR und Surya
- Tesseract muss installiert sein und im PATH liegen
- Installation der Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Verwendung

1. Lege Testbilder und passende .txt-Dateien (Ground Truth) im Ordner `input/` ab oder verwende die bereits abgelegten. Achte dabei darauf, dass neue Bilder und Textdateien denselben Namenskonventionen folgen wie die bestehenden Dateien.

   ```
   input/
   ├─ muster_bild.jpg
   └─ muster.txt
   ```

2. Starte das Skript:

   ```bash
   python compare_ocr.py
   ```

3. Ergebnisse findest du in `output/`

## Ausgabe

In der Konsole erscheinen z. B.:

```
Easyocr     : Time: 1.24s | CER: 0.045 | WER: 0.091
Paddle      : Time: 0.98s | CER: 0.031 | WER: 0.060
```

Und in `output/`:

- Annotierte Bilder: `result_<bild>_<tool>.jpg`
- Erkannter Text: `result_<bild>_<tool>.txt`
- Normalisierter erkannter Text: `normalised_<bild>_<tool>.txt`

## OCR-Tools aktivieren

Im Skript konfigurierbar:

```python
ENABLED_OCR_TOOLS = ["surya", "paddle", "easyocr", "tesseract"]
```

## Ergebnisse

Die Testergebnisse sind in einer CSV-Datei un die berechnung der Endergebnisse sind in einer Excel-Datei unter `result/` erfasst. Die Datei `result/ocr_auswertung.py` dient der Erstellung von Plots.
