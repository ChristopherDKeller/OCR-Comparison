import time
import os
from PIL import Image, ImageDraw, ImageFont
import Levenshtein
import re
import numpy as np

# OCR-Engines
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from paddleocr import PaddleOCR
import easyocr
import pytesseract
from pytesseract import Output

# Paths
font_path = 'fonts/arial.ttf'
output_dir = 'output'

os.makedirs(output_dir, exist_ok=True)

# Globale Konfiguration
font_size = 64
line_spacing = 8
line_height = font_size + line_spacing
padding = 40
font = ImageFont.truetype(font_path, size=font_size)

def get_text_properties(text_lines, font):
    dummy_image = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_image)
    max_text_width = 0
    total_height = len(text_lines) * line_height
    for idx, line in enumerate(text_lines, 1):
        text = f"{idx}. {line}"
        dummy_draw.text((10, 10 + idx * line_height), text, fill='black', font=font)
        try:
            text_width = dummy_draw.textlength(text, font=font)
        except ValueError:
            print(f"Skipping line {idx} due to multiline issue: {repr(text)}")
            continue
        max_text_width = max(max_text_width, text_width)
    
    return int(max_text_width), int(total_height)

def draw_polygons_and_text(image, polygons, texts, font):
    """Zeichnet nummerierte Polygone ins Bild und gibt image copy, text_lines und img_text_lines zurück."""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    text_lines = []
    img_text_lines = []

    for idx, (polygon, text) in enumerate(zip(polygons, texts), 1):
        draw.polygon(polygon, outline='red')

        x = max(p[0] for p in polygon)
        y = min(p[1] for p in polygon)
        draw.text((x, y - font_size), str(idx), fill='red', font=font)
        text_lines.append(text)
        img_text_lines.append(f"{idx}. {text}")

    return draw_image, text_lines, img_text_lines

def create_extended_image_with_text(base_image, img_text_lines, text_lines, font):
    """Erzeugt ein erweitertes Bild mit Text darunter."""
    text_width, text_height = get_text_properties(text_lines, font)

    new_image = Image.new('RGB', (max(text_width, base_image.width), base_image.height + text_height + padding), color='white')
    new_image.paste(base_image, (0, 0))
    draw_new = ImageDraw.Draw(new_image)

    for i, line in enumerate(img_text_lines):
        draw_new.text((40, base_image.height + i * line_height + padding), line, fill='black', font=font)

    return new_image

def run_surya(image_path):
    image = Image.open(image_path).convert('RGB')
    recognition_predictor = RecognitionPredictor()
    detection_predictor = DetectionPredictor()
    langs = ['de', 'en']
    start = time.time()
    result = recognition_predictor([image], [langs], detection_predictor)
    end = time.time()

    polygons = [ [tuple(point) for point in line.polygon] for line in result[0].text_lines ]
    texts = [ line.text for line in result[0].text_lines ]

    image_copy, text_lines, img_text_lines = draw_polygons_and_text(image, polygons, texts, font)
    new_image = create_extended_image_with_text(image_copy, img_text_lines, text_lines, font)

    new_image.save(os.path.join(output_dir, f"result_{remove_suffix(image_path)}_surya.jpg"))
    with open(os.path.join(output_dir, f"result_{remove_suffix(image_path)}_surya.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

def run_paddle(image_path):
    image = Image.open(image_path).convert('RGB')

    file_name = os.path.basename(image_path)
    base_name = file_name.split('_')[0]
    if base_name =='untermietantrag':
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    else:
        ocr = PaddleOCR(use_angle_cls=True, lang='german')

    start = time.time()
    results = ocr.ocr(image_path, cls=True)
    end = time.time()

    boxes = []
    texts = []
    for result in results:
        for line in result:
            box = line[0]
            text = line[1][0]
            boxes.append([tuple(point) for point in box])
            texts.append(text)

    image_copy, text_lines, img_text_lines = draw_polygons_and_text(image, boxes, texts, font)
    new_image = create_extended_image_with_text(image_copy, img_text_lines, text_lines, font)

    new_image.save(os.path.join(output_dir, f"result_{remove_suffix(image_path)}_paddle.jpg"))
    with open(os.path.join(output_dir, f"result_{remove_suffix(image_path)}_paddle.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

def run_easyocr(image_path):
    image = Image.open(image_path).convert('RGB')
    ocr = easyocr.Reader(['de', 'en'], gpu=False)
    start = time.time()
    result = ocr.readtext(image_path)
    end = time.time()

    polygons = [ [tuple(point) for point in bbox] for (bbox, _, _) in result ]
    texts = [ text for (_, text, _) in result ]

    image_copy, text_lines, img_text_lines = draw_polygons_and_text(image, polygons, texts, font)
    new_image = create_extended_image_with_text(image_copy, img_text_lines, text_lines, font)

    new_image.save(os.path.join(output_dir, f"result_{remove_suffix(image_path)}_easy.jpg"))
    with open(os.path.join(output_dir, f"result_{remove_suffix(image_path)}_easy.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

def run_tesseract(image_path):
    image = Image.open(image_path).convert('RGB')
    # bessere Ergebnisse mit '--oem 3 --psm 6'
    config = '-l deu+eng'

    start = time.time()
    data = pytesseract.image_to_data(image_path, config=config, output_type=Output.DICT)
    end = time.time()

    polygons = []
    texts = []

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text and int(data['conf'][i]) > 0:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            polygons.append(polygon)
            texts.append(text)

    image_copy, text_lines, img_text_lines = draw_polygons_and_text(image, polygons, texts, font)
    new_image = create_extended_image_with_text(image_copy, img_text_lines, text_lines, font)

    new_image.save(os.path.join(output_dir, f"result_{remove_suffix(image_path)}_tesseract.jpg"))
    with open(os.path.join(output_dir, f"result_{remove_suffix(image_path)}_tesseract.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

def remove_suffix(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

# Error Rate Funktionen

def normalize_text(text):
    text = text.lower()
    text = text.replace('…', '...')
    # Gepunktete Linien für z.B. Unterschriften entfernen, da Inhaltlich irrelevant
    text = re.sub(r'\.{4,}', ' ', text)
    text = re.sub(r'[^\x20-\x7EäöüÄÖÜß\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_cer(reference, hypothesis):
    # Entferne Leerzeichen, da hier nur Zeichenerkennung und nciht Worttrennung relevant sind
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(hypothesis).replace(" ", "")
    return Levenshtein.distance(ref, hyp) / len(ref)

def calculate_wer(reference, hypothesis):
    # Siehe https://thepythoncode.com/article/calculate-word-error-rate-in-python
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return wer

def ocr(image_path):
    with open(f"{image_path.rsplit('_',1)[0]}.txt", encoding='utf-8') as f:
        ground_truth = f.read().strip()

    results = {}

    if "surya" in ENABLED_OCR_TOOLS:
        surya_time, surya_text = run_surya(image_path)
        result_text = ' '.join(surya_text)
        results["surya"] = (surya_time, result_text)

    if "paddle" in ENABLED_OCR_TOOLS:
        paddle_time, paddle_text = run_paddle(image_path)
        result_text = ' '.join(paddle_text)
        results["paddle"] = (paddle_time, result_text)

    if "easyocr" in ENABLED_OCR_TOOLS:
        easy_time, easy_text = run_easyocr(image_path)
        result_text = ' '.join(easy_text)
        results["easyocr"] = (easy_time, result_text)

    if "tesseract" in ENABLED_OCR_TOOLS:
        tesseract_time, tesseract_text = run_tesseract(image_path)
        result_text = ' '.join(tesseract_text)
        results["tesseract"] = (tesseract_time, result_text)

    # Texte speichern und Metriken berechnen
    for name, (elapsed_time, result_text) in results.items():
        normalized = normalize_text(result_text)
        filename = f'output/normalised_{remove_suffix(image_path)}_{name}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(normalized)

        cer_score = calculate_cer(ground_truth, result_text)
        wer_score = calculate_wer(ground_truth, result_text)
        print(f"{name.title():13}: Time: {elapsed_time:.2f}s | CER: {cer_score:.3f} | WER: {wer_score:.3f}")    

def main():
    start = time.time()
    print("Running OCR comparison...\n")
    input_dir = "input"
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            ocr(f"{input_dir}/{filename}")

    total_time = time.time() - start
    print(f"\nFinished after {total_time:.2f} seconds.")

ENABLED_OCR_TOOLS = [
    "surya",
    "paddle",
    "easyocr",
    "tesseract"
]

if __name__ == "__main__":
    main()
