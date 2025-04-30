import time
import os
from PIL import Image, ImageDraw, ImageFont
import Levenshtein
import re

# OCR-Engines
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from paddleocr import PaddleOCR
import easyocr
import pytesseract
from pytesseract import Output

# Paths
input_image_path = 'input/ocr_test_01.png'
font_path = 'fonts/arial.ttf'
output_dir = 'output'

os.makedirs(output_dir, exist_ok=True)

# Globale Konfiguration
image = Image.open(input_image_path).convert('RGB')
font_size = 16
line_spacing = 4
line_height = font_size + line_spacing
padding = 20
font = ImageFont.truetype(font_path, size=font_size)

def get_text_properties(text_lines, font):
    dummy_image = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_image)
    max_text_width = 0
    total_height = len(text_lines) * line_height
    for idx, line in enumerate(text_lines, 1):
        text = f"{idx}. {line}"
        dummy_draw.text((10, 10 + idx * line_height), text, fill='black', font=font)
        text_width = dummy_draw.textlength(text, font=font)
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
        draw_new.text((10, base_image.height + i * line_height + padding), line, fill='black', font=font)

    return new_image

def run_surya(image):
    start = time.time()
    recognition_predictor = RecognitionPredictor()
    detection_predictor = DetectionPredictor()
    langs = ['de', 'en']
    result = recognition_predictor([image], [langs], detection_predictor)
    end = time.time()

    polygons = [ [tuple(point) for point in line.polygon] for line in result[0].text_lines ]
    texts = [ line.text for line in result[0].text_lines ]

    image_copy, text_lines, img_text_lines = draw_polygons_and_text(image, polygons, texts, font)
    new_image = create_extended_image_with_text(image_copy, img_text_lines, text_lines, font)

    new_image.save(os.path.join(output_dir, f"result_{remove_suffix(input_image_path)}_surya.jpg"))
    with open(os.path.join(output_dir, f"result_{remove_suffix(input_image_path)}_surya.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

def run_paddle(image_path):
    start = time.time()
    ocr = PaddleOCR(use_angle_cls=True, lang='german')
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

    new_image.save(os.path.join(output_dir, f"result_{remove_suffix(input_image_path)}_paddle.jpg"))
    with open(os.path.join(output_dir, f"result_{remove_suffix(input_image_path)}_paddle.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

def run_easyocr(image_path):
    start = time.time()
    ocr = easyocr.Reader(['de', 'en'], gpu=False)
    result = ocr.readtext(image_path)
    end = time.time()

    polygons = [ [tuple(point) for point in bbox] for (bbox, _, _) in result ]
    texts = [ text for (_, text, _) in result ]

    image_copy, text_lines, img_text_lines = draw_polygons_and_text(image, polygons, texts, font)
    new_image = create_extended_image_with_text(image_copy, img_text_lines, text_lines, font)

    new_image.save(os.path.join(output_dir, f"result_{remove_suffix(input_image_path)}_easy.jpg"))
    with open(os.path.join(output_dir, f"result_{remove_suffix(input_image_path)}_easy.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

def run_tesseract(image_path):
    start = time.time()
    # bessere Ergebnisse mit '--oem 3 --psm 6'
    config = '-l deu+eng'

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

    new_image.save(os.path.join(output_dir, f"result_{remove_suffix(input_image_path)}_tesseract.jpg"))
    with open(os.path.join(output_dir, f"result_{remove_suffix(input_image_path)}_tesseract.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

def remove_suffix(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

# Error Rate Funktionen

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\x20-\x7EäöüÄÖÜß\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_cer(reference, hypothesis):
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(hypothesis).replace(" ", "")
    return Levenshtein.distance(ref, hyp) / max(1, len(ref))

def calculate_wer(reference, hypothesis):
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    ref_str = ' '.join(ref_words)
    hyp_str = ' '.join(hyp_words)
    return Levenshtein.distance(ref_str, hyp_str) / max(1, len(ref_words))

def main():
    start = time.time()
    print("Running OCR comparison...\n")

    with open('input/ground_truth_01.txt', encoding='utf-8') as f:
        ground_truth = f.read().strip()

    surya_time, surya_text = run_surya(image)
    paddle_time, paddle_text = run_paddle(input_image_path)
    easy_time, easy_text = run_easyocr(input_image_path)
    tesseract_time, tesseract_text = run_tesseract(input_image_path)

    # Text zu einem String zusammenfassen
    surya_result = ' '.join(surya_text)
    paddle_result = ' '.join(paddle_text)
    easy_result = ' '.join(easy_text)
    tesseract_result = ' '.join(tesseract_text)

    # Normalisierte Texte speichern
    def save_normalised(name, result):
        normalised = normalize_text(result)
        
        filename = f'output/normalised_{remove_suffix(input_image_path)}_{name}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(normalised)

    save_normalised('surya', surya_result)
    save_normalised('paddle', paddle_result)
    save_normalised('easyocr', easy_result)
    save_normalised('tesseract', tesseract_result)

    # Metriken berechnen
    print(f"Surya OCR:    Time: {surya_time:.2f}s | CER: {calculate_cer(ground_truth, surya_result):.3f} | WER: {calculate_wer(ground_truth, surya_result):.3f}")
    print(f"PaddleOCR:    Time: {paddle_time:.2f}s | CER: {calculate_cer(ground_truth, paddle_result):.3f} | WER: {calculate_wer(ground_truth, paddle_result):.3f}")
    print(f"EasyOCR:      Time: {easy_time:.2f}s | CER: {calculate_cer(ground_truth, easy_result):.3f} | WER: {calculate_wer(ground_truth, easy_result):.3f}")
    print(f"TesseractOCR: Time: {tesseract_time:.2f}s | CER: {calculate_cer(ground_truth, tesseract_result):.3f} | WER: {calculate_wer(ground_truth, tesseract_result):.3f}")

    end = time.time()
    total_time = end - start
    print(f"\nFinished after {total_time: .2f} seconds. Results saved in /output/")

if __name__ == "__main__":
    main()
