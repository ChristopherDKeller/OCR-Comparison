import time
import os
from PIL import Image, ImageDraw, ImageFont
import cv2

# OCR-Engines
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from paddleocr import PaddleOCR, draw_ocr
import easyocr
import pytesseract

# Paths
input_image_path = 'input/ocr_test_01.png'
font_path = 'fonts/arial.ttf'
output_dir = 'output'

os.makedirs(output_dir, exist_ok=True)

# Globale Konfiguration
image = Image.open(input_image_path).convert('RGB')
font = ImageFont.truetype(font_path, size=16)

def get_text_width(text_lines, font):
    dummy_image = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_image)
    max_text_width = 0
    for idx, line in enumerate(text_lines, 1):
        text = f"{idx}. {line}"
        dummy_draw.text((10, 10 + idx * 20), text, fill='black', font=font)
        text_width = dummy_draw.textlength(text, font=font)
        max_text_width = max(max_text_width, text_width)
    return int(max_text_width)

def draw_polygons_and_text(image, polygons, texts, font):
    """Zeichnet nummerierte Polygone ins Bild und gibt image copy, text_lines und img_text_lines zurück."""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    text_lines = []
    img_text_lines = []

    for idx, (polygon, text) in enumerate(zip(polygons, texts), 1):
        draw.polygon(polygon, outline='red')
        draw.text(polygon[0], str(idx), fill='red', font=font)
        text_lines.append(text)
        img_text_lines.append(f"{idx}. {text}")

    return draw_image, text_lines, img_text_lines

def create_extended_image_with_text(base_image, img_text_lines, text_lines, font):
    """Erzeugt ein erweitertes Bild mit Text rechts daneben."""
    text_width = get_text_width(text_lines, font)
    new_width = base_image.width + 20 + text_width

    new_image = Image.new('RGB', (new_width, base_image.height), color='white')
    new_image.paste(base_image, (0, 0))
    draw_new = ImageDraw.Draw(new_image)

    for i, line in enumerate(img_text_lines):
        draw_new.text((base_image.width + 10, 10 + i * 20), line, fill='black', font=font)

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

    new_image.save(os.path.join(output_dir, 'result_surya.jpg'))
    with open(os.path.join(output_dir, 'result_surya.txt'), 'w', encoding='utf-8') as f:
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

    new_image.save(os.path.join(output_dir, 'result_paddle.jpg'))
    with open(os.path.join(output_dir, 'result_paddle.txt'), 'w', encoding='utf-8') as f:
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

    new_image.save(os.path.join(output_dir, 'result_easy.jpg'))
    with open(os.path.join(output_dir, 'result_easy.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

def run_tesseract(image_path):
    start = time.time()
    config = '--oem 3 --psm 6 -l deu+eng'
    text = pytesseract.image_to_string(image_path, config=config)
    end = time.time()

    text_lines = text.strip().splitlines()
    img_text_lines = [f"{idx}. {line}" for idx, line in enumerate(text_lines, 1)]

    new_image = create_extended_image_with_text(image, img_text_lines, text_lines, font)

    new_image.save(os.path.join(output_dir, 'result_tesseract.jpg'))
    with open(os.path.join(output_dir, 'result_tesseract.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

def main():
    start = time.time()
    print("Running OCR comparison...\n")
    surya_time, surya_text = run_surya(image)
    paddle_time, paddle_text = run_paddle(input_image_path)
    easy_time, easy_text = run_easyocr(input_image_path)
    tesseract_time, tesseract_text = run_tesseract(input_image_path)

    print(f"Surya OCR completed in {surya_time:.2f} seconds.")
    print(f"PaddleOCR completed in {paddle_time:.2f} seconds.")
    print(f"EasyOCR completed in {easy_time:.2f} seconds.")
    print(f"TesseractOCR completed in {tesseract_time:.2f} seconds.")

    end = time.time()
    total_time = end - start
    print(f"\nFinished after {total_time: .2f} seconds. Results saved in /output/")

if __name__ == "__main__":
    main()
