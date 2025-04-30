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

# Common load
image = Image.open(input_image_path).convert('RGB')
font = ImageFont.truetype(font_path, size=16)

def run_surya(image):
    start = time.time()
    recognition_predictor = RecognitionPredictor()
    detection_predictor = DetectionPredictor()
    langs = ['de', 'en']
    result = recognition_predictor([image], [langs], detection_predictor)
    end = time.time()
    
    text_lines = []
    img_text_lines = []
    draw = ImageDraw.Draw(image.copy())

    for idx, line in enumerate(result[0].text_lines, 1):
        polygon = [tuple(point) for point in line.polygon]
        draw.polygon(polygon, outline='red')
        draw.text(polygon[0], str(idx), fill='red', font=font)
        text_lines.append(line.text)
        img_text_lines.append(f"{idx}. {line.text}")

    text_width = get_text_width(text_lines, font)
    new_width = image.width + 20 + text_width

    # Save results
    new_image = Image.new('RGB', (new_width, image.height), color='white')
    new_image.paste(image, (0, 0))
    draw_new = ImageDraw.Draw(new_image)
    for i, line in enumerate(img_text_lines):
        draw_new.text((image.width + 10, 10 + i * 20), line, fill='black', font=font)

    new_image.save(os.path.join(output_dir, 'result_surya.jpg'))
    with open(os.path.join(output_dir, 'result_surya.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

def run_paddle(image_path):
    start = time.time()

    ocr = PaddleOCR(use_angle_cls=True, lang='german')
    results = ocr.ocr(image_path, cls=True)
    end = time.time()

    text_lines = []
    img_text_lines = []
    draw = ImageDraw.Draw(image.copy())

    for result in results: # result enthält die OCR-Ergebnisse für ein Bild
        print(result)
        for idx, line in enumerate(result, 1):  
            box = line[0]
            text = line[1][0]
            text_lines.append(text)
            img_text_lines.append(f"{idx}. {text}")
            polygon = [tuple(point) for point in box]
            draw.polygon(polygon, outline='red')
            draw.text(polygon[0], str(idx), fill='red', font=font)

    text_width = get_text_width(text_lines, font)
    new_width = image.width + 20 + text_width

    new_image = Image.new('RGB', (new_width, image.height), color='white')
    new_image.paste(image, (0, 0))
    draw_new = ImageDraw.Draw(new_image)
    for i, line in enumerate(img_text_lines):
        draw_new.text((image.width + 10, 10 + i * 20), line, fill='black', font=font)

    new_image.save(os.path.join(output_dir, 'result_paddle.jpg'))
    with open(os.path.join(output_dir, 'result_paddle.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines


def run_easyocr(image_path):
    start = time.time()

    ocr = easyocr.Reader(['de','en'], gpu=False)
    result = ocr.readtext(image_path)

    end = time.time()

    draw = ImageDraw.Draw(image.copy())

    text_lines = []
    img_text_lines = []

    for idx, (bbox, text, conf) in enumerate(result, 1):
        text_lines.append(text)
        img_text_lines.append(f"{idx}. {text}")
        polygon = [tuple(point) for point in bbox]
        draw.polygon(polygon, outline='red')
        draw.text(polygon[0], str(idx), fill='red', font=font)

    text_width = get_text_width(text_lines, font)
    new_width = image.width + 20 + text_width

    new_image = Image.new('RGB', (new_width, image.height), color='white')
    new_image.paste(image, (0, 0))
    draw_new = ImageDraw.Draw(new_image)
    for i, line in enumerate(img_text_lines):
        draw_new.text((image.width + 10, 10 + i * 20), line, fill='black', font=font)

    new_image.save(os.path.join(output_dir, 'result_easy.jpg'))
    with open(os.path.join(output_dir, 'result_easy.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

def run_tesseract(image_path):
    start = time.time()

    config = '--oem 3 --psm 6 -l deu+eng'  # OCR Engine Mode 3 = LSTM, Page Segmentation Mode 6 = Assume a block of text
    text = pytesseract.image_to_string(image_path, config=config)

    end = time.time()

    # Split text into lines
    text_lines = text.strip().splitlines()

    text_width = get_text_width(text_lines, font)
    new_width = image.width + 20 + text_width

    # Draw result
    new_image = Image.new('RGB', (new_width, image.height), color='white') #image.width + 1000
    new_image.paste(image, (0, 0))
    draw_new = ImageDraw.Draw(new_image)

    for idx, line in enumerate(text_lines, 1):
        draw_new.text((image.width + 10, 10 + idx * 20), f"{idx}. {line}", fill='black', font=font)

    new_image.save(os.path.join(output_dir, 'result_tesseract.jpg'))
    with open(os.path.join(output_dir, 'result_tesseract.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))

    return end - start, text_lines

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

def main():
    print("Running OCR comparison...\n")
    surya_time, surya_text = run_surya(image)
    paddle_time, paddle_text = run_paddle(input_image_path)
    easy_time, easy_text = run_easyocr(input_image_path)
    tesseract_time, tesseract_text = run_tesseract(input_image_path)

    print(f"Surya OCR completed in {surya_time:.2f} seconds.")
    print(f"PaddleOCR completed in {paddle_time:.2f} seconds.")
    print(f"EasyOCR completed in {easy_time:.2f} seconds.")
    print(f"TesseractOCR completed in {tesseract_time:.2f} seconds.")

    print("\nFinished. Results saved in /output/")

if __name__ == "__main__":
    main()
