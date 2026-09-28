import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_path = r"C:\Users\prajw\IntelliMedpr\static\uploads\test2.webp.webp"

img = cv2.imread(image_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.resize(gray, None, fx=2, fy=2)

_, thresh = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

text = pytesseract.image_to_string(
    thresh,
    config="--psm 6"
)

print("========== OCR RESULT ==========")
print(text)
print("================================")