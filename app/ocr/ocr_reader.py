import cv2
import pytesseract


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def read_text(image):

    # Make image larger
    image = cv2.resize(
        image,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Remove small noise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Create binary image
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # OCR pass 1
    text1 = pytesseract.image_to_string(
        binary,
        config="--psm 11"
    )

    # OCR pass 2
    text2 = pytesseract.image_to_string(
        gray,
        config="--psm 11"
    )

    # Combine results
    text = text1 + "\n" + text2

    return text.strip()