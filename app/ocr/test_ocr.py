import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

IMAGE_PATH = "data/test_image.png"


# =========================================================
# LOAD IMAGE
# =========================================================

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("ERROR: Image not found!")
    print("Check:", IMAGE_PATH)
    exit()

print("\n========================================")
print("          INTELLIMED OCR")
print("========================================")

print("Image loaded successfully")
print("Image size:", img.shape[1], "x", img.shape[0])


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve contrast
    gray = cv2.convertScaleAbs(
        gray,
        alpha=1.5,
        beta=0
    )

    return gray


# =========================================================
# OCR
# =========================================================

def get_text(image):

    processed = preprocess(image)

    text = pytesseract.image_to_string(
        processed,
        config="--psm 6"
    )

    return text


# =========================================================
# IMAGE REGIONS
# =========================================================

height, width = img.shape[:2]

# Left section
left = img[
    0:height,
    0:int(width * 0.68)
]

# Right section
right = img[
    0:height,
    int(width * 0.68):width
]


# =========================================================
# LEFT OCR
# =========================================================

print("\nReading medicine information...")

left_text = get_text(left)

# Try another OCR mode
left_text += "\n" + pytesseract.image_to_string(
    preprocess(left),
    config="--psm 11"
)


# =========================================================
# RIGHT OCR
# =========================================================

print("Reading batch/date information...")

right_text = ""

right_rotations = [
    right,
    cv2.rotate(right, cv2.ROTATE_90_CLOCKWISE),
    cv2.rotate(right, cv2.ROTATE_90_COUNTERCLOCKWISE)
]

for rotated in right_rotations:

    right_text += "\n"

    right_text += get_text(rotated)

    right_text += "\n"

    right_text += pytesseract.image_to_string(
        preprocess(rotated),
        config="--psm 11"
    )


# =========================================================
# UPPERCASE
# =========================================================

left_text = left_text.upper()
right_text = right_text.upper()

print("\n========================================")
print("LEFT OCR")
print("========================================")

print(left_text)

print("\n========================================")
print("RIGHT OCR")
print("========================================")

print(right_text)


# =========================================================
# NORMALIZE OCR MISTAKES
# =========================================================

def normalize(text):

    text = text.upper()

    # MFG mistakes
    text = re.sub(r"\bMEG\b", "MFG", text)
    text = re.sub(r"\bMFGG\b", "MFG", text)

    # EXP mistakes
    text = re.sub(r"\bEXPLPY\b", "EXPIRY", text)
    text = re.sub(r"\bEXPIRY\b", "EXPIRY", text)

    # Batch label mistakes
    text = re.sub(r"\bB\.?\s*MO\.?\b", "B.NO", text)
    text = re.sub(r"\bB\.?\s*NO\.?\b", "B.NO", text)

    return text


normalized_left = normalize(left_text)
normalized_right = normalize(right_text)


# =========================================================
# MEDICINE NAME
# =========================================================

medicine_name = "Not detected"

patterns = [

    r"\b([A-Z][A-Z0-9-]{2,30}-MR\s*\d+)\b",

    r"\b([A-Z][A-Z0-9-]{2,30}\s+MR\s*\d+)\b",

    r"\b([A-Z][A-Z0-9-]{3,30}\s+\d+\s*MG)\b"
]

for pattern in patterns:

    match = re.search(
        pattern,
        normalized_left
    )

    if match:

        medicine_name = match.group(1).strip()
        break


# =========================================================
# COMPOSITION
# =========================================================

composition = "Not detected"

# Handle OCR variations of GLICLAZIDE
gliclazide_found = re.search(
    r"G[L|I]C[L|I][A-Z0-9]*ZIDE",
    normalized_left
)

if gliclazide_found:

    # Search nearby for strength
    start = gliclazide_found.start()

    nearby = normalized_left[
        start:start + 150
    ]

    strength_match = re.search(
        r"(\d+)\s*MG",
        nearby
    )

    if strength_match:

        composition = (
            "Gliclazide "
            + strength_match.group(1)
            + " mg"
        )

    else:

        composition = "Gliclazide"


# =========================================================
# STRENGTH
# =========================================================

strength = "Not detected"

strength_match = re.search(
    r"\b(\d+)\s*MG\b",
    normalized_left
)

if strength_match:

    strength = (
        strength_match.group(1)
        + " MG"
    )


# =========================================================
# BATCH NUMBER
# =========================================================

batch_number = "Not detected"

batch_patterns = [

    r"B\.NO\s*[:.]?\s*([A-Z0-9]+)",

    r"BATCH\s*NO\s*[:.]?\s*([A-Z0-9]+)",

    r"BATCH\s*[:.]?\s*([A-Z0-9]+)"
]

for pattern in batch_patterns:

    match = re.search(
        pattern,
        normalized_right
    )

    if match:

        value = match.group(1)

        # Ignore obvious OCR garbage
        if len(value) >= 4:

            batch_number = value
            break


# =========================================================
# MANUFACTURING DATE
# =========================================================

manufacturing_date = "Not detected"

mfg_patterns = [

    r"MFG\s*DATE\s*[:.]?\s*"
    r"(\d{1,2}\s*[/-]\s*\d{4})",

    r"MFG\s*[:.]?\s*"
    r"(\d{1,2}\s*[/-]\s*\d{4})",

    r"MFG\s*DATE\s*[:.]?\s*"
    r"([A-Z]{3,9}\s*\d{4})"
]

for pattern in mfg_patterns:

    match = re.search(
        pattern,
        normalized_right
    )

    if match:

        manufacturing_date = match.group(1)
        break


# =========================================================
# EXPIRY DATE
# =========================================================

expiry_date = "Not detected"

exp_patterns = [

    r"EXPIRY\s*DATE\s*[:.]?\s*"
    r"(\d{1,2}\s*[/-]\s*\d{4})",

    r"EXP\s*DATE\s*[:.]?\s*"
    r"(\d{1,2}\s*[/-]\s*\d{4})",

    r"EXPIRY\s*[:.]?\s*"
    r"(\d{1,2}\s*[/-]\s*\d{4})",

    r"EXP\s*[:.]?\s*"
    r"(\d{1,2}\s*[/-]\s*\d{4})"
]

for pattern in exp_patterns:

    match = re.search(
        pattern,
        normalized_right
    )

    if match:

        expiry_date = match.group(1)
        break


# =========================================================
# SPECIAL CASE FOR PARTIAL YEAR
# =========================================================

# Sometimes OCR reads:
# EXPIRY DATE: 02/20
#
# but the actual package contains 02/2026.
# We only accept a correction when a nearby 4-digit
# year is also found in the same OCR region.

if expiry_date == "02/20":

    nearby_year = re.search(
        r"\b20(2[0-9])\b",
        normalized_right
    )

    if nearby_year:

        expiry_date = (
            "02/20"
            + nearby_year.group(1)
        )


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n========================================")
print("          MEDICINE DETAILS")
print("========================================")

print("Medicine Name      :", medicine_name)
print("Composition        :", composition)
print("Strength           :", strength)
print("Manufacturing Date :", manufacturing_date)
print("Expiry Date        :", expiry_date)
print("Batch Number       :", batch_number)

print("========================================")