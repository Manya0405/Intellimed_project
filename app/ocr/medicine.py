def process_medicine_text(text):

    text = text.strip()

    if not text:
        return {
            "medicine": "No medicine detected",
            "text": ""
        }

    # Convert OCR text to lowercase
    lower_text = text.lower()

    # Known medicine names
    known_medicines = [
        "paracetamol",
        "ibuprofen",
        "cetirizine",
        "amoxicillin",
        "omeprazole",
        "azithromycin",
        "loratadine",
        "shelcal",
        "shelcal-500",
        "shelcal 500"
    ]

    # Search for known medicine
    for medicine in known_medicines:

        if medicine in lower_text:

            return {
                "medicine": medicine,
                "text": text
            }

    # If no known medicine is found,
    # use the first non-empty line
    lines = text.splitlines()

    for line in lines:

        line = line.strip()

        if line:
            return {
                "medicine": line,
                "text": text
            }

    return {
        "medicine": "No medicine detected",
        "text": text
    }