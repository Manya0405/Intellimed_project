def get_medicine_info(medicine_name):

    medicine_name = medicine_name.lower().strip()

    medicines = {

        "paracetamol": {
            "name": "Paracetamol",
            "use": "Used to reduce fever and relieve pain",
            "form": "Tablet"
        },

        "ibuprofen": {
            "name": "Ibuprofen",
            "use": "Used for pain, fever and inflammation",
            "form": "Tablet"
        },

        "cetirizine": {
            "name": "Cetirizine",
            "use": "Used to relieve allergy symptoms",
            "form": "Tablet"
        },

        "amoxicillin": {
            "name": "Amoxicillin",
            "use": "Antibiotic used for certain bacterial infections",
            "form": "Capsule"
        },

        "omeprazole": {
            "name": "Omeprazole",
            "use": "Used to reduce stomach acid",
            "form": "Capsule"
        },

        "azithromycin": {
            "name": "Azithromycin",
            "use": "Antibiotic used for certain bacterial infections",
            "form": "Tablet"
        },

        "loratadine": {
            "name": "Loratadine",
            "use": "Used to relieve allergy symptoms",
            "form": "Tablet"
        },

        "shelcal": {
            "name": "Shelcal-500",
            "use": "Calcium and Vitamin D3 supplement",
            "form": "Tablet"
        }

    }

    for key in medicines:

        if key in medicine_name:

            return medicines[key]

    return {
        "name": medicine_name,
        "use": "Medicine information not available",
        "form": "Unknown"
    }