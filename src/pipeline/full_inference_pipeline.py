from helmet_detection import load_helmet_model, is_helmet_detected
from number_plate_detection import load_number_plate_model, detect_number_plates
from ocr import crop_number_plates_in_memory, recognize_license_numbers

def run_full_inference(image_path):
    """
    Runs the complete helmet + number plate + OCR pipeline.

    Args:
        image_path (str): Path to the input image.

    Returns:
        result (dict): Final detection results.
    """
    # Load models
    helmet_model = load_helmet_model()
    plate_model = load_number_plate_model()

    # Step 1: Check if helmet is worn
    helmet_present = is_helmet_detected(helmet_model, image_path)

    if helmet_present:
        return {
            "helmet_detected": True,
            "license_numbers": [],
            "message": "Helmet detected. No further action needed."
        }
    
    # Step 2: No helmet -> Detect number plates
    bboxes = detect_number_plates(plate_model, image_path)

    if not bboxes:
        return {
            "helmet_detected": False,
            "license_numbers": [],
            "message": "No helmet detected but no number plate found."
        }
    
    # Step 3: Crop plates (in memory)
    cropped_plates = crop_number_plates_in_memory(image_path, bboxes)

    # Step 4: OCR on plates
    license_numbers = recognize_license_numbers(cropped_plates)

    return {
        "helmet_detected": False,
        "license_numbers": license_numbers,
        "message": "No helmet detected. License numbers extracted."
    }

# --- Sample Usage ---
if __name__ == "__main__":
    # Example input
    img_path = "images/sample_biker.jpg"

    results = run_full_inference(img_path)
    
    print("Final Results:", results)
