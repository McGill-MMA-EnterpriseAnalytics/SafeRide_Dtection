import cv2
import easyocr

def crop_number_plates_in_memory(image_path, bboxes):
    """
    Crops number plates from the image based on bounding boxes (in memory, no save).

    Args:
        image_path (str): Path to the original image
        bboxes (list): List of bounding boxes [x1, y1, x2, y2]

    Returns:
        List of cropped plate images (as numpy arrays)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {image_path}")

    cropped_plates = []

    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        cropped = img[y1:y2, x1:x2]
        cropped_plates.append(cropped)

    return cropped_plates

def recognize_license_numbers(cropped_plates):
    """
    Recognizes license numbers from cropped plate images using EasyOCR.

    Args:
        cropped_plates (list): List of cropped plate images (numpy arrays)

    Returns:
        List of detected license numbers (strings)
    """
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR

    license_numbers = []

    for idx, plate_img in enumerate(cropped_plates):
        # EasyOCR expects file paths OR numpy arrays (BGR)
        results = reader.readtext(plate_img)

        # Extract the text part from results
        plate_text = ""
        for bbox, text, confidence in results:
            plate_text += text + " "

        plate_text = plate_text.strip()

        license_numbers.append(plate_text)

    return license_numbers
