from ocr import crop_number_plates_in_memory, recognize_license_numbers
import numpy as np

def test_crop_number_plates_in_memory():
    bboxes = [[10, 10, 100, 100]]
    crops = crop_number_plates_in_memory("tests/dummy_images/plate.jpeg", bboxes)
    assert isinstance(crops, list)
    assert isinstance(crops[0], np.ndarray)

def test_recognize_license_numbers():
    bboxes = [[10, 10, 100, 100]]
    crops = crop_number_plates_in_memory("tests/dummy_images/plate.jpeg", bboxes)
    ocr_results = recognize_license_numbers(crops)
    assert isinstance(ocr_results, list)
