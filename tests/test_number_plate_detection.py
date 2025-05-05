from number_plate_detection import load_number_plate_model, detect_number_plates

def test_load_number_plate_model():
    model = load_number_plate_model()
    assert model is not None

def test_plate_inference_on_dummy_image():
    model = load_number_plate_model()
    output = detect_number_plates(model, "tests/dummy_images/two_wheeler4.jpg")
    assert isinstance(output, list)
