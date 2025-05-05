from helmet_detection import load_helmet_model, is_helmet_detected

def test_load_helmet_model():
    model = load_helmet_model()
    assert model is not None

def test_helmet_inference_on_dummy_image():
    model = load_helmet_model()
    output = is_helmet_detected(model, "tests/dummy_images/two_wheeler1.jpg")
    assert isinstance(output, bool)
