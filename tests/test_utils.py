from utils.drawing_utils import draw_bounding_boxes
from utils.logging_utils import setup_logger
import os

def test_draw_bounding_boxes():
    bboxes = [[10, 10, 100, 100]]
    img = draw_bounding_boxes("tests/dummy_images/plate.jpg", bboxes)
    assert img is not None

def test_setup_logger():
    logger = setup_logger()
    assert logger is not None
