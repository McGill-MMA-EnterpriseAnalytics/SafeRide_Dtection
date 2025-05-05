import cv2
import os

def draw_bounding_boxes(image_path, bboxes, labels=None, output_path=None):
    """
    Draws bounding boxes on an image.

    Args:
        image_path (str): Path to input image
        bboxes (list): List of bounding boxes [x1, y1, x2, y2]
        labels (list, optional): Labels for each box
        output_path (str, optional): If provided, saves annotated image

    Returns:
        Annotated image (numpy array)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {image_path}")

    for idx, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if labels:
            label = labels[idx]
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)

    return img
