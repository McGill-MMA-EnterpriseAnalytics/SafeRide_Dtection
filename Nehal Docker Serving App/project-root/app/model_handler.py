import torch, easyocr, cv2, numpy as np, re
from torchvision.ops import nms
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])
helmet_model = torch.load("Final_Helmet.pt", map_location="cpu", weights_only=False)['model'].eval()
plate_model = torch.load("Final_Plates.pt", map_location="cpu", weights_only=False)['model'].eval()
reader = easyocr.Reader(['en'], gpu=False)

HELMET_CONF_THRESH = 0.5
PLATE_CONF_THRESH = 0.4
NMS_THRESH = 0.5

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).contiguous().half() / 255.0
    return img_tensor, img_resized

def postprocess(pred_tensor, conf_thresh, nms_thresh):
    if isinstance(pred_tensor, (tuple, list)):
        pred_tensor = pred_tensor[0]
    pred = pred_tensor[0].permute(1, 0)
    boxes, scores = pred[:, :4], pred[:, 4]
    mask = scores > conf_thresh
    boxes, scores = boxes[mask], scores[mask]
    if boxes.shape[0] == 0: return [], []
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    indices = nms(boxes_xyxy.float(), scores.float(), nms_thresh)
    return boxes_xyxy[indices], scores[indices]

def expand_box(x1, y1, x2, y2, img_shape, expand_ratio=0.15):
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(x1 - int(w * expand_ratio), 0), max(y1 - int(h * expand_ratio), 0)
    x2, y2 = min(x2 + int(w * expand_ratio), img_shape[1] - 1), min(y2 + int(h * expand_ratio), img_shape[0] - 1)
    return x1, y1, x2, y2

def preprocess_plate_for_ocr(plate_crop):
    plate_crop = cv2.copyMakeBorder(plate_crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def safe_easyocr_read(plate_crop, full_img=None):
    enhanced = preprocess_plate_for_ocr(plate_crop)
    def clean_text(texts): return re.sub(r'[^A-Z0-9]', '', ''.join(texts).upper()) if texts else ""
    raw, enhanced_text = clean_text(reader.readtext(plate_crop, detail=0)), clean_text(reader.readtext(enhanced, detail=0))
    return enhanced_text if len(enhanced_text) >= len(raw) else (raw or "Unrecognized")

def detect_plate_numbers(image_path):
    offenders = []
    img_tensor, img = preprocess_image(image_path)
    with torch.no_grad(): plates, _ = postprocess(plate_model(img_tensor), PLATE_CONF_THRESH, NMS_THRESH)
    for box in plates:
        x1, y1, x2, y2 = map(int, expand_box(*box.tolist(), img.shape))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue
        offenders.append(safe_easyocr_read(crop, full_img=img))
    return offenders
