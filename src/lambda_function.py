import os
import json
import boto3
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import logging
from paddleocr import PaddleOCR

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize boto3 client
s3_client = boto3.client('s3')

# Configuration
MODEL_BUCKET = 'is2-project'
HELMET_MODEL_KEY = 'model/best_helmet_detection_model.pt'
PLATE_MODEL_KEY = 'model/best_plate_detection_model.pt'
BUCKET = 'is2-project'

# Initialize PaddleOCR once (only when Lambda container warms up)
ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_box_thresh=0.3)

def download_model_from_s3(model_key):
    """Download model from S3 to temporary file"""
    logger.info(f"Downloading model {model_key}")
    response = s3_client.get_object(Bucket=MODEL_BUCKET, Key=model_key)
    model_data = response['Body'].read()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    temp_file.write(model_data)
    temp_file.close()
    
    return temp_file.name

def process_helmet_detection(image_data, helmet_model_path):
    """Run helmet detection"""
    model = YOLO(helmet_model_path)

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(img)

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            return {
                'helmet_detected': True,
                'detections': [
                    {
                        'class': 1,
                        'confidence': 1.0,
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    } for box in boxes
                ]
            }

    return {'helmet_detected': False, 'detections': []}

def preprocess_plate(plate_crop):
    """Preprocessing for OCR"""
    plate_crop = cv2.copyMakeBorder(plate_crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    target_height = 48
    h, w = plate_crop.shape[:2]
    scaling_factor = target_height / float(h)
    new_w = int(w * scaling_factor)
    plate_crop_resized = cv2.resize(plate_crop, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(plate_crop_resized, cv2.COLOR_BGR2GRAY)
    stretched = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    stretched_color = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
    return stretched_color

def process_plate_detection(image_data, plate_model_path):
    """Run plate detection + OCR"""
    model = YOLO(plate_model_path)

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(img)

    plates = []

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                plate_crop = img[y1:y2, x1:x2]

                # Preprocess for OCR
                preprocessed_plate = preprocess_plate(plate_crop)

                ocr_result_raw = ocr.ocr(plate_crop, cls=True)
                ocr_result_clean = ocr.ocr(preprocessed_plate, cls=True)

                plate_number_raw = ""
                plate_number_clean = ""

                if ocr_result_raw and isinstance(ocr_result_raw[0], list) and len(ocr_result_raw[0]) > 0:
                    plate_number_raw = ocr_result_raw[0][0][1][0]
                if ocr_result_clean and isinstance(ocr_result_clean[0], list) and len(ocr_result_clean[0]) > 0:
                    plate_number_clean = ocr_result_clean[0][0][1][0]

                final_plate_number = plate_number_clean if len(plate_number_clean) >= len(plate_number_raw) else plate_number_raw
                final_plate_number = final_plate_number.strip() if final_plate_number else "N/A"

                plates.append({
                    'plate_number': final_plate_number,
                    'confidence': float(conf[i]),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })

    return plates

def lambda_handler(event, context):
    logger.info("Lambda function started")
    try:
        body = {}
        if 'body' in event and event['body']:
            body = json.loads(event['body'])

        input_prefix = body.get('input_prefix', 'images/')
        output_prefix = body.get('output_prefix', 'results/')

        if not input_prefix.endswith('/'):
            input_prefix += '/'
        if not output_prefix.endswith('/'):
            output_prefix += '/'

        # Load models from S3
        helmet_model_path = download_model_from_s3(HELMET_MODEL_KEY)
        plate_model_path = download_model_from_s3(PLATE_MODEL_KEY)

        paginator = s3_client.get_paginator('list_objects_v2')
        processed_files = []

        for page in paginator.paginate(Bucket=BUCKET, Prefix=input_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if any(key.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                        logger.info(f"Processing image: {key}")

                        response = s3_client.get_object(Bucket=BUCKET, Key=key)
                        image_data = response['Body'].read()

                        # Step 1: Helmet Detection
                        helmet_results = process_helmet_detection(image_data, helmet_model_path)

                        if helmet_results['helmet_detected']:
                            output = {
                                'detection_type': 'helmet',
                                'detections': helmet_results['detections']
                            }
                        else:
                            # Step 2: Plate Detection + OCR
                            plate_results = process_plate_detection(image_data, plate_model_path)
                            output = {
                                'detection_type': 'plate',
                                'detections': plate_results
                            }

                        relative_path = os.path.relpath(key, input_prefix)
                        output_key = os.path.join(output_prefix, os.path.splitext(relative_path)[0] + '.json')

                        s3_client.put_object(
                            Bucket=BUCKET,
                            Key=output_key,
                            Body=json.dumps(output),
                            ContentType='application/json'
                        )

                        processed_files.append({
                            'input_file': key,
                            'output_file': output_key,
                            'detection_type': output['detection_type'],
                            'detections_count': len(output['detections'])
                        })

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': 'Batch processing completed',
                'processed_files': processed_files
            })
        }

    except Exception as e:
        logger.error(f"Lambda function failed: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
